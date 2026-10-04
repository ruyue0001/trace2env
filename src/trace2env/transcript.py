"""Deterministic terminal transcript scaffolds.

A terminal capture has a deterministic skeleton — the shell prompt, the echoed command lines,
heredoc continuation lines — and uncertain program output in between. The harness can build the
skeleton from what it already knows (the last observed prompt line, the tracked working directory,
the action's keystrokes) and leave only the output to the world-model agent. Nothing here reads the
package: it is pure text construction over the action and the observed prompt.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any

PROMPT_RE = re.compile(r"(?P<user>[A-Za-z_][A-Za-z0-9_.-]*)@(?P<host>[^\s:]+):(?P<cwd>[^\s#$]+)(?P<sigil>[#$])")
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(?P<terminator>[A-Za-z0-9_]+)['\"]?")
CD_RE = re.compile(r"^\s*cd(?:\s+(?P<target>[^\s;&|]+))?\s*(?:$|&&|;|\|\|)")
OUTPUT_PLACEHOLDER = "<output of: {command}>"
FINAL_PROMPT_PLACEHOLDER = "<returned prompt, only if the shell is idle again before the capture>"


def prompt_lines(text: str) -> list[dict[str, str]]:
    """Every shell prompt that starts a line of ``text``, in order (user, host, cwd, sigil, prompt)."""
    found: list[dict[str, str]] = []
    for line in text.splitlines():
        match = PROMPT_RE.match(line.lstrip())
        if match:
            parts = match.groupdict()
            found.append({**parts, "prompt": f"{parts['user']}@{parts['host']}:{parts['cwd']}{parts['sigil']}"})
    return found


CONTROL_KEY_RE = re.compile(r"^(C-[^\s]|M-[^\s]|Escape|Enter|Tab|Up|Down|Left|Right|BSpace|Space|PageUp|PageDown|Home|End)$", re.I)


def prompt_signature(observations: list[str]) -> dict[str, str] | None:
    """The shell prompt's user/host/sigil as the most frequent prompt line across recent captures.

    Robust to a single garbled line (``firoot@host:/app#`` after a stray heredoc terminator): the
    signature that most captures agree on wins; its cwd is taken from the last line carrying it.
    """
    counts: dict[tuple[str, str, str], int] = {}
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for text in observations:
        for line in prompt_lines(text):
            key = (line["user"], line["host"], line["sigil"])
            counts[key] = counts.get(key, 0) + 1
            latest[key] = line
    if not counts:
        return None
    best = max(counts, key=lambda key: (counts[key], len(latest[key]["cwd"]) >= 0))
    return dict(latest[best])


def last_prompt(text: str) -> dict[str, str] | None:
    """The last prompt line in a capture, plus whether the capture ends idle at that prompt."""
    lines = prompt_lines(text)
    if not lines:
        return None
    last = dict(lines[-1])
    tail = text.rstrip().splitlines()[-1].strip() if text.strip() else ""
    last["idle_at_end"] = str(bool(PROMPT_RE.fullmatch(tail) or PROMPT_RE.match(tail) and tail.endswith(last["sigil"])))
    return last


def initial_prompt_state(observation: str) -> dict[str, str] | None:
    """A ``Current State`` screen that is only an idle prompt yields the working directory deterministically."""
    stripped = observation.strip()
    if not stripped:
        return None
    lines = [line for line in stripped.splitlines() if line.strip()]
    match = PROMPT_RE.fullmatch(lines[-1].strip())
    if match is None or len(lines) > 3:
        return None
    parts = match.groupdict()
    return {**parts, "prompt": f"{parts['user']}@{parts['host']}:{parts['cwd']}{parts['sigil']}"}


def resolve_cwd(current: str, target: str | None) -> str:
    if target is None or target == "~":
        return "/root"
    if target.startswith("~/"):
        target = "/root/" + target[2:]
    if target.startswith("-"):
        return current
    target = target.strip("'\"")
    joined = target if target.startswith("/") else posixpath.join(current or "/", target)
    return posixpath.normpath(joined) or "/"


def command_lines(arguments: dict[str, Any]) -> list[str] | None:
    """The typed lines of a shell action (None for waits, key presses, or non-terminal actions)."""
    if "command" in arguments and isinstance(arguments["command"], str):
        return [arguments["command"]]
    if isinstance(arguments.get("commands"), list):
        return [str(line) for line in arguments["commands"]]
    keystrokes = arguments.get("keystrokes")
    if isinstance(keystrokes, list):
        text = "".join(str(entry.get("keystrokes", "")) for entry in keystrokes if isinstance(entry, dict))
        if text.strip() and text.endswith("\n"):
            return [line for line in text.split("\n") if line.strip()]
    return None


def wait_hint(previous_observation: str, *, tail_chars: int = 400) -> dict[str, Any]:
    """What a capture without keystrokes (a wait or a key press) continues from.

    The previous capture either ended at an idle prompt (nothing pending), at a prompt followed by a
    typed command that has not printed anything yet (this capture starts with that command's output,
    Terminus often repeats the command line first), or inside a running program's output (this
    capture continues that output; a prompt returns only if the program finishes).
    """
    stripped = previous_observation.rstrip("\n")
    lines = [line for line in stripped.splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    match = PROMPT_RE.match(last)
    if match and not last[match.end():].strip():
        state, pending = "idle_prompt", None
    elif match:
        state, pending = "command_typed_no_output_yet", last[match.end():].strip()
    else:
        state, pending = "program_output_in_progress", None
    guidance = {
        "idle_prompt": "The shell was idle: nothing was pending, so the capture is empty unless a background job prints.",
        "command_typed_no_output_yet": "The command below had produced no output yet: this capture shows its output (Terminus "
                                       "captures often repeat the prompt+command line first), then the prompt if it finished.",
        "program_output_in_progress": "A program was still printing: this capture continues its output from where the previous "
                                      "capture stopped; the prompt returns only if it finishes within the wait.",
    }[state]
    return {"previous_capture_state": state, "pending_command": pending, "previous_capture_tail": stripped[-tail_chars:],
            "guidance": guidance + " No keystrokes are typed now, so no new command is echoed and none may be invented."}


def build_scaffold(arguments: dict[str, Any], prompt: dict[str, str] | None, cwd: str | None) -> dict[str, Any] | None:
    """Skeleton of the capture for a shell action: prompt + echo per command, heredoc bodies as ``> `` lines.

    ``prompt`` is the last observed prompt line (user/host/sigil); ``cwd`` the working directory at the
    start of the action (falls back to the prompt's). Output placeholders mark what the agent must
    produce. Returns None when the action types no command (wait, key presses).
    """
    lines = command_lines(arguments)
    if not lines:
        return None
    user = (prompt or {}).get("user", "root")
    host = (prompt or {}).get("host")
    sigil = (prompt or {}).get("sigil", "#")
    current = cwd or (prompt or {}).get("cwd") or "/"

    def prompt_text(directory: str) -> str:
        return f"{user}@{host}:{directory}{sigil}" if host else f"<prompt for {directory}>"

    skeleton: list[str] = []
    notes: list[str] = []
    terminator: str | None = None
    commands = 0
    for line in lines:
        if terminator is not None:
            skeleton.append(f"> {line}")
            if line.strip() == terminator:
                terminator = None
                skeleton.append(OUTPUT_PLACEHOLDER.format(command="the heredoc command above (nothing for cat > file)"))
            continue
        if CONTROL_KEY_RE.match(line.strip()):
            # A key press is not echoed as a command; it goes to whatever owns the terminal.
            skeleton.append(f"<key press {line.strip()} sent to the foreground program; ^C may appear, no command echo>")
            notes.append(f"{line.strip()} is a key press: if a program was running it is interrupted before the next command")
            continue
        skeleton.append(f"{prompt_text(current)} {line}")
        commands += 1
        heredoc = HEREDOC_RE.search(line)
        if heredoc:
            terminator = heredoc.group("terminator")
            continue
        skeleton.append(OUTPUT_PLACEHOLDER.format(command=line if len(line) <= 80 else line[:77] + "..."))
        cd = CD_RE.match(line)
        if cd:
            new = resolve_cwd(current, cd.group("target"))
            if new != current:
                notes.append(f"after `{line.strip()}` later prompts show {new} if the cd succeeded (else {current} stays)")
                current = new
    if terminator is not None:
        notes.append(f"heredoc terminator {terminator!r} was not typed: the shell is still waiting for input (no returned prompt)")
    notes.append(f"the capture ends with the returned prompt `{prompt_text(current)}` only if the last command finished before it")
    if not host:
        notes.append("the shell prompt's user@host was not observed; copy it from the most recent observation")
    return {"prompt": prompt_text(current), "cwd": current, "commands": commands, "skeleton": "\n".join(skeleton), "notes": notes,
            "status": "expected structure when the shell was idle at the last prompt; earlier captures are the authority "
                      "for the exact prompt bytes, and a program that owns the terminal (editor, pager, REPL, tmux, a "
                      "password prompt) makes this skeleton inapplicable"}
