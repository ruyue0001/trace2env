"""Harness v5.2: compatibility-gated package knowledge.

The reconstructed package carries raw turns, demonstrations, and notes from *other* episodes (other task containers).
Matching commands, file names, or operands make such an item relevant, never proof that its observation applies here:
the current trajectory stays authoritative, and tracked state is used conservatively (only definite, transcript-backed
values count). Every package item shown to the world-model agent, in the initial brief or in a tool result, receives an
applicability decision (label, provenance, reason, disposition), and the model-facing view of an item whose concrete
values are unsafe is a *sanitized* copy: container-specific tokens (hosts, unknown paths and file names, sizes, dates,
versions, hashes) become typed placeholders so the output shape survives and the facts cannot be copied. Raw package
files are never changed; only the views are. When no item supports the transition, the gate abstains and says so.

Labels: ``supporting`` (same-family signals, compatible with this episode; shown intact), ``uncertain`` (a single weak
relevance signal; sanitized), ``format_only`` (no shared specifics; sanitized), ``contradicted`` (a note whose stated
convention conflicts with what this episode's transcript already exhibits; rejected), ``untested`` / ``consistent`` for
notes without or with a passing transcript check (shown).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from trace2env.models import EnvironmentNote, EnvironmentState, LocalTransitionEvidence, NormalizedAction

# Names that identify nothing about a particular container: never a relevance signal, never masked.
GENERIC_NAMES = {
    "app", "tmp", "opt", "usr", "bin", "sbin", "etc", "var", "home", "root", "dev", "proc", "sys", "lib", "lib64",
    "mnt", "srv", "workspace", "work", "data", "src", "test", "tests", "build", "dist", "node_modules", "venv",
    ".venv", ".git", "__pycache__", "local", "share", "include", "logs", "log", "output", "outputs", "input", "inputs",
    "readme.md", "readme", "requirements.txt", "setup.py", "main.py", "app.py", "test.py", "makefile", "dockerfile",
    "package.json", "pyproject.toml", "solution.txt", "answer.txt", "output.txt", "result.txt", "results.txt",
    "input.txt", "config.json", "config.yaml", "config.yml", ".gitignore", "license", "index.html", "main.c",
    "main.cpp", "script.sh", "run.sh", "build.sh", "test.sh", "python3", "python", "pip", "pip3", "apt-get", "ls",
    "cat", "cd", "mkdir", "rm", "cp", "mv", "git", "gcc", "make", "bash", "sh", "2>&1", "/dev/null", "dev/null",
    "/usr/bin/python3", "/usr/bin/python", "/usr/local/bin", "/usr/bin", "/usr/lib", "/usr/local", "/bin/bash",
    "/bin/sh", "python3.10", "python3.11", "python3.12", "python3.13", "site-packages", "dist-packages",
    # v5.3.1: directory words of every project and the dot-files of every container's home directory
    "cache", ".cache", "config", ".config", "configs", "default", "defaults", "application", "latest", "current",
    "health", "live", "containers", "images", "volumes", "static", "templates", "assets", "public", "private", "user",
    "users", "project", "projects", "source", "target", "result", "results", "temp", "docs", "doc", "examples",
    "example", "scripts", "script", "notebooks", "models", "model", "checkpoints", "weights", "settings", "conf",
    "secrets", "credentials", ".bashrc", ".bash_logout", ".bash_history", ".profile", ".ssh", ".dockerenv", ".lesshst",
    ".local", ".npm", ".gitconfig", ".wget-hsts", ".python_history", "id_rsa", "id_rsa.pub", "known_hosts",
    "authorized_keys",
}
# v5.3.1: tokens that are not file names although they contain a slash: URLs (stripped before tokenizing), host:port
# and IP path fragments (`2375/containers/json`, `169.254.169.254/latest/meta-data`), protocol versions (`HTTP/1.1`)
# and media types (`application/json`).
URL = re.compile(r"\w+://\S+")
PAGE_URL = re.compile(r"Page URL:\s*(\S+)")
PROXY_PARAM = re.compile(r"[?&]proxy=[^&\s]*")


SCREEN_STATE_ID = re.compile(r"\*\*State ID:\*\*\s*(\S+)")
SCREEN_APP = re.compile(r"\*\*App:\*\*\s*(?:[^\n(]*\()?([a-zA-Z][\w.]*)\)?")
RESOURCE_ID = re.compile(r"\b([a-zA-Z][\w.]*:id/[A-Za-z0-9_]+)\b")
SYSTEM_PACKAGES = {"android", "com.android.systemui", "com.google.android.apps.nexuslauncher", "com.google.android.inputmethod.latin",
                   "com.android.launcher3", "com.android.internal.app", "com.android.intentresolver", "com.android.permissioncontroller"}
SCREEN_MIN_IDS = 3
SCREEN_JACCARD = 0.8


def screen_signature(text: str | None) -> tuple[str, Any] | None:
    """v5.3.3 (Android): the identity of a UI screen in a state or observation text. A DroidBot-style ``State ID`` names
    the screen directly; otherwise the foreground app (the ``App:`` line, else the app whose resource ids dominate) and
    the set of app resource ids on screen (``res="pkg:id/name"`` or quoted ``pkg:id/name``; ids of system packages such
    as the status bar or the launcher are ignored); screens with fewer than three app resource ids (a launcher, a
    keyboard, web content) have no identity."""
    if not text:
        return None
    state = SCREEN_STATE_ID.search(text)
    if state:
        return ("state", state.group(1).strip())
    # system-package ids (status bar, launcher, keyboard, share sheet) appear on every screen and identify none
    ids = {i for i in RESOURCE_ID.findall(text) if i.split(":id/")[0] not in SYSTEM_PACKAGES}
    if len(ids) < SCREEN_MIN_IDS:
        return None
    app_match = SCREEN_APP.search(text)
    app = app_match.group(1) if app_match else None
    if not app or app in SYSTEM_PACKAGES:
        counts = Counter(i.split(":id/")[0] for i in ids)
        app = counts.most_common(1)[0][0]
    return (app, frozenset(ids))


def same_screen(a: tuple[str, Any] | None, b: tuple[str, Any] | None) -> bool:
    if a is None or b is None or a[0] != b[0]:
        return False
    if a[0] == "state":
        return a[1] == b[1]
    x, y = a[1], b[1]
    return len(x & y) / max(1, len(x | y)) >= SCREEN_JACCARD


def normalize_page(url: str | None) -> str | None:
    """A page identity for the v5.3.2 gate: scheme and host lowercased, the benchmark's per-worker `proxy` query
    parameter and fragments removed, no trailing slash. Two observations of the same page normalize to one key."""
    if not url or not isinstance(url, str):
        return None
    text = PROXY_PARAM.sub("", url.strip().split("#", 1)[0])
    text = text.replace("?&", "?").rstrip("?&")
    match = re.match(r"^(\w+://[^/]+)(.*)$", text)
    if match:
        text = match.group(1).lower() + match.group(2)
    return text.rstrip("/") or None
NON_PATH_SLASH = re.compile(r"^(?:\d[\d.]*/|HTTPS?/\d|(?:application|text|image|audio|video|multipart|message|font|model)/[\w.+-]+$)", re.I)
# Dotted tokens without a slash count as file names only with a file extension from this list (v5.3): `np.abs`,
# `json.load`, `time.sleep`, `f.write` are code attributes, not files.
FILE_EXTENSIONS = {
    "txt", "md", "rst", "py", "pyc", "ipynb", "c", "h", "cc", "cpp", "hpp", "cxx", "hh", "cs", "java", "class", "jar", "go",
    "rs", "rb", "php", "pl", "pm", "lua", "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd", "js", "jsx", "ts", "tsx", "mjs",
    "cjs", "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "config", "env", "xml", "html", "htm", "css", "scss",
    "sql", "db", "sqlite", "sqlite3", "csv", "tsv", "parquet", "npy", "npz", "pkl", "pickle", "pt", "pth", "ckpt",
    "safetensors", "onnx", "h5", "hdf5", "bin", "dat", "log", "out", "err", "pem", "key", "crt", "cer", "pub", "gz", "tgz",
    "tar", "zip", "xz", "bz2", "7z", "deb", "rpm", "whl", "egg", "so", "a", "o", "dylib", "dll", "exe", "pdf", "png", "jpg",
    "jpeg", "gif", "svg", "bmp", "ico", "mp3", "mp4", "wav", "tex", "bib", "r", "rmd", "m", "mat", "jl", "scala", "kt",
    "swift", "dart", "ex", "exs", "erl", "hs", "ml", "mli", "v", "vo", "sv", "vhd", "red", "cmake", "mk", "in", "am", "ac",
    "m4", "lock", "sum", "mod", "cabal", "gradle", "sbt", "proto", "thrift", "graphql", "patch", "diff", "service",
    "socket", "timer", "desktop", "list", "asm", "s", "wasm", "ttf", "otf", "woff", "vim", "el", "org", "adoc", "bak",
}
STANDARD_HEADERS = {
    "stdio.h", "stdlib.h", "string.h", "math.h", "stdint.h", "stddef.h", "stdbool.h", "unistd.h", "errno.h", "time.h",
    "ctype.h", "assert.h", "limits.h", "float.h", "signal.h", "fcntl.h", "pthread.h", "python.h", "arrayobject.h",
    "iostream", "vector", "string", "algorithm", "map", "set", "memory", "numpy/arrayobject.h",
}
TOKEN = re.compile(r"[A-Za-z0-9_~][\w./+~-]*")
IDENTIFIER = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}|[0-9a-f]{16,64}")
ACTION_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
HOST_PROMPT = re.compile(r"\b([A-Za-z0-9_-]+)@([A-Za-z0-9][A-Za-z0-9.-]*)")
PROMPT_LINE = re.compile(r"[A-Za-z0-9_-]+@[A-Za-z0-9.-]+:([^\s#$]*)[#$]")
PROMPT_COMMAND = re.compile(r"^(?:\(\S+\)\s*)?[A-Za-z0-9_-]+@[A-Za-z0-9.-]+:[^\s#$]*[#$]\s+(.+)$")
DOMAIN_LIKE = re.compile(r"\.(?:com|org|net|io|co|edu|gov|dev|ai)(?:/|$)", re.I)
ABBREVIATIONS = {"i.e", "e.g", "etc", "vs", "a.k.a"}
KEYSTROKES = re.compile(r'"keystrokes":\s*"((?:[^"\\]|\\.)*)"')
LONG_LISTING = re.compile(r"^([-dlcbps][-rwxsStT]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+)(\d+)(\s+\w{3}\s+\d{1,2}\s+[\d:]{4,5}\s+)(.+)$")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
HEX_ID = re.compile(r"\b[0-9a-f]{12,40}\b")
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
VERSION = re.compile(r"\b\d+\.\d+(?:\.\d+)+(?:[+~-][\w.+~-]*)?\b|\b\d+\.\d+[+~-][\w.+~-]+\b")
DATE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:\d{2}:\d{2}|\d{4})\b")
LARGE_NUMBER = re.compile(r"(?<![\w.])\d{3,}(?![\w.])")
TOTAL_LINE = re.compile(r"^total \d+$", re.M)
CONTINUATION_BARE = re.compile(r"^>$", re.M)
CONTINUATION_SPACE = re.compile(r"^> $", re.M)
COMMAND_NOT_FOUND = re.compile(r"^bash: .+: command not found$", re.M)


def _strip(token: str) -> str:
    return token.strip("`'\"()[]{}<>,;:!?").rstrip(".")


def specific_tokens(text: str) -> set[str]:
    """Tokens that identify concrete files or directories: paths, names with extensions, non-generic directory
    operands, plus their base names and non-generic path components. Numbers, versions, flags, and generic names are
    excluded, so the result is a relevance signal about *this* task's files rather than about shell vocabulary."""
    out: set[str] = set()
    for raw in TOKEN.findall(URL.sub(" ", text or "")):
        token = _strip(raw)
        if len(token) < 3 or token.startswith("-") or token.lower() in GENERIC_NAMES:
            continue
        if IDENTIFIER.fullmatch(token):  # a UUID or long hex id names one object of one workspace
            out.add(token)
            continue
        if re.fullmatch(r"[\d./+~-]+", token):
            continue
        directory_operand = token.endswith("/")
        token = token.rstrip("/")
        if token.startswith("./"):
            token = token[2:]
        if not token or token.lower() in GENERIC_NAMES:
            continue
        if token.lower() in ABBREVIATIONS or DOMAIN_LIKE.search(token) or NON_PATH_SLASH.match(token):
            continue
        has_slash = "/" in token
        ext = re.search(r"\.([A-Za-z0-9]{1,6})$", token)
        has_ext = ext is not None and not token.startswith(".") and ext.group(1).lower() in FILE_EXTENSIONS
        if not (has_slash or has_ext or directory_operand):
            continue
        if token.lower() in STANDARD_HEADERS or token.lower().rsplit("/", 1)[-1] in STANDARD_HEADERS:
            continue
        parts = [p for p in token.split("/") if p]
        if has_slash and all(p.lower() in GENERIC_NAMES or re.fullmatch(r"[\d.+~-]+", p) for p in parts):
            continue  # a path made only of generic components (app/result.txt) names nothing specific
        out.add(token)
        for part in parts:
            if len(part) >= 4 and part.lower() not in GENERIC_NAMES and not re.fullmatch(r"[\d.+~-]+", part):
                if "." not in part or re.search(r"\.([A-Za-z0-9]{1,6})$", part) and re.search(r"\.([A-Za-z0-9]{1,6})$", part).group(1).lower() in FILE_EXTENSIONS:
                    out.add(part)
    return out


def documented_format(system_text: str, action_type: str) -> bool:
    """Whether the official input documents this action's observation format: a tool-call example naming the action
    (``"name": "<action>"``) followed by its ``Environment Observation``. Tool-environment prompts carry such examples
    per tool; terminal prompts (keystroke examples, no ``name`` field) do not, so this is inert there."""
    if not system_text or not action_type:
        return False
    pattern = re.compile(r'"name"\s*:\s*"' + re.escape(action_type) + r'"')
    return any("Environment Observation" in system_text[m.end(): m.end() + 4000] for m in pattern.finditer(system_text))


def _command_line(text: str) -> str:
    """The command line proper: the first line of a command string (heredoc and script bodies are content)."""
    return str(text).split("\n", 1)[0]


OPERAND_KEYS = {"commands", "command", "argv", "program", "path", "paths", "file", "files", "args", "keystrokes"}
PAYLOAD_KEYS = {"content", "contents", "body", "text", "data", "file_content", "new_str", "old_str", "new_string", "old_string",
                "message", "html", "markdown", "code", "script", "edits", "children", "properties", "value", "values"}
OPERAND_CHARS = 300  # an operand string beyond this length is content, not a name


def action_text(action: NormalizedAction | dict[str, Any]) -> str:
    """The operand-bearing text of an action: command lines and argv, never heredoc or script bodies."""
    data = action if isinstance(action, dict) else action.model_dump(mode="json")
    arguments = data.get("arguments") or {}
    pieces: list[str] = [str(data.get("type") or "")]
    for key in ("commands", "command", "argv", "program", "path", "paths", "file", "files", "args"):
        value = arguments.get(key)
        if isinstance(value, list):
            pieces.extend(_command_line(v) for v in value)
        elif value is not None:
            pieces.append(_command_line(value))
    keystrokes = arguments.get("keystrokes")
    if isinstance(keystrokes, list):
        pieces.extend(_command_line(k.get("keystrokes", "")) if isinstance(k, dict) else _command_line(k) for k in keystrokes)
    elif isinstance(keystrokes, str):
        pieces.append(_command_line(keystrokes))
    # Tool-call arguments (MCP and similar environments): every other string or list-of-strings argument is an
    # operand (a path, an identifier, a query, a name), except payloads, which are content.
    for key, value in arguments.items():
        if key in OPERAND_KEYS or key in PAYLOAD_KEYS:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.strip():
                pieces.append(_command_line(item)[:OPERAND_CHARS])
    return "\n".join(pieces)


def transcript_names(text: str) -> set[str]:
    """Names a transcript establishes as this episode's own: operands of the commands typed after a prompt, and the
    entries of long listings. Free text (file contents, program output) is not a source of names."""
    names: set[str] = set()
    for line in (text or "").splitlines():
        stripped = line.rstrip()
        match = PROMPT_COMMAND.match(stripped)
        if match:
            names.update(specific_tokens(match.group(1)))
            continue
        listing = LONG_LISTING.match(stripped)
        if listing:
            base = _strip(listing.group(4).split(" -> ")[0])
            names.update(specific_tokens(base) or ({base} if base and base.lower() not in GENERIC_NAMES and len(base) >= 4 else set()))
    for block in ACTION_BLOCK.findall(text or ""):  # tool calls shown as ```json {"name": ..., "arguments": {...}}```
        try:
            call = json.loads(block)
        except ValueError:
            continue
        if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
            names.update(specific_tokens(action_text({"type": call.get("name") or "", "arguments": call["arguments"]})))
    return names


def _fact_paths(facts: list[Any]) -> set[str]:
    """File paths named by evidence facts and preconditions (``world.files`` keys and path-valued facts)."""
    out: set[str] = set()
    for fact in facts or []:
        value = getattr(fact, "value", None)
        subject = str(getattr(fact, "subject", ""))
        if isinstance(value, dict) and subject.startswith("world."):
            # keys of any world map: file paths, and in tool environments page ids, table names, repository names
            out.update(specific_tokens(" ".join(str(k) for k in value.keys())))
        elif isinstance(value, str) and "/" in value and " " not in value.strip() and "\n" not in value:
            out.update(specific_tokens(value))  # a path-valued fact; free text is not a source of names
        if subject.startswith("world.files.") or subject.startswith("world.files/"):
            out.update(specific_tokens(subject[len("world.files"):].lstrip("./")))
    return out


def transcript_cwd(text: str) -> str | None:
    """The working directory of the last shell prompt line in a transcript, when one is visible."""
    found = PROMPT_LINE.findall(text or "")
    return found[-1] if found else None


@dataclass
class EpisodeContext:
    """What this episode has established: its own specific names, its cwd, and the conventions its transcript shows."""

    shown_names: set[str] = field(default_factory=set)
    cwd: str | None = None
    continuation_style: str | None = None  # "bare" (`>`), "space" (`> `), or None when no blank continuation shown
    has_command_not_found: bool | None = None  # None: no such line; True: bash shape; False: another shape
    has_long_listing: bool = False
    transcript: str = ""  # the episode's own history text (for the judge's anchor verification and its summary)
    current_page: str | None = None  # v5.3.2: the page the current state shows (normalized), when the transcript has one
    current_page_raw: str | None = None  # the same URL as the transcript prints it (for reasons the audit can verify)
    shown_pages: set[str] = field(default_factory=set)  # normalized pages the episode's own history visited
    current_screen: tuple[str, Any] | None = None  # v5.3.3: the screen the current state shows (Android)


def build_episode_context(texts: list[str], state: EnvironmentState | None = None, *, current_text: str | None = None) -> EpisodeContext:
    """Context from the episode's own transcript (official messages or memory observations) and, conservatively,
    its tracked state: file paths the state declares count as shown; the cwd comes from the last visible prompt line
    and only falls back to the tracked cwd when no prompt is visible."""
    joined = "\n".join(t for t in texts if t)
    names = transcript_names(joined)
    # v5.3.2 page identity: every page the history showed, and the page the current state shows (the current message
    # is not part of the history context, so its page is taken from ``current_text`` only).
    shown_pages = {p for p in (normalize_page(u) for u in PAGE_URL.findall(joined)) if p}
    current_raw = (PAGE_URL.findall(current_text or "") or [None])[-1]
    current_state_text = (current_text or "").split("**Action:**")[0]
    for t in texts:  # action messages (no prompt line): the typed command lines, decoded from the keystroke JSON
        if t and not PROMPT_LINE.search(t):
            for keystrokes in KEYSTROKES.findall(t):
                try:
                    typed = json.loads(f'"{keystrokes}"')
                except ValueError:
                    typed = keystrokes
                names.update(specific_tokens("\n".join(_command_line(line) for line in typed.splitlines() if line.strip())))
    if state is not None:
        files = state.world.get("files") if isinstance(state.world, dict) else None
        if isinstance(files, dict):
            names.update(specific_tokens(" ".join(str(k) for k in files.keys())))
    cwd = transcript_cwd(joined)
    if cwd is None and state is not None:
        value = state.session.get("cwd") if isinstance(state.session, dict) else None
        cwd = value if isinstance(value, str) and value else None
    style = None
    if CONTINUATION_BARE.search(joined):
        style = "bare"
    elif CONTINUATION_SPACE.search(joined):
        style = "space"
    cnf: bool | None = None
    if "command not found" in joined:
        cnf = COMMAND_NOT_FOUND.search(joined) is not None
    return EpisodeContext(shown_names=names, cwd=cwd, continuation_style=style, has_command_not_found=cnf,
                          has_long_listing=bool(re.search(r"^total \d+$", joined, re.M)), transcript=joined,
                          current_page=normalize_page(current_raw), current_page_raw=current_raw, shown_pages=shown_pages,
                          current_screen=screen_signature(current_state_text))


@dataclass
class GateDecision:
    id: str
    kind: str  # evidence | demonstration | note
    label: str  # supporting | uncertain | format_only | contradicted | consistent | untested
    disposition: str  # shown | sanitized | rejected
    reason: str
    provenance: dict[str, Any] = field(default_factory=dict)
    masked: int = 0
    judge: dict[str, Any] | None = None  # v5.3: the judge's verdict, anchors and verification, when a judge ran

    def view(self) -> dict[str, Any]:
        return {"label": self.label, "disposition": self.disposition, "reason": self.reason, "provenance": self.provenance,
                **({"masked_tokens": self.masked} if self.masked else {}),
                **({"judge": {k: self.judge[k] for k in ("label", "verified_anchors", "rule_label") if k in self.judge}} if self.judge else {})}

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, **self.view()}


class KnowledgeGate:
    """Applicability decisions and sanitized views for one step; ``decisions`` is the record of everything shown."""

    MODES = ("full", "format_only")

    NAMES = ("paths", "pages", "screens")

    def __init__(self, inspector: Any, context: EpisodeContext, action: NormalizedAction | dict[str, Any] | None,
                 mode: str = "full", judge_llm: Any = None, format_documented: bool = False, names: str = "paths"):
        if mode not in self.MODES:
            raise ValueError(f"knowledge gate mode must be one of {self.MODES}")
        if names not in self.NAMES:
            raise ValueError(f"knowledge gate names must be one of {self.NAMES}")
        # v5.3.2 (web): with names="pages", an evidence item whose page before the action is this episode's current
        # page, or whose page after a navigation is the current navigation's destination, is supporting — the sites
        # are one shared instance, so the same page is the same object whatever task visited it. "paths" is v5.3.1.
        self.names = names
        # Option B (2026-09-22, MCP): when the official input documents this action's observation format with an
        # example, package items that would only show a format (format_only, uncertain) are withheld — the official
        # example is authoritative for format; supporting (same-task) items are still shown.
        self.format_documented = format_documented
        self.inspector = inspector
        self.context = context
        # "full" is v5.2. "format_only" is the pre-analysis ablation: evidence labelled supporting or uncertain is
        # withheld (disposition rejected) so the agent receives only sanitized format-level knowledge; notes unchanged.
        self.mode = mode
        # v5.3: an optional structured-LLM judge of applicability (role knowledge_applicability). It proposes a label
        # and anchors per candidate; the gate verifies anchors against the episode before honouring "supporting".
        self.judge_llm = judge_llm
        self.action = action
        self.action_tokens = specific_tokens(action_text(action)) if action is not None else set()
        arguments = (action.arguments if isinstance(action, NormalizedAction) else (action or {}).get("arguments") or {}) if action is not None else {}
        action_type = action.type if isinstance(action, NormalizedAction) else str((action or {}).get("type") or "")
        target = arguments.get("url") if isinstance(arguments, dict) and action_type == "browser_navigate" else None
        self.target_page_raw = target if isinstance(target, str) else None
        self.target_page = normalize_page(self.target_page_raw)
        self.decisions: list[GateDecision] = []
        self._by_id: dict[str, GateDecision] = {}
        self._verdicts: dict[str, dict[str, Any]] = {}
        self.judge_calls = 0
        self.judge_errors: list[str] = []

    # ─── Episode inventories (cached on the inspector; raw package files untouched) ───────────────────────────

    def _inventories(self) -> dict[str, set[str]]:
        cached = getattr(self.inspector, "_gate_inventories", None)
        if cached is not None:
            return cached
        inventories: dict[str, set[str]] = {}
        index = self.inspector._evidence_index()
        for record in index["by_id"].values():
            names = inventories.setdefault(record.episode_id, set())
            names.update(specific_tokens(action_text(record.action)))
            names.update(_fact_paths(record.preconditions))
            names.update(_fact_paths(record.observation_facts))
            for line in record.observation_text.splitlines():
                match = LONG_LISTING.match(line.rstrip())
                if match:
                    names.update(specific_tokens(match.group(4)) or {_strip(match.group(4).split(" -> ")[0])})
        self.inspector._gate_inventories = inventories
        return inventories

    def _episode_task(self, episode_id: str) -> str:
        cached = getattr(self.inspector, "_gate_episode_tasks", None)
        if cached is None:
            cached = {}
            try:
                from trace2env.storage import load_models  # local import: storage has no gate dependency
                from trace2env.models import Episode
                for episode in load_models(self.inspector.package.root / "evidence" / "episodes.jsonl", Episode):
                    original = str((episode.metadata or {}).get("episode_id") or "")
                    parts = original.split(":")
                    if parts[0] == "awb" and len(parts) >= 3:  # awb-export episodes: awb:<split>:<trajectory id>
                        cached[episode.id] = ":".join(parts[2:])
                    else:
                        cached[episode.id] = parts[1] if len(parts) >= 2 else (original or episode.id)
            except Exception:  # noqa: BLE001 - provenance is informational; a missing file must not break a step
                cached = {}
            self.inspector._gate_episode_tasks = cached
        return cached.get(episode_id, episode_id)

    # ─── Evidence and demonstrations ────────────────────────────────────────────────────────────────────────────

    def _name_tasks(self) -> dict[str, set[str]]:
        """For every name in the package's inventories, the distinct tasks whose episodes own it (cached)."""
        cached = getattr(self.inspector, "_gate_name_tasks", None)
        if cached is not None:
            return cached
        cached = {}
        for episode_id, names in self._inventories().items():
            task = self._episode_task(episode_id)
            for name in names:
                cached.setdefault(name, set()).add(task)
        self.inspector._gate_name_tasks = cached
        return cached

    def specific_in_package(self, name: str) -> bool:
        """v5.3.1: a name is a relevance signal only when it is rare in the package. A name that several tasks'
        episodes own (`default`, `solve.py`, `application/json`) identifies none of them."""
        return len(self._name_tasks().get(name, ())) <= 1

    @staticmethod
    def distinct_signals(names: set[str]) -> list[str]:
        """v5.3.1: one signal per path. A shared path and its own components (`etc/ssh/sshd_config`, `sshd_config`)
        are one piece of evidence, not two."""
        kept: list[str] = []
        for name in sorted(names, key=lambda n: (-len(n), n)):
            if not any(name in k.split("/") or k.endswith("/" + name) for k in kept):
                kept.append(name)
        return sorted(kept)

    def _page_before(self, record: LocalTransitionEvidence) -> str | None:
        """The page the evidence episode showed before this action: a ``surface.page.url`` precondition, else the page
        after the preceding transition of the same episode."""
        for fact in record.preconditions:
            subject = str(getattr(fact, "subject", ""))
            value = getattr(fact, "value", None)
            if subject == "surface.page.url" and isinstance(value, str):
                return normalize_page(value)
            if subject == "surface.page" and isinstance(value, dict) and isinstance(value.get("url"), str):
                return normalize_page(value["url"])
        index = self.inspector._evidence_index()
        position = index["position"].get(record.id)
        if position:
            episode_id, number = position
            sequence = index["sequences"].get(episode_id, [])
            if number >= 2:
                previous = index["by_id"].get(sequence[number - 2])
                if previous is not None:
                    return self._page_after(previous)
        return None

    @staticmethod
    def _page_after(record: LocalTransitionEvidence) -> str | None:
        found = PAGE_URL.findall(record.observation_text or "")
        return normalize_page(found[-1]) if found else None

    def _screen_before(self, record: LocalTransitionEvidence) -> tuple[str, Any] | None:
        """The screen the evidence episode showed before this action: the preceding transition's observation."""
        index = self.inspector._evidence_index()
        position = index["position"].get(record.id)
        if position:
            episode_id, number = position
            sequence = index["sequences"].get(episode_id, [])
            if number >= 2:
                previous = index["by_id"].get(sequence[number - 2])
                if previous is not None:
                    return screen_signature(previous.observation_text)
        return None

    def _screen_decision(self, record: LocalTransitionEvidence) -> str | None:
        """v5.3.3: a reason when the evidence is about this episode's current screen (same app, same resource ids or the
        same DroidBot state id), or the same `open_app` target."""
        if self.names != "screens":
            return None
        current = self.context.current_screen
        if current is not None and same_screen(self._screen_before(record), current):
            if current[0] == "state":
                return f"same screen (state id '{current[1]}' as this episode's current state)"
            shared = sorted(current[1] & (self._screen_before(record) or (None, frozenset()))[1])
            anchor = next((i for i in shared if i.split(":id/")[0] not in SYSTEM_PACKAGES), shared[0] if shared else current[0])
            return f"same screen (app {current[0]}, {len(shared)} shared resource ids such as '{anchor}')"
        mine_type = str((self.action.type if isinstance(self.action, NormalizedAction) else (self.action or {}).get("type")) or "")
        if mine_type.endswith("open_app") and record.action.type == mine_type:  # open_app / phone.open_app
            mine = (self.action.arguments if isinstance(self.action, NormalizedAction) else (self.action or {}).get("arguments") or {})
            theirs = record.action.arguments or {}
            name = mine.get("app_name") or mine.get("app") or mine.get("text")
            if name and str(name).strip().lower() == str(theirs.get("app_name") or theirs.get("app") or theirs.get("text") or "").strip().lower():
                return f"same app opened ('{name}')"
        return None

    def _page_decision(self, record: LocalTransitionEvidence) -> tuple[str, str] | None:
        """v5.3.2: (kind, raw url) when the evidence is about this episode's current page or the navigation target."""
        if self.names != "pages":
            return None
        if self.target_page and self._page_after(record) == self.target_page:
            return "destination", self.target_page_raw or self.target_page
        if self.context.current_page and self._page_before(record) == self.context.current_page:
            return "page", self.context.current_page_raw or self.context.current_page
        return None

    def _rule_decision(self, record: LocalTransitionEvidence) -> tuple[str, str, str, dict[str, Any], bool]:
        """The deterministic label: (label, disposition, reason, provenance, cwd_conflict) from name overlap alone."""
        screen = self._screen_decision(record)
        if screen is not None:
            provenance = {"episode": record.episode_id, "task": self._episode_task(record.episode_id), "container": "the same app"}
            return "supporting", "shown", screen, provenance, False
        page = self._page_decision(record)
        if page is not None:
            provenance = {"episode": record.episode_id, "task": self._episode_task(record.episode_id), "container": "the shared site instance"}
            reason = (f"same destination '{page[1]}' as the current navigation" if page[0] == "destination"
                      else f"same page '{page[1]}' as this episode's current page")
            return "supporting", "shown", reason, provenance, False
        evidence_tokens = specific_tokens(action_text(record.action))
        shared_operands = self.distinct_signals({n for n in self.action_tokens & evidence_tokens if self.specific_in_package(n)})
        inventory = self._inventories().get(record.episode_id, set())
        # Names this episode's *history* established that the evidence episode also owns; the current action's operands
        # are not in the history context, so a name counts here only when the episode showed it independently.
        shared_files = self.distinct_signals({n for n in inventory & self.context.shown_names if self.specific_in_package(n)})
        cwd_pre = next((str(f.value) for f in record.preconditions if str(f.subject) == "session.cwd" and isinstance(f.value, str)), None)
        cwd_conflict = bool(cwd_pre and self.context.cwd and cwd_pre.rstrip("/") != self.context.cwd.rstrip("/"))
        score = len(shared_operands) + min(len(shared_files), 3)
        provenance = {"episode": record.episode_id, "task": self._episode_task(record.episode_id), "container": "another run"}
        if score >= 2 and (shared_operands or len(shared_files) >= 2) and not cwd_conflict:
            label, disposition = "supporting", "shown"
            reason = f"shares specific operands {shared_operands[:4]} with the current action and names {shared_files[:4]} with this episode's history"
        elif score >= 1:
            label, disposition = "uncertain", "sanitized"
            reason = f"one weak relevance signal (operands {shared_operands[:3]}, files {shared_files[:3]})" + (
                f"; its cwd {cwd_pre} differs from this episode's {self.context.cwd}" if cwd_conflict else "")
        else:
            label, disposition = "format_only", "sanitized"
            reason = "no specific operand or file shared with this episode: another container; shape only"
        return label, disposition, reason, provenance, cwd_conflict

    def decide_evidence(self, record: LocalTransitionEvidence, *, kind: str = "evidence") -> GateDecision:
        key = f"{kind}:{record.id}"
        if key in self._by_id:
            return self._by_id[key]
        label, disposition, reason, provenance, cwd_conflict = self._rule_decision(record)
        judge: dict[str, Any] | None = None
        if self.judge_llm is not None:
            self.prejudge([record])
            verdict = self._verdicts.get(record.id)
            if verdict is not None:
                judge = dict(verdict)
                rule_label = label
                if verdict["label"] == "supporting" and verdict["verified_anchors"] and not cwd_conflict:
                    label, disposition = "supporting", "shown"
                    reason = f"judge: same task family, anchors {verdict['verified_anchors'][:4]} verified in this episode; {verdict['reason'][:160]}"
                elif verdict["label"] == "supporting":
                    # No verified anchor: an upgrade is not honoured; a rule-supported item the judge agrees with stays
                    # supporting (verification guards upgrades, it does not overrule an agreement).
                    reason = (f"judge agrees (supporting) though its anchors {verdict['anchors'][:4]} are not names; kept the rule's {label}"
                              if rule_label == "supporting" else
                              f"judge proposed supporting but no anchor was verified ({verdict['anchors'][:4]}); kept {label}")
                elif verdict["label"] == "contradicted":
                    label, disposition = "contradicted", "sanitized"
                    reason = f"judge: contradicted by this episode; {verdict['reason'][:160]}"
                elif rule_label == "supporting":  # judge downgrades a rule-supported item: honoured
                    label, disposition = "format_only", "sanitized"
                    reason = f"judge: format only ({verdict['reason'][:160]}); rule said supporting"
                judge["rule_label"] = rule_label
        if self.mode == "format_only" and label in ("supporting", "uncertain"):
            disposition = "rejected"
            reason += " (withheld: format_only ablation shows no task-specific or uncertain evidence)"
        if self.format_documented and label in ("format_only", "uncertain") and disposition != "rejected":
            disposition = "rejected"
            reason += " (withheld: the official input documents this action's observation format)"
        decision = GateDecision(id=key, kind=kind, label=label, disposition=disposition, reason=reason, provenance=provenance)
        if judge is not None:
            decision.judge = judge
        self.decisions.append(decision)
        self._by_id[key] = decision
        return decision

    # ─── v5.3 judge: propose (LLM), then verify (deterministic) ─────────────────────────────────────────────────

    def prejudge(self, records: list[LocalTransitionEvidence]) -> None:
        """Ask the judge about candidates not judged yet, in one batched structured call; store verdicts with their
        anchors verified against this episode's transcript and current action. Failures fall back to the rules."""
        if self.judge_llm is None:
            return
        pending = [r for r in records if r.id not in self._verdicts]
        if not pending:
            return
        from trace2env.models import ApplicabilityJudgement  # local import: models has no gate dependency
        from trace2env.prompts import KNOWLEDGE_APPLICABILITY
        candidates = []
        for record in pending:
            inventory = sorted(self._inventories().get(record.episode_id, set()))[:20]
            rule_label, _, rule_reason, _, _ = self._rule_decision(record)
            candidates.append({
                "id": record.id, "task": self._episode_task(record.episode_id), "command_lines": action_text(record.action).split("\n")[1:6],
                "observation_head": record.observation_text[:700], "episode_names": inventory,
                "rule_label": rule_label, "rule_reason": rule_reason,
            })
        payload = {
            "episode": {"cwd": self.context.cwd, "shown_names": sorted(self.context.shown_names)[:60],
                        "transcript_tail": self.context.transcript[-2000:]},
            "current_action": action_text(self.action).split("\n")[1:8] if self.action is not None else [],
            "candidates": candidates,
        }
        try:
            self.judge_calls += 1
            result = self.judge_llm.complete(system=KNOWLEDGE_APPLICABILITY, user=json.dumps(payload, ensure_ascii=False, default=str),
                                             response_model=ApplicabilityJudgement, role="knowledge_applicability")
        except Exception as exc:  # noqa: BLE001 - the deterministic label is the fallback; the failure is recorded
            self.judge_errors.append(f"{type(exc).__name__}: {exc}"[:200])
            for record in pending:
                self._verdicts[record.id] = None  # type: ignore[assignment]
            return
        by_id = {item.id.split(":", 1)[-1]: item for item in result.items}
        haystack = self.context.transcript + "\n" + (action_text(self.action) if self.action is not None else "")
        for record in pending:
            item = by_id.get(record.id)
            if item is None:
                self._verdicts[record.id] = None  # type: ignore[assignment]
                continue
            anchors = [a.strip() for a in item.anchors if isinstance(a, str) and a.strip()]
            verified = [a for a in anchors if self._verified_anchor(a, haystack)]
            self._verdicts[record.id] = {"label": item.label, "anchors": anchors[:8], "verified_anchors": verified[:8], "reason": item.reason}

    def _verified_anchor(self, anchor: str, haystack: str) -> bool:
        """An anchor the judge named counts only if it occurs verbatim in this episode's transcript or current action
        and is itself a specific name: a file, path or directory operand that is neither generic nor shared by several
        package tasks (v5.3.1: `/app/result.txt` or `default` never verify, whatever the judge says)."""
        if len(anchor) < 3 or anchor not in haystack:
            return False
        stripped = anchor.strip().rstrip("/")
        if stripped.startswith("./"):
            stripped = stripped[2:]
        candidate = stripped.lstrip("/")
        tokens = specific_tokens(anchor)
        if not tokens or (candidate not in tokens and stripped not in tokens):
            return False  # not a name by itself (a phrase, a generic path, a code identifier)
        return self.specific_in_package(candidate) and self.specific_in_package(stripped)

    def sanitize(self, text: str, record: LocalTransitionEvidence | None, decision: GateDecision) -> str:
        """The model-facing copy of a foreign observation: hosts, unknown names, sizes, dates, versions, ids masked."""
        if decision.disposition != "sanitized" or not text:
            return text
        shown = self.context.shown_names
        count = 0
        foreign = set()
        if record is not None:
            foreign = {t for t in self._inventories().get(record.episode_id, set()) if t not in shown and t.lower() not in GENERIC_NAMES}

        def host(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return f"{match.group(1)}@<host>"

        text = HOST_PROMPT.sub(host, text)
        for pattern, placeholder in ((UUID, "<id>"), (HEX_ID, "<hash>"), (IPV4, "<ip>"), (DATE, "<date>"), (VERSION, "<version>")):
            text, n = pattern.subn(placeholder, text)
            count += n

        def listing(match: re.Match[str]) -> str:
            nonlocal count
            name = match.group(4)
            base = _strip(name.split(" -> ")[0])
            if base and base not in shown and base.lower() not in GENERIC_NAMES:
                count += 2
                return f"{match.group(1)}<n>{match.group(3)}<name>"
            count += 1
            return f"{match.group(1)}<n>{match.group(3)}{name}"

        text = "\n".join(LONG_LISTING.sub(listing, line) if LONG_LISTING.match(line.rstrip()) else line for line in text.split("\n"))

        def token(match: re.Match[str]) -> str:
            nonlocal count
            raw = match.group(0)
            stripped = _strip(raw).rstrip("/")
            if not stripped or stripped in shown or stripped.lower() in GENERIC_NAMES:
                return raw
            specific = specific_tokens(raw)
            if stripped in foreign or (specific and not (specific & shown)):
                count += 1
                if "/" in stripped:
                    return "<path>"
                ext = re.search(r"\.[A-Za-z0-9]{1,6}$", stripped)
                return f"<file{ext.group(0)}>" if ext else "<name>"
            return raw

        text = TOKEN.sub(token, text)
        text, n = LARGE_NUMBER.subn("<n>", text)
        count += n
        text, n = TOTAL_LINE.subn("total <n>", text)
        count += n
        decision.masked += count
        return text

    def sanitize_value(self, value: Any, record: LocalTransitionEvidence | None, decision: GateDecision) -> Any:
        if isinstance(value, str):
            return self.sanitize(value, record, decision)
        if isinstance(value, list):
            return [self.sanitize_value(v, record, decision) for v in value]
        if isinstance(value, dict):
            return {k: self.sanitize_value(v, record, decision) for k, v in value.items()}
        return value

    # ─── Notes ──────────────────────────────────────────────────────────────────────────────────────────────────

    def decide_note(self, note: EnvironmentNote | dict[str, Any]) -> GateDecision:
        data = note if isinstance(note, dict) else note.model_dump(mode="json")
        key = f"note:{data.get('id')}"
        if key in self._by_id:
            return self._by_id[key]
        statement = str(data.get("statement") or "")
        lowered = statement.lower()
        label, reason = "untested", "no transcript check applies to this note"
        ctx = self.context
        if "continuation prompt" in lowered or "continuation-prompt" in lowered:
            literal = re.search(r"`(>\s?)`", statement)
            claimed = None if literal is None else ("space" if literal.group(1).endswith(" ") else "bare")
            if claimed and ctx.continuation_style and claimed != ctx.continuation_style:
                label, reason = "contradicted", f"the note claims a {claimed!r} blank continuation line; this episode shows {ctx.continuation_style!r}"
            elif claimed and ctx.continuation_style:
                label, reason = "consistent", "the continuation prompt matches this episode's transcript"
        elif "command not found" in lowered:
            if ctx.has_command_not_found is True:
                label, reason = "consistent", "this episode shows the same command-not-found shape"
            elif ctx.has_command_not_found is False:
                label, reason = "contradicted", "this episode shows a different command-not-found shape"
        elif "`ls" in statement or "ls -la" in lowered or "long metadata row" in lowered:
            if ctx.has_long_listing:
                label, reason = "consistent", "this episode already shows a long listing with a total line"
        disposition = "rejected" if label == "contradicted" else "shown"
        decision = GateDecision(id=key, kind="note", label=label, disposition=disposition, reason=reason,
                                provenance={"status": data.get("status"), "confidence": data.get("confidence")})
        self.decisions.append(decision)
        self._by_id[key] = decision
        return decision

    # ─── Summary ────────────────────────────────────────────────────────────────────────────────────────────────

    def supporting_ids(self) -> list[str]:
        return [d.id for d in self.decisions if d.label == "supporting" and d.disposition == "shown"]

    def summary(self, *, brief_items: int | None = None) -> dict[str, Any]:
        labels: dict[str, int] = {}
        dispositions: dict[str, int] = {}
        for d in self.decisions:
            labels[d.label] = labels.get(d.label, 0) + 1
            dispositions[d.disposition] = dispositions.get(d.disposition, 0) + 1
        judged = [d for d in self.decisions if d.judge]
        return {
            "mode": self.mode, "names": self.names, "items": len(self.decisions), "labels": labels, "dispositions": dispositions,
            "page_matches": sum(1 for d in self.decisions if d.reason.startswith(("same page ", "same destination "))),
            "screen_matches": sum(1 for d in self.decisions if d.reason.startswith(("same screen", "same app opened"))),
            "judge": {"enabled": self.judge_llm is not None, "calls": self.judge_calls, "judged_items": len(judged),
                      "upgraded": sum(1 for d in judged if d.label == "supporting" and d.judge.get("rule_label") != "supporting"),
                      "downgraded": sum(1 for d in judged if d.judge.get("rule_label") == "supporting" and d.label != "supporting"),
                      "unverified_upgrades": sum(1 for d in judged if d.judge.get("label") == "supporting" and not d.judge.get("verified_anchors")),
                      "errors": self.judge_errors[:3]} if self.judge_llm is not None else None,
            "masked_tokens": sum(d.masked for d in self.decisions), "supporting": self.supporting_ids(),
            "abstained": not self.supporting_ids(),
            "format_documented": self.format_documented,
            "withheld_documented_format": sum(1 for d in self.decisions if "documents this action's observation format" in d.reason),
            "context": {"shown_names": len(self.context.shown_names), "cwd": self.context.cwd,
                        "continuation_style": self.context.continuation_style},
            **({"brief_items": brief_items} if brief_items is not None else {}),
            "decisions": [d.record() for d in self.decisions[:120]],
        }

    def brief_notice(self) -> dict[str, Any]:
        supporting = self.supporting_ids()
        if supporting and self.names == "screens":
            text = ("Package items labelled 'supporting' are about this episode's current screen (the same app and the same "
                    "on-screen elements, or the same screen state id) or the app the current action opens, recorded in other "
                    "tasks on the same apps, and are shown intact; this episode's transcript and state remain authoritative "
                    "wherever they differ (the app's data, typed text, task-specific items). Items labelled 'uncertain' or "
                    "'format_only' are sanitized copies: use their shape only.")
        elif supporting and self.names == "pages":
            text = ("Package items labelled 'supporting' are about this episode's current page or the page the current "
                    "navigation leads to (the sites are one shared instance, so the same page is the same object in every "
                    "episode) and are shown intact; this episode's transcript and state remain authoritative wherever they "
                    "differ (login state, form values, objects a task created). Items labelled 'uncertain' or 'format_only' "
                    "are sanitized copies: use their shape only.")
        elif supporting:
            text = ("Package items labelled 'supporting' share specific files or operands with this episode and are shown "
                    "intact; they come from another run of a similar task, so this episode's transcript and state remain "
                    "authoritative wherever they differ. Items labelled 'uncertain' or 'format_only' are sanitized copies "
                    "(hosts, unknown names, sizes, dates, versions masked): use their shape only.")
        else:
            text = ("Abstention: no package item supports this transition in this environment (nothing shares this "
                    "episode's specific files or operands). Package items are sanitized copies from other containers: "
                    "use their output shape only; take every concrete name, size, date, version, and content from this "
                    "episode's transcript and state, or predict it from general knowledge.")
        if self.format_documented:
            text += (" The official input documents this action's observation format with an example; package items that "
                     "would only show a format were withheld. Follow the official example's format exactly.")
        return {"supporting_items": supporting, "abstained": not supporting, "format_documented": self.format_documented, "guidance": text}
