Environment: a Linux shell inside a Docker container for one Terminal-Bench 2.0 task, driven through a
tmux session by the Terminus-2 agent harness. The task agent never sees the container directly; it only
types keystrokes and reads the terminal screen.

Actions. Each action is a batch of keystrokes typed into the terminal, normalized as follows:
- a single simple command line ending in a newline -> action type = the command name (ls, cat, cd,
  python3, pip, git, ...), with arguments `command` (the full line), `argv` (the remaining words) and
  `keystrokes` (the raw batch entries with per-entry wait `duration` in seconds);
- several lines, pipelines, heredocs, or shell control operators -> action type `shell` with `commands`
  (the non-empty lines) and `keystrokes`;
- input without a trailing newline, or control sequences such as C-c, Enter, arrow keys -> `keys` with
  `keys` and `keystrokes`;
- no keystrokes at all -> `wait` with `duration`.

Observations. After typing, the harness waits the requested duration and captures the terminal. The
observation is the terminal text produced since the previous capture (or the current screen after a
wait): the echoed command line(s), program output, and the shell prompt `root@<container-id>:<cwd>#`
when the shell is idle again. Long-running commands may still be running when the screen is captured;
their remaining output then appears in later observations, and a foreground program consumes later
keystrokes as its own input. Output of interactive programs (editors, pagers, REPLs, `ssh`, `python`
prompts) replaces the shell prompt until they exit.

State that matters. The current working directory (shown in the prompt), files and directories with
their contents and permissions, environment variables and shell variables, installed programs and
Python packages, running foreground/background processes and started network services, the exit
status of the last command, the shell's mode (idle prompt, foreground program, heredoc continuation
`>`), and what is on screen.

Suggested state paths (use these names where they fit; add others only when the evidence needs them):
- session.cwd (string): the working directory shown in the prompt;
- world.files (object): absolute path -> {"type": "file"|"dir", "content": text when observed, "size",
  "mode", ...}; edit entries with op merge / op remove, never replace the whole map;
- world.env (object): environment and shell variables that were set or observed;
- world.processes (object): running foreground/background programs and services, keyed by name or pid;
- world.repositories (object): git repository path -> {"branch", "head", "status", "log", ...} as observed;
  other stateful subsystems (a database, a running service, a container) likewise get one world.<name>
  map keyed by path or name;
- world.installed (object): program or package name -> true when it ran, false when the shell reported
  "command not found" or an import/module error showed it missing;
- world.last_exit_status (integer) when it is observed or shown;
- surface.mode (string): "prompt" (idle shell), "program:<name>" (a foreground program owns the
  terminal), or "continuation" (the shell is waiting for more input, e.g. a heredoc `>` prompt).
Content that a command reveals (a file listing, a file's text, a git log) is recorded under world.files
or the matching world.* map exactly as displayed; values a program computes from it are output, not
state. Do not keep the screen text in state (there is no surface.screen field). epistemic.* is rare:
only for bookkeeping of hidden values that remain unknown after this step.

Scope. Every episode starts from a fresh container built for its task (task files typically live under
/app; the container hostname in the prompt is unique to the episode). File contents, task data, and
pre-installed tools are facts about one task's container, not universal environment behavior; the
behavior of the shell and of standard Unix tools (error messages, prompt format, command-not-found
text, exit conventions) is shared by every episode. Commands such as ls or cat reveal pre-existing
content that only the container knows; that content is observed, not derivable.
