# AgentWorldBench swe, Claude Code sub-source

A developer's own machine (macOS, Linux or Windows paths such as `/Users/<name>/...`, `/home/...`, `/app`, `D:\...`)
with a working copy of the developer's project — a different private repository in every session (Java/Spring
services, Vue and React front ends, Python research code, Node tools, documentation sites, notebooks) — driven by the
Claude Code coding agent. The world model plays the tools' execution results; the repository's files, git state and
installed toolchain are the state, and the session's earlier observations are the only source of their contents.

Actions are tool calls `{"name": <tool>, "arguments": <JSON string>}` with the Claude Code tool set: `Read`
(`file_path`, optional `offset`/`limit`), `Edit` (`file_path`, `old_string`, `new_string`, `replace_all`), `Write`
(`file_path`, `content`), `Bash` (`command`, optional `description`, `timeout`, `run_in_background`), `Glob`
(`pattern`, optional `path`), `Grep` (`pattern`, optional `path`, `glob`, `output_mode`, `-n`, `-i`, ...), `TodoWrite`
(`todos` list), `Task` (`subagent_type`, `prompt`, `description`), `ExitPlanMode`, `AskUserQuestion`, `WebSearch`,
`WebFetch`, `BashOutput` (`bash_id`), `KillShell`, `NotebookEdit`, `Skill`, `SlashCommand`.

Observation formats (the text after `**Environment Observation:**`): `Read` prints the file with right-aligned line
numbers and an arrow, `     1→import ...` (an offset starts at that line; binary/absent files yield an error line;
very long files are truncated with a note); `Edit` answers `The file <path> has been updated. Here's the result of
running `cat -n` on a snippet of the edited file:` followed by the numbered snippet, or an error when `old_string` is
not found or not unique; `Write` answers `File created successfully at: <path>` or `The file <path> has been updated
successfully.`; `Bash` returns the command's stdout/stderr verbatim (empty for silent commands; `Exit code N` and the
error text on failure; a shell id line for background commands, whose output later comes from `BashOutput`); `Glob`
lists matching absolute paths one per line (`No files found` otherwise); `Grep` prints `path:line:content` lines (or
file names / counts by `output_mode`); `TodoWrite` answers `Todos have been modified successfully. Ensure that you
continue to use the todo list to track your progress. Please proceed with the current tasks if applicable`; `Task`
returns the sub-agent's final report; `AskUserQuestion` returns the user's answers; `WebSearch`/`WebFetch` return
search results or page text. Some observations carry a trailing `<system-reminder>` block.

Behaviour: reads reflect the exact file contents established by earlier reads, writes and edits in the session; an edit
changes only the matched span and renumbers the snippet; shell commands behave like a real shell in the project
directory (tests, builds, git, package managers) with output that depends on the project's real files; errors are
realistic (missing files, failing tests, non-unique edit anchors). Each transition is teacher-forced: the next
observation is the real tool result recorded in the session.
