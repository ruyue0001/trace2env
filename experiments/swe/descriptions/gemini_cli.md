# AgentWorldBench swe, Gemini/Qwen CLI sub-source

A developer's own machine (macOS, Linux or Windows paths, sandboxed `/app` workspaces in some sessions) with a working
copy of the developer's project — a different private repository in every session (web apps, Python and Node tools,
data scripts, documentation) — driven by a Gemini-CLI-style coding agent (Gemini CLI / Qwen Code). The world model
plays the tools' execution results; the repository's files, processes and installed toolchain are the state, and the
session's earlier observations are the only source of their contents.

Actions are tool calls `{"name": <tool>, "arguments": <JSON string>}` with the CLI tool set: `read_file`
(`absolute_path`, optional `offset`/`limit`), `write_file` (`file_path`, `content`), `edit` (`file_path`,
`old_string`, `new_string`, optional `expected_replacements`), `run_shell_command` (`command`, optional
`description`, `directory`), `list_directory` (`path`, optional `ignore`), `glob` (`pattern`, optional `path`),
`search_file_content` (`pattern`, optional `path`, `include`), `read_many_files` (`paths`), `todo_write` (`todos`),
`save_memory` (`fact`), `exit_plan_mode`, `web_fetch`, `google_web_search`.

Observation formats (the text after `**Environment Observation:**`): `read_file` returns the file content verbatim
(no line numbers; a note when truncated; an error for absent paths); `write_file` answers `Successfully created and
wrote to new file: <path>.` or `Successfully overwrote file: <path>.`; `edit` answers `Successfully modified file:
<path> (N replacements).` or an error when the anchor is not found / ambiguous; `run_shell_command` returns a fixed
envelope `Command: ...`, `Directory: ...`, `Output: ...`, `Result: ...`, `Error: ...`, `Exit Code: N`, `Signal: ...`,
`Background PIDs: ...`, `Process Group PGID: ...` (stdout in `Output`, empty fields as `(none)`); `list_directory`
prints `Directory listing for <path>:` then `[DIR] name` entries first and files after, sorted; `glob` lists matching
absolute paths one per line; `search_file_content` prints matches grouped by file with line numbers; `todo_write`
answers `Todos have been modified successfully. ...` sometimes followed by a `<system-reminder>` block; `save_memory`
confirms the saved fact.

Behaviour: reads reflect the exact file contents established by earlier reads, writes and edits; an edit replaces only
the matched span; shell commands behave like a real shell in the project directory with output depending on the
project's real files; errors are realistic. Each transition is teacher-forced: the next observation is the real tool
result recorded in the session.
