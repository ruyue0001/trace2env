# AgentWorldBench prompts

The files in this directory are copied verbatim from the Qwen-AgentWorld repository
(https://github.com/QwenLM/Qwen-AgentWorld, `prompts/{domain}/`), Apache License 2.0.

- `judge_system_prompt.txt`: the official LLM-judge system prompt used by `trace2env awb-judge`.
  Trace2Env uses these unchanged so that scores are computed the same way as `eval/eval.py judge`
  in that repository.
- `system_prompt.txt`: the reference world-model system prompt template for the domain. During
  evaluation each AgentWorldBench record carries its own `system_str`, which takes precedence;
  the templates are kept for reference and for describing a domain during reconstruction.
