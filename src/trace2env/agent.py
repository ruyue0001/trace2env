"""The world-model agent loop: tool use over the workspace until a typed transition is submitted.

The loop is a plain function-calling agent: the model receives the tool catalog, calls tools,
sees their results as tool messages, and finishes by calling ``submit_transition``. The harness
bounds the number of tool calls, forces the submission when the budget is spent, and keeps a
trace of every call for the audit record.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from trace2env.llm import AgentLLM
from trace2env.models import SUBMIT_TOOL_NAME, TransitionSubmission
from trace2env.workspace import WorkspaceTools, clip_value


# Names a model may use for the final submission; the contract's tool is SUBMIT_TOOL_NAME, the rest are tolerated.
SUBMIT_ALIASES = {SUBMIT_TOOL_NAME, "final", "submit", "submit_final", "final_submission", "transition_submission"}


class AgentProtocolError(RuntimeError):
    pass


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class AgentLoop:
    def __init__(self, llm: AgentLLM, tools: WorkspaceTools, *, system: str, max_tool_calls: int = 8,
                 role: str = "runtime_agent_turn", max_invalid_submissions: int = 2, max_rounds: int | None = None):
        self.llm = llm
        self.tools = tools
        self.system = system
        self.max_tool_calls = max(0, max_tool_calls)
        self.role = role
        # Invalid submissions (a submit call whose arguments do not validate) are answered with the validation error
        # and do not spend the tool budget, so they need their own bound: more than this many is a protocol error.
        # A cap on model calls per step bounds every path (a model answered 171 rounds of empty submissions before).
        self.max_invalid_submissions = max(0, max_invalid_submissions)
        self.max_rounds = max_rounds if max_rounds is not None else 2 * self.max_tool_calls + 4

    def run(self, user: str) -> tuple[TransitionSubmission, list[dict[str, Any]], dict[str, int]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        specs = self.tools.specs()
        trace: list[dict[str, Any]] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0}
        used = 0
        nudges = 0
        invalid = 0
        force_nudged = False
        while True:
            if usage["model_calls"] >= self.max_rounds:
                raise AgentProtocolError(f"The world-model agent made {usage['model_calls']} calls without a valid submission")
            force = SUBMIT_TOOL_NAME if used >= self.max_tool_calls else None
            reply = self.llm.respond(messages=messages, tools=specs, role=self.role, force_tool=force)
            usage["model_calls"] += 1
            for key in ("prompt_tokens", "completion_tokens"):
                usage[key] += int(reply.usage.get(key, 0) or 0)
            if not reply.tool_calls:
                # A bare answer is not a transition. Ask once for the typed submission, then give up.
                nudges += 1
                if nudges > 1:
                    raise AgentProtocolError("The world-model agent replied without submitting a transition")
                messages.append({"role": "assistant", "content": reply.content or ""})
                messages.append({"role": "user", "content": f"Call {SUBMIT_TOOL_NAME} to finish; a plain reply is not a transition."})
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": _dumps(call.arguments)}}
                        for call in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                if (call.name or "").strip().lower() in SUBMIT_ALIASES:
                    try:
                        submission = TransitionSubmission.model_validate(call.arguments)
                    except ValidationError as exc:
                        invalid += 1
                        trace.append({"tool": SUBMIT_TOOL_NAME, "arguments": clip_value(call.arguments, 200), "error": "invalid submission"})
                        if invalid > self.max_invalid_submissions:
                            raise AgentProtocolError(f"The world-model agent submitted an invalid transition {invalid} times") from exc
                        messages.append({"role": "tool", "tool_call_id": call.id,
                                         "content": _dumps({"error": "Invalid submission: "
                                                            f"{exc.errors(include_input=False, include_url=False)[:3]}"})})
                        continue
                    trace.append({"tool": SUBMIT_TOOL_NAME, "arguments": {"rule_ids": submission.rule_ids,
                                  "citations": submission.citations, "effects": len(submission.effects)}})
                    return submission, trace, usage
                if force is not None:
                    if not force_nudged:
                        # One reminder before failing: models that ignore a forced tool choice usually comply when told.
                        force_nudged = True
                        trace.append({"tool": call.name, "error": "refused: tool budget exhausted"})
                        messages.append({"role": "tool", "tool_call_id": call.id,
                                         "content": _dumps({"error": f"The tool budget is exhausted; call {SUBMIT_TOOL_NAME} now "
                                                                     "with the complete transition submission."})})
                        continue
                    raise AgentProtocolError(f"Tool budget of {self.max_tool_calls} calls is exhausted; {call.name!r} was refused")
                used += 1
                result = self.tools.call(call.name, call.arguments)
                trace.append(self.tools.calls[-1])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _dumps(result)})
            if used > self.max_tool_calls + 1:
                raise AgentProtocolError("The world-model agent exceeded its tool budget")
