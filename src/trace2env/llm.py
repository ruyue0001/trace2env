"""Provider boundary for structured, free-text, and tool-calling LLM calls, with deterministic disk caches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from trace2env.models import SUBMIT_TOOL_NAME, AgentReply, AgentTurn, ToolCall, TransitionSubmission
from trace2env.storage import read_json, write_json

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: str,
    ) -> T: ...


class ChatLLM(Protocol):
    """Free-text chat completion for protocols that fix their own message layout and output tags."""

    def chat(self, *, messages: list[dict[str, str]], role: str) -> str: ...


class AgentLLM(Protocol):
    """One turn of a function-calling agent: the reply either calls tools or answers in text."""

    def respond(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: str,
        force_tool: str | None = None,
    ) -> AgentReply: ...


# One attempt per request: the SDK's default of two silent retries after its 600 s timeout re-runs
# (and re-bills) long generations; ``call_with_retry`` handles transient failures explicitly instead.
CLIENT_TIMEOUT_S = 1800.0
CLIENT_MAX_RETRIES = 0


def _openai_client(client: Any, base_url: str | None, api_key: str | None) -> Any:
    if client is not None:
        return client
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to use OpenAI: pip install -e .") from exc
    # Local OpenAI-compatible servers (vLLM, SGLang) accept any key; the official eval uses "EMPTY".
    key = api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if base_url else None)
    return OpenAI(base_url=base_url, api_key=key, timeout=CLIENT_TIMEOUT_S, max_retries=CLIENT_MAX_RETRIES)


def extract_json_text(content: str) -> str:
    """Return the JSON object inside a chat answer, tolerating code fences and surrounding prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.S)
    if fenced:
        return fenced.group(1)
    start, end = content.find("{"), content.rfind("}")
    return content[start:end + 1] if 0 <= start < end else content


def format_validation_errors(exc: ValidationError, limit: int = 10) -> str:
    lines = []
    for error in exc.errors()[:limit]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        lines.append(f"- {location}: {error.get('msg')}")
    return "\n".join(lines)


def repair_instruction(errors: str) -> str:
    return (
        "Your previous JSON did not satisfy the schema. Fix exactly these problems and return the complete "
        "corrected JSON object only, keeping everything else unchanged:\n" + errors
    )



def _response_provider(response: Any) -> str | None:
    """OpenRouter's upstream provider name for a completion (a top-level `provider` field the SDK keeps as an extra)."""
    value = getattr(response, "provider", None)
    if not isinstance(value, str):
        extra = getattr(response, "model_extra", None) or {}
        value = extra.get("provider") if isinstance(extra, dict) else None
    return value if isinstance(value, str) and value else None

def usage_dict(usage: Any, provider: Any = None) -> dict[str, float] | None:
    """Token counts (and OpenRouter's ``cost``) from a provider usage object, or None when absent."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = usage
    else:
        data = dict(vars(usage)) if hasattr(usage, "__dict__") else {}
    out: dict[str, float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        if isinstance(data.get(key), (int, float)):
            out[key] = int(data[key])
    details = data.get("prompt_tokens_details") or data.get("input_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), (int, float)):
        out["cached_tokens"] = int(details["cached_tokens"])
    completion_details = data.get("completion_tokens_details") or data.get("output_tokens_details") or {}
    if hasattr(completion_details, "model_dump"):
        completion_details = completion_details.model_dump()
    if isinstance(completion_details, dict) and isinstance(completion_details.get("reasoning_tokens"), (int, float)):
        out["reasoning_tokens"] = int(completion_details["reasoning_tokens"])
    if isinstance(provider, str) and provider:
        out["provider"] = provider  # the upstream provider OpenRouter routed to (its behaviour drifts between runs)
    if isinstance(data.get("cost"), (int, float)):
        out["cost_usd"] = float(data["cost"])
    return out or None


def _add_usage(total: dict[str, float] | None, usage: dict[str, float] | None) -> dict[str, float] | None:
    if not usage:
        return total
    merged = dict(total or {})
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = merged.get(key, 0) + value
        else:
            merged[key] = value  # labels such as the upstream provider: the latest attempt's value
    merged["attempts"] = int(merged.get("attempts", 0)) + 1
    return merged


class TransientProviderError(RuntimeError):
    """A provider answer that carries no usable output (gateway error, empty choices, error finish reason)."""


class ProviderRefusal(RuntimeError):
    """The provider declined to answer this request (content policy); retrying cannot help."""


class StructuredOutputError(RuntimeError):
    """The model's answer never satisfied the response model, even after the bounded repair attempts."""


class OutputTruncated(StructuredOutputError):
    """The answer was cut off at the output limit; re-asking cannot repair it (raise the limit or shrink the input)."""


REFUSAL_MARKERS = ("flagged", "content policy", "content_policy", "usage policies", "cybersecurity", "content_filter",
                   "invalid_prompt", "refused", "safety system")


def _refusal_text(detail: Any) -> str | None:
    if detail is None:
        return None
    message = detail.get("message", detail) if isinstance(detail, dict) else detail
    text = str(message)
    return text if any(marker in text.lower() for marker in REFUSAL_MARKERS) else None


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, TransientProviderError):
        return True
    module = type(exc).__module__ or ""
    name = type(exc).__name__
    if not module.startswith("openai"):
        return False
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    status = getattr(exc, "status_code", None)
    if status == 402 and "in-flight" in str(exc).lower() or status == 402 and "in_flight" in str(exc).lower():
        # OpenRouter reserves credit for concurrent requests; this clears once they settle.
        return True
    return isinstance(status, int) and (status == 429 or status >= 500)


def call_with_retry(request, *, role: str, retries: int = 5, backoff_s: float = 2.0):
    """Run ``request()`` again after transient provider failures with exponential backoff.

    Chat gateways (OpenRouter) surface upstream trouble as HTTP 429/5xx, as responses without
    ``choices``, or as an ``error`` finish reason with empty content; none of those are answers.
    """
    delay = backoff_s
    for attempt in range(retries + 1):
        try:
            response = request()
        except Exception as exc:
            refusal = _refusal_text(str(exc)) if (type(exc).__module__ or "").startswith("openai") else None
            if refusal:
                raise ProviderRefusal(f"Provider refused role {role}: {refusal}") from exc
            if attempt >= retries or not _is_transient(exc):
                raise
            time.sleep(delay)
            delay *= 2
            continue
        choices = getattr(response, "choices", None)
        if not choices:
            detail = getattr(response, "error", None)
            refusal = _refusal_text(detail)
            if refusal:
                raise ProviderRefusal(f"Provider refused role {role}: {refusal}")
            problem = TransientProviderError(f"Provider returned no choices for role {role}: {detail}")
        elif getattr(choices[0], "finish_reason", None) == "error" and not (getattr(choices[0].message, "content", None) or "").strip():
            problem = TransientProviderError(f"Provider reported an error finish for role {role}: {getattr(response, 'error', None)}")
        else:
            return response
        if attempt >= retries:
            raise problem
        time.sleep(delay)
        delay *= 2
    raise TransientProviderError(f"Provider request for role {role} did not succeed")


class OpenAIResponsesLLM:
    """OpenAI Responses API adapter. Import and credential checks are intentionally lazy."""

    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        client: Any = None,
        store: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
        repair_attempts: int = 2,
    ):
        self.client = _openai_client(client, base_url, api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.store = store
        self.repair_attempts = repair_attempts
        self._local = threading.local()  # per-thread bookkeeping: concurrent callers must not see each other's usage

    @property
    def last_usage(self) -> dict[str, float] | None:
        """Token counts of this thread's most recent complete() call."""
        return getattr(self._local, "usage", None)

    @last_usage.setter
    def last_usage(self, value: dict[str, float] | None) -> None:
        self._local.usage = value

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: str,
    ) -> T:
        common = {
            "model": self.model,
            "instructions": system,
            "reasoning": {"effort": self.reasoning_effort},
            "metadata": {"trace2env_role": role},
            "store": self.store,
        }
        schema = response_model.model_json_schema()
        self.last_usage = None
        if _requires_flexible_schema(schema):
            # Trace evidence necessarily contains environment-defined JSON values and argument
            # names. Use server-side JSON Schema in non-strict mode, then enforce the complete
            # Pydantic contract locally, feeding validation errors back for a bounded repair.
            text = {"format": {"type": "json_schema", "name": response_model.__name__.lower(), "schema": schema, "strict": False}}
            prompt = user
            last_error: Exception | None = None
            for _ in range(self.repair_attempts + 1):
                response = self.client.responses.create(**common, input=prompt, text=text)
                self.last_usage = _add_usage(self.last_usage, usage_dict(getattr(response, "usage", None), provider=_response_provider(response)))
                content = response.output_text or ""
                if not content.strip():
                    last_error = RuntimeError(f"OpenAI returned no JSON output for role {role}")
                    continue
                try:
                    return response_model.model_validate_json(content)
                except ValidationError as exc:
                    last_error = exc
                    prompt = f"{user}\n\n# Previous attempt\n{content}\n\n{repair_instruction(format_validation_errors(exc))}"
            raise StructuredOutputError(f"Structured output for role {role} failed validation after repair attempts: {last_error}") from last_error
        response = self.client.responses.parse(**common, input=user, text_format=response_model)
        self.last_usage = usage_dict(getattr(response, "usage", None), provider=_response_provider(response))
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"OpenAI returned no parsed output for role {role}")
        return parsed if isinstance(parsed, response_model) else response_model.model_validate(parsed)


TRUNCATION_CEILING = 65536  # largest cap a truncation retry may grow to
SCHEMA_MODES = {
    "json_schema",  # response_format json_schema (non-strict): the GPT-5.6-Sol and DeepSeek runs' transport
    "json_object",  # response_format json_object with the schema quoted in the system text: providers whose schema-
    #                 constrained decoding rejects Pydantic's schemas or empties free-form maps (Anthropic via OpenRouter)
}
DEGENERATE_COMPRESSION_RATIO = 0.02  # zlib-compressed / raw bytes below this: a repeating fragment, not an answer
DEGENERATE_MIN_BYTES = 4000


def looks_degenerate(text: str) -> bool:
    """True when a long answer is one short fragment repeated (a runaway generation), judged by compressibility.

    The runaway answers seen live (DeepSeek V4-Pro, typed agent turns) opened a JSON string and repeated a few tokens
    until the 65,536-token cap; they compress to under 1 % of their size, where real answers of that length compress to
    10 % or more.
    """
    data = text.encode("utf-8", "replace")
    return len(data) >= DEGENERATE_MIN_BYTES and len(zlib.compress(data)) / len(data) < DEGENERATE_COMPRESSION_RATIO


def _sampling(temperature: float | None, max_tokens: int | None, reasoning_effort: str | None,
              reasoning: dict[str, Any] | None = None, provider_routing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Only send sampling parameters that were set: reasoning models reject fixed temperatures.

    ``reasoning`` is the OpenRouter-style ``reasoning`` object (``{"enabled": false}``, ``{"effort": "low"}``,
    ``{"max_tokens": N}``) sent in the request body for gateways and models that do not honour the OpenAI
    ``reasoning_effort`` parameter; when it is given, ``reasoning_effort`` is not sent.
    """
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens:
        params["max_tokens"] = max_tokens
    if reasoning:
        params["extra_body"] = {"reasoning": dict(reasoning)}
    elif reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    if provider_routing:  # OpenRouter provider routing, e.g. {"order": ["DeepSeek"], "allow_fallbacks": false}
        params.setdefault("extra_body", {})["provider"] = dict(provider_routing)
    return params


class OpenAIChatStructuredLLM:
    """StructuredLLM over Chat Completions, for OpenAI-compatible servers that lack the Responses API.

    The schema is sent in non-strict ``json_schema`` mode and the complete Pydantic contract is
    enforced locally, exactly as the Responses adapter does for free-form evidence values.
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = 32768,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        provider_routing: dict[str, Any] | None = None,
        repair_attempts: int = 2,
        retry_backoff_s: float = 2.0,
        truncation_retries: int = 1,
        loop_retries: int = 1,
        schema_mode: str = "json_schema",
    ):
        if schema_mode not in SCHEMA_MODES:
            raise ValueError(f"schema_mode must be one of {sorted(SCHEMA_MODES)}")
        self.client = _openai_client(client, base_url, api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning = reasoning
        self.provider_routing = provider_routing  # OpenRouter upstream pin; part of the cache identity
        self.schema_mode = schema_mode  # how the response schema reaches the provider (see SCHEMA_MODES)
        self.repair_attempts = repair_attempts
        self.retry_backoff_s = retry_backoff_s  # transient gateway failures are retried with exponential backoff
        self.truncation_retries = truncation_retries  # an answer cut off at max_tokens is retried once with a doubled cap
        self.loop_retries = loop_retries  # a degenerate repetition cut off at max_tokens is resampled once at the same cap
        self._local = threading.local()  # per-thread bookkeeping: concurrent callers must not see each other's usage

    @property
    def last_usage(self) -> dict[str, float] | None:
        """Token counts summed over this thread's most recent call, including repair attempts."""
        return getattr(self._local, "usage", None)

    @last_usage.setter
    def last_usage(self, value: dict[str, float] | None) -> None:
        self._local.usage = value

    @property
    def last_attempts(self) -> list[dict[str, Any]]:
        """Raw content and validation error of each attempt of this thread's most recent call."""
        return getattr(self._local, "attempts", [])

    @last_attempts.setter
    def last_attempts(self, value: list[dict[str, Any]]) -> None:
        self._local.attempts = value

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: str,
    ) -> T:
        schema = response_model.model_json_schema()
        if self.schema_mode == "json_object":
            instruction = "\n\nRespond with one JSON object that matches this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
            response_format: dict[str, Any] = {"type": "json_object"}
        else:
            instruction = "\n\nRespond with one JSON object that matches the provided schema."
            response_format = {"type": "json_schema", "json_schema": {"name": response_model.__name__.lower(), "schema": schema, "strict": False}}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system + instruction},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        self.last_usage = None
        self.last_attempts = []
        max_tokens = self.max_tokens
        grown = 0  # truncation retries used (separate from the repair budget)
        resampled = 0  # degenerate repetitions resampled so far
        repairs = 0  # invalid or empty answers re-asked so far
        while True:
            response = call_with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **_sampling(self.temperature, max_tokens, self.reasoning_effort, self.reasoning, self.provider_routing),
            ), role=role, backoff_s=self.retry_backoff_s)
            self.last_usage = _add_usage(self.last_usage, usage_dict(getattr(response, "usage", None), provider=_response_provider(response)))
            choice = response.choices[0]
            content = choice.message.content or ""
            attempt: dict[str, Any] = {"finish_reason": getattr(choice, "finish_reason", None), "content": content, "error": None}
            self.last_attempts.append(attempt)
            if attempt["finish_reason"] == "length":
                if resampled < self.loop_retries and looks_degenerate(content):
                    # A runaway repetition is a sampling accident, not a long answer: a fresh sample at the same cap
                    # recovers it (every looped input seen live succeeded when re-asked); doubling the cap would not.
                    resampled += 1
                    attempt["error"] = "degenerate repetition at the output limit; resampling once at the same cap"
                    continue
                if grown < self.truncation_retries and max_tokens and max_tokens < TRUNCATION_CEILING:
                    # Reasoning models spend part of max_tokens on reasoning; one retry with a doubled cap recovers most
                    # answers that were cut off (the cap is bounded so a runaway generation cannot run for hours).
                    grown += 1
                    max_tokens = min(max_tokens * 2, TRUNCATION_CEILING)
                    attempt["error"] = f"cut off at the output limit; retrying with max_tokens={max_tokens}"
                    continue
                attempt["error"] = "cut off at the output limit"
                raise OutputTruncated(
                    f"Structured output for role {role} was cut off at the output limit "
                    f"(max_tokens={max_tokens}); raise --max-output-tokens or reduce the batch size"
                )
            if not content.strip():
                last_error = RuntimeError(f"The chat model returned no JSON output for role {role}")
                attempt["error"] = "empty output"
                if repairs >= self.repair_attempts:
                    break
                repairs += 1
                continue
            try:
                return response_model.model_validate_json(extract_json_text(content))
            except ValidationError as exc:
                # The schema is enforced locally; give the model its errors once or twice before failing.
                last_error = exc
                attempt["error"] = format_validation_errors(exc)
                if repairs >= self.repair_attempts:
                    break
                repairs += 1
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": repair_instruction(format_validation_errors(exc))},
                ]
        raise StructuredOutputError(f"Structured output for role {role} failed validation after repair attempts: {last_error}") from last_error


class OpenAIChatLLM:
    """Free-text Chat Completions adapter; the defaults mirror the official AgentWorldBench script."""

    def __init__(
        self,
        model: str,
        temperature: float | None = 0.6,
        max_tokens: int | None = 32768,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        provider_routing: dict[str, Any] | None = None,
    ):
        self.client = _openai_client(client, base_url, api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning = reasoning
        self.provider_routing = provider_routing  # OpenRouter upstream pin; part of the cache identity
        self._local = threading.local()

    @property
    def last_usage(self) -> dict[str, float] | None:
        """Token counts of this thread's most recent chat() call."""
        return getattr(self._local, "usage", None)

    @last_usage.setter
    def last_usage(self, value: dict[str, float] | None) -> None:
        self._local.usage = value

    def chat(self, *, messages: list[dict[str, str]], role: str) -> str:
        self.last_usage = None
        response = call_with_retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **_sampling(self.temperature, self.max_tokens, self.reasoning_effort, self.reasoning, self.provider_routing),
        ), role=role)
        self.last_usage = usage_dict(getattr(response, "usage", None), provider=_response_provider(response))
        return response.choices[0].message.content or ""


class OpenAIToolLLM:
    """Native function calling over Chat Completions (OpenAI, vLLM, SGLang, and other compatible servers)."""

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = 8192,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        provider_routing: dict[str, Any] | None = None,
    ):
        self.client = _openai_client(client, base_url, api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning = reasoning
        self.provider_routing = provider_routing  # OpenRouter upstream pin; part of the cache identity

    def respond(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: str,
        force_tool: str | None = None,
    ) -> AgentReply:
        tool_choice: Any = {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
        # When a tool is forced, offer only that tool: gateways that ignore a named tool_choice (observed with
        # moonshotai/kimi-k2.6 via OpenRouter) then still cannot answer with a different call.
        offered = ([tool for tool in tools if tool.get("function", {}).get("name") == force_tool] or tools) if force_tool else tools
        params = dict(model=self.model, messages=messages, tools=offered,
                      **_sampling(self.temperature, self.max_tokens, self.reasoning_effort, self.reasoning, self.provider_routing))
        from openai import BadRequestError  # local import: the openai package stays optional for scripted runs

        forced_unsupported = False
        try:
            response = self.client.chat.completions.create(tool_choice=tool_choice, **params)
        except BadRequestError as exc:
            # Some providers reject a named tool_choice (DeepSeek's thinking mode: "Thinking mode does not support this
            # tool_choice"). Retry with automatic choice while still offering only the forced tool; the agent loop's own
            # reminder handles a model that then answers with something else.
            if not force_tool or "tool_choice" not in str(exc):
                raise
            forced_unsupported = True
            response = self.client.chat.completions.create(tool_choice="auto", **params)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            raw = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {"_raw": raw}
            calls.append(ToolCall(id=call.id, name=call.function.name,
                                  arguments=arguments if isinstance(arguments, dict) else {"value": arguments}))
        usage = getattr(response, "usage", None)
        return AgentReply(
            content=message.content,
            tool_calls=calls,
            usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                   **({"forced_tool_unsupported": 1} if forced_unsupported else {})} if usage else ({"forced_tool_unsupported": 1} if forced_unsupported else {}),
        )


class StructuredAgentLLM:
    """Drive the agent loop through a StructuredLLM by asking for typed ``AgentTurn`` decisions.

    Serves providers without native function calling and reuses the structured-call cache: each
    turn serializes the conversation so far and the tool catalog into one structured request.
    """

    def __init__(self, inner: StructuredLLM, role: str = "runtime_agent_turn"):
        self.inner = inner
        self.role = role
        self.counter = 0

    def respond(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: str,
        force_tool: str | None = None,
    ) -> AgentReply:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        # Static content first (tool catalog, then the brief that opens the conversation) so the
        # provider's prefix cache covers everything that does not change between the turns of one step.
        payload = {
            "tools": [tool["function"] for tool in tools if tool["function"]["name"] != SUBMIT_TOOL_NAME],
            "conversation": [message for message in messages if message["role"] != "system"],
            "instruction": "Return the final submission now; no further tool calls are allowed."
            if force_tool else "Choose exactly one tool call, or return the final submission.",
        }
        if force_tool == SUBMIT_TOOL_NAME:
            # At budget exhaustion the only acceptable answer is the submission itself, so ask for exactly that
            # schema: the "tool or final" choice confused models that reason inside the free `tool` string.
            payload["instruction"] = ("The tool budget is exhausted. Return the transition submission itself "
                                      "(effects, outcome, observation, rule_ids, citations); no tool call is possible.")
            submission = self.inner.complete(
                system=system + "\n\nRespond with one TransitionSubmission JSON object.",
                user=json.dumps(payload, ensure_ascii=False, default=str),
                response_model=TransitionSubmission,
                role=self.role,
            )
            self.counter += 1
            return AgentReply(content=None, tool_calls=[ToolCall(id=f"call_{self.counter}", name=SUBMIT_TOOL_NAME,
                                                                  arguments=submission.model_dump(mode="json"))])
        turn = self.inner.complete(
            system=system + "\n\nRespond with one AgentTurn: either `tool` with `arguments`, or `final` with the transition submission.",
            user=json.dumps(payload, ensure_ascii=False, default=str),
            response_model=AgentTurn,
            role=self.role,
        )
        self.counter += 1
        if turn.final is not None:
            call = ToolCall(id=f"call_{self.counter}", name=SUBMIT_TOOL_NAME, arguments=turn.final.model_dump(mode="json"))
        else:
            call = ToolCall(id=f"call_{self.counter}", name=str(turn.tool), arguments=turn.arguments)
        return AgentReply(content=turn.reason or None, tool_calls=[call])


def _requires_flexible_schema(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("additionalProperties") not in (None, False):
            return True
        structural_keys = {"type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const"}
        if value and not (structural_keys & value.keys()) and set(value) - {"title", "description", "default"} == set():
            return True
        return any(_requires_flexible_schema(item) for item in value.values())
    if isinstance(value, list):
        return any(_requires_flexible_schema(item) for item in value)
    return False


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _provider_identity(inner: Any) -> dict[str, Any]:
    identity = {
        "provider": type(inner).__qualname__,
        "model": getattr(inner, "model", None),
        "reasoning_effort": getattr(inner, "reasoning_effort", None),
        "temperature": getattr(inner, "temperature", None),
        "max_tokens": getattr(inner, "max_tokens", None),
    }
    reasoning = getattr(inner, "reasoning", None)
    if reasoning:  # only when set, so cache keys written before this field existed stay valid
        identity["reasoning"] = reasoning
    schema_mode = getattr(inner, "schema_mode", "json_schema")
    if schema_mode != "json_schema":  # same reason: the default leaves earlier keys untouched
        identity["schema_mode"] = schema_mode
    routing = getattr(inner, "provider_routing", None)
    if routing:  # a pinned upstream provider answers differently from the routed pool: never share cache entries
        identity["provider_routing"] = routing
    return identity


class CachedLLM:
    def __init__(self, inner: StructuredLLM, cache_dir: str | Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.last_hit: bool | None = None  # whether the most recent complete() was served from disk

    @property
    def last_usage(self) -> dict[str, float] | None:
        return None if self.last_hit else getattr(self.inner, "last_usage", None)

    @property
    def last_attempts(self) -> list[dict[str, Any]]:
        return [] if self.last_hit else list(getattr(self.inner, "last_attempts", []) or [])

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: str,
    ) -> T:
        fingerprint = _fingerprint(
            {"system": system, "user": user, "schema": response_model.model_json_schema(), "role": role,
             **_provider_identity(self.inner)}
        )
        path = self.cache_dir / f"{fingerprint}.json"
        if path.exists():
            self.last_hit = True
            return response_model.model_validate(read_json(path))
        self.last_hit = False
        result = self.inner.complete(system=system, user=user, response_model=response_model, role=role)
        write_json(path, result)
        return result


class TracingLLM:
    """Append every structured call — inputs, validated output, usage, timing, and errors — to a JSONL log.

    Wrap it outside a ``CachedLLM`` to log cache hits too (``cache_hit`` says which); the log is the
    audit trail for inspecting what a reconstruction stage actually asked and received.
    """

    def __init__(self, inner: StructuredLLM, log_path: str | Path):
        self.inner = inner
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._sequence = 0

    @property
    def last_usage(self) -> dict[str, float] | None:
        return getattr(self.inner, "last_usage", None)

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: str,
    ) -> T:
        started = time.monotonic()
        record: dict[str, Any] = {
            "call_id": _fingerprint({"system": system, "user": user, "role": role})[:16],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "response_model": response_model.__name__,
            "system": system,
            "user": user,
            "user_bytes": len(user.encode("utf-8")),
        }
        try:
            result = self.inner.complete(system=system, user=user, response_model=response_model, role=role)
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", elapsed_s=round(time.monotonic() - started, 3),
                          usage=self.last_usage, attempts=self._failed_attempts())
            self._write(record)
            raise
        record.update(response=result.model_dump(mode="json"), elapsed_s=round(time.monotonic() - started, 3),
                      usage=self.last_usage, cache_hit=getattr(self.inner, "last_hit", None),
                      attempts=self._failed_attempts())
        self._write(record)
        return result

    def _failed_attempts(self) -> list[dict[str, Any]]:
        """Raw outputs the provider had to repair (their validation errors explain malformed structured output)."""
        attempts = getattr(self.inner, "last_attempts", None) or []
        return [attempt for attempt in attempts if attempt.get("error")]

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            record["sequence"] = self._sequence
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class CachedChatLLM:
    def __init__(self, inner: ChatLLM, cache_dir: str | Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.last_hit = False

    @property
    def last_usage(self) -> dict[str, float] | None:
        """Usage of the most recent call when it went to the provider (a cache hit reports the stored usage, if any)."""
        return self._hit_usage if self.last_hit else getattr(self.inner, "last_usage", None)

    def chat(self, *, messages: list[dict[str, str]], role: str) -> str:
        fingerprint = _fingerprint({"messages": messages, "role": role, **_provider_identity(self.inner)})
        path = self.cache_dir / f"chat-{fingerprint}.json"
        if path.exists():
            cached = read_json(path)
            self.last_hit = True
            self._hit_usage = cached.get("usage")
            return str(cached["content"])
        self.last_hit = False
        content = self.inner.chat(messages=messages, role=role)
        write_json(path, {"content": content, "usage": getattr(self.inner, "last_usage", None)})
        return content


class TracingAgentLLM:
    """Append every world-model agent call (native function-calling transport) to the JSONL call log: the
    conversation, the offered tools, the forced tool, the reply with its tool calls, usage, timing, and errors.
    Wrap it outside a ``CachedAgentLLM`` so cache hits are logged too (``cache_hit``)."""

    def __init__(self, inner: AgentLLM, log_path: str | Path):
        self.inner = inner
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._sequence = 0

    def respond(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: str,
        force_tool: str | None = None,
    ) -> AgentReply:
        started = time.monotonic()
        user = json.dumps({"conversation": messages, "tools": tools, "force_tool": force_tool}, ensure_ascii=False, default=str)
        record: dict[str, Any] = {
            "call_id": _fingerprint({"messages": messages, "tools": tools, "role": role, "force_tool": force_tool})[:16],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "response_model": "AgentReply",
            "system": next((m.get("content") for m in messages if m.get("role") == "system"), ""),
            "user": user,
            "user_bytes": len(user.encode("utf-8")),
        }
        try:
            reply = self.inner.respond(messages=messages, tools=tools, role=role, force_tool=force_tool)
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", elapsed_s=round(time.monotonic() - started, 3), usage=None, attempts=[])
            self._write(record)
            raise
        record.update(response=reply.model_dump(mode="json"), elapsed_s=round(time.monotonic() - started, 3),
                      usage=reply.usage or None, cache_hit=getattr(self.inner, "last_hit", None), attempts=[])
        self._write(record)
        return reply

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            record["sequence"] = self._sequence
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class CachedAgentLLM:
    def __init__(self, inner: AgentLLM, cache_dir: str | Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)

    def respond(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: str,
        force_tool: str | None = None,
    ) -> AgentReply:
        fingerprint = _fingerprint(
            {"messages": messages, "tools": tools, "role": role, "force_tool": force_tool, **_provider_identity(self.inner)}
        )
        path = self.cache_dir / f"agent-{fingerprint}.json"
        if path.exists():
            self.last_hit = True
            return AgentReply.model_validate(read_json(path))
        self.last_hit = False
        reply = self.inner.respond(messages=messages, tools=tools, role=role, force_tool=force_tool)
        write_json(path, reply)
        return reply


class ScriptedLLM:
    """Test/dry-run provider returning queued structured objects by pipeline role."""

    def __init__(self, responses: dict[str, list[BaseModel | dict[str, Any]]]):
        self.responses = {key: list(values) for key, values in responses.items()}
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: str,
    ) -> T:
        self.calls.append({"role": role, "system": system, "user": user})
        queue = self.responses.get(role, [])
        if not queue:
            raise RuntimeError(f"No scripted response remains for role {role}")
        value = queue.pop(0)
        return value if isinstance(value, response_model) else response_model.model_validate(value)


class ScriptedChatLLM:
    """Test provider returning queued free-text answers by role."""

    def __init__(self, responses: dict[str, list[str]]):
        self.responses = {key: list(values) for key, values in responses.items()}
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, messages: list[dict[str, str]], role: str) -> str:
        self.calls.append({"role": role, "messages": messages})
        queue = self.responses.get(role, [])
        if not queue:
            raise RuntimeError(f"No scripted chat response remains for role {role}")
        return queue.pop(0)


class ScriptedToolLLM:
    """Test provider replaying queued agent replies (tool calls or text) in order."""

    def __init__(self, replies: list[AgentReply | dict[str, Any]]):
        self.replies = [reply if isinstance(reply, AgentReply) else AgentReply.model_validate(reply) for reply in replies]
        self.calls: list[dict[str, Any]] = []

    def respond(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: str,
        force_tool: str | None = None,
    ) -> AgentReply:
        self.calls.append({"role": role, "messages": list(messages), "tools": [tool["function"]["name"] for tool in tools],
                           "force_tool": force_tool})
        if not self.replies:
            raise RuntimeError("No scripted agent reply remains")
        return self.replies.pop(0)
