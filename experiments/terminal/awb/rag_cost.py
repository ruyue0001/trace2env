#!/usr/bin/env python3
"""Cost and size of the prompting+rag run versus plain prompting (whose run recorded no usage).

Plain prompting's prompt size is deterministic (official messages), so its tokens are estimated from the
tokens-per-character ratio measured on the prompting+rag rows; completion tokens are assumed equal.
"""
import json
from statistics import mean

A = "work/exp-v1_20_r=1/awb"
PRICE = {"prompt": 2e-6, "completion": 1e-5, "cached": 2e-7}
import sys
sys.path.insert(0, "src")
from trace2env.agentworld import case_from_row, inference_messages  # noqa: E402

rag = [json.loads(l) for l in open(f"{A}/pred-prompting+rag.jsonl")]
usage = [r["trace2env"].get("usage") or {} for r in rag]
prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage)
completion_tokens = sum(u.get("completion_tokens", 0) for u in usage)
reasoning = sum(u.get("reasoning_tokens", 0) for u in usage)
reported = sum(u.get("cost_usd", 0.0) for u in usage)
chars = sum(r["trace2env"].get("prompt_chars", 0) for r in rag)
ratio = prompt_tokens / chars if chars else 0
plain_chars = sum(sum(len(m["content"]) for m in inference_messages(case_from_row(r))) for r in rag)
plain_prompt_tokens = plain_chars * ratio
listed = lambda p, c: p * PRICE["prompt"] + c * PRICE["completion"]
print(f"prompting+rag: rows={len(rag)} prompt tokens={prompt_tokens} ({prompt_tokens/len(rag):.0f}/row) completion={completion_tokens} "
      f"({completion_tokens/len(rag):.0f}/row, reasoning {reasoning/len(rag):.0f}) reported cost=${reported:.2f} (${reported/len(rag):.3f}/row) "
      f"list-price cost=${listed(prompt_tokens, completion_tokens):.2f}; prompt chars/row={chars/len(rag):.0f}; retrieved block chars/row="
      f"{mean(r['trace2env']['rag']['block_chars'] for r in rag):.0f}; hits/row={mean(len(r['trace2env']['rag']['hits']) for r in rag):.2f}")
print(f"plain prompting (estimated at {ratio:.3f} tokens/char): prompt tokens≈{plain_prompt_tokens:.0f} ({plain_prompt_tokens/len(rag):.0f}/row), "
      f"list-price cost≈${listed(plain_prompt_tokens, completion_tokens):.2f} (${listed(plain_prompt_tokens, completion_tokens)/len(rag):.3f}/row) "
      f"assuming the same completion length")
lat = {name: mean(json.loads(l)["trace2env"]["latency_seconds"] for l in open(f"{A}/pred-{name}.jsonl")) for name in ("prompting", "prompting+rag")}
print("latency s/row:", {k: round(v, 1) for k, v in lat.items()})
