"""llm.py — model-agnostic chat adapter for the OpenAI model family.

The June 2026 harness hard-coded ``gpt-4o`` with ``temperature=0.1``. The
resubmission compares several generators, and the GPT-5 / o-series
reasoning models reject ``temperature`` and ``max_tokens`` (they take
``reasoning_effort`` and ``max_completion_tokens`` instead). This module hides
that difference, retries transient API errors with exponential back-off, and
returns latency and token usage for every call so cost can be reported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_PREFIXES)


@dataclass
class ChatResult:
    text: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    snapshot: str          # dated model id actually served (e.g. gpt-4o-2024-11-20)
    finish_reason: str


def chat(
    client,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    json_mode: bool = False,
    reasoning_effort: str = "low",
    retries: int = 6,
) -> ChatResult:
    """One chat completion with per-family parameters and retry."""
    kwargs: dict = {"model": model, "messages": messages}
    if is_reasoning_model(model):
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["max_completion_tokens"] = max_tokens or 2000
    else:
        kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — classified by name below
            last_err = e
            name = type(e).__name__
            if name in ("BadRequestError", "AuthenticationError", "PermissionDeniedError"):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        latency = time.time() - t0
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text and is_reasoning_model(model) and attempt < retries - 1:
            # Reasoning tokens exhausted the budget before any visible output.
            kwargs["max_completion_tokens"] = int(kwargs["max_completion_tokens"] * 2)
            continue
        usage = resp.usage
        return ChatResult(
            text=text,
            latency_s=latency,
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            snapshot=str(getattr(resp, "model", model) or model),
            finish_reason=str(choice.finish_reason or ""),
        )
    raise RuntimeError(f"chat() failed after {retries} attempts: {last_err}")
