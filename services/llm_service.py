"""Single entry point for Groq chat completions. Everything else depends on this module only."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """No API key, or Groq could not be reached after retries."""


_lock = threading.Lock()
_last_call = 0.0
_MIN_INTERVAL = 1.0  # seconds between calls: crude token bucket for the free tier
_budget_hook: Optional[Callable[[int], None]] = None


def set_budget_hook(hook: Optional[Callable[[int], None]]) -> None:
    """Pages register a per-session cap (raises LimitReached when exhausted)."""
    global _budget_hook
    _budget_hook = hook


def is_configured() -> bool:
    return bool(config.GROQ_API_KEY())


def get_client():
    key = config.GROQ_API_KEY()
    if not key:
        raise LLMUnavailable("GROQ_API_KEY is not set.")
    from groq import Groq

    return Groq(api_key=key, timeout=60.0, max_retries=0)


def _throttle() -> None:
    global _last_call
    with _lock:
        wait = _MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _with_retry(fn, attempts: int = 4):
    delay = 2.0
    last: Exception | None = None
    for _ in range(attempts):
        try:
            _throttle()
            return fn()
        except LLMUnavailable:
            raise
        except Exception as e:  # groq.RateLimitError, APIConnectionError, ...
            last = e
            name = type(e).__name__
            status = getattr(e, "status_code", None)
            retryable = name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"} \
                or status in (429, 500, 502, 503)
            if not retryable:
                raise LLMUnavailable(f"Groq request failed: {name}: {str(e)[:200]}") from e
            time.sleep(delay)
            delay *= 2
    raise LLMUnavailable(f"Groq unavailable or rate limited after retries ({type(last).__name__}).")


def chat_text(system: str, user: str, max_tokens: int = 700, temperature: float = 0.2) -> str:
    if _budget_hook:
        _budget_hook(1)
    client = get_client()

    def call():
        r = client.chat.completions.create(
            model=config.LLM_MODEL(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return (r.choices[0].message.content or "").strip()

    return _with_retry(call)


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def chat_json(system: str, user: str, schema: Type[T], max_tokens: int = 600) -> T:
    """JSON-mode call validated against a pydantic schema. One repair attempt on invalid output."""
    if _budget_hook:
        _budget_hook(1)
    client = get_client()
    system = system + "\nReply with a single JSON object only."

    def call():
        r = client.chat.completions.create(
            model=config.LLM_MODEL(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.0, response_format={"type": "json_object"},
        )
        return (r.choices[0].message.content or "").strip()

    raw = _with_retry(call)
    for attempt in range(2):
        try:
            return schema.model_validate(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt == 1:
                raise LLMUnavailable(f"LLM returned invalid JSON: {str(e)[:150]}") from e
            raw = _with_retry(lambda: client.chat.completions.create(
                model=config.LLM_MODEL(),
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user},
                          {"role": "assistant", "content": raw},
                          {"role": "user", "content": f"That was invalid ({str(e)[:200]}). Return corrected JSON only."}],
                max_tokens=max_tokens, temperature=0.0, response_format={"type": "json_object"},
            ).choices[0].message.content or "")
    raise LLMUnavailable("unreachable")
