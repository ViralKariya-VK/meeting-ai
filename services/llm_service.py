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


_MODEL_FALLBACKS = ["llama-3.1-8b-instant", "openai/gpt-oss-20b"]
_working: dict = {}          # remembers the model that last worked
last_error: Optional[str] = None  # most recent failure text, surfaced in the UI for debugging

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


def candidate_models() -> list[str]:
    first = [_working["model"]] if _working.get("model") else []
    return list(dict.fromkeys(first + [config.LLM_MODEL()] + _MODEL_FALLBACKS))


def _complete(messages: list[dict], max_tokens: int, temperature: float, json_mode: bool) -> str:
    """Try the configured model, then fall back to other models and to non-JSON mode, so a
    deprecated/blocked model or a JSON-mode quirk degrades gracefully instead of failing every call."""
    global last_error
    client = get_client()
    errors: list[str] = []
    for model in candidate_models():
        for use_json in ([True, False] if json_mode else [False]):
            kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                text = _with_retry(lambda: client.chat.completions.create(**kwargs).choices[0].message.content or "")
                _working["model"] = model
                last_error = None
                return text.strip()
            except LLMUnavailable as e:
                errors.append(f"[{model}{' json' if use_json else ''}] {e}")
    last_error = " | ".join(errors)[-700:]
    raise LLMUnavailable(last_error)


def _with_retry(fn, attempts: int = 3):
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
    return _complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     max_tokens, temperature, json_mode=False)


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
    system = system + "\nReply with a single JSON object only."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = _complete(messages, max_tokens, 0.0, json_mode=True)
    for attempt in range(2):
        try:
            return schema.model_validate(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt == 1:
                raise LLMUnavailable(f"LLM returned invalid JSON: {str(e)[:150]}") from e
            raw = _complete(messages + [{"role": "assistant", "content": raw},
                                        {"role": "user", "content": f"That was invalid ({str(e)[:200]}). Return corrected JSON only."}],
                            max_tokens, 0.0, json_mode=True)
    raise LLMUnavailable("unreachable")
