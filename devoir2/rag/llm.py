"""LLM backends: Groq (cloud) and MLX (local, Apple Silicon). Same `chat()` interface."""
from __future__ import annotations

import re
import threading
import time
from functools import lru_cache

from . import config

# Groq free tier: per-request output must fit the model's output-tokens-per-minute limit.
_OUTPUT_CAP = {"qwen/qwen3.8-27b": 950}


def available_models() -> list[str]:
    models = list(config.GROQ_MODELS) if config.GROQ_API_KEY else []
    try:
        import mlx_lm  # noqa: F401

        models += list(config.LOCAL_MODELS)
    except ImportError:
        pass
    return models


class QuotaExceeded(RuntimeError):
    """Daily quota of a cloud model is exhausted (retrying now is pointless)."""


def chat(model: str, messages: list[dict], max_tokens: int = 700, temperature: float = 0.0,
         fallback: bool = False) -> str:
    """Call one model. With fallback=True, walk config.FALLBACK_CHAIN when the model is out of quota."""
    chain = [model] + ([m for m in config.FALLBACK_CHAIN if m != model] if fallback else [])
    last = None
    for m in chain:
        try:
            if m in config.LOCAL_MODELS:
                return _mlx_chat(config.LOCAL_MODELS[m], messages, max_tokens)
            if m == "gemini" or m.startswith("gemini-"):
                return _gemini_chat(config.GEMINI_MODEL if m == "gemini" else m, messages, max_tokens, temperature)
            return _groq_chat(config.GROQ_MODELS.get(m, m), messages, max_tokens, temperature)
        except QuotaExceeded as e:
            last = e
            continue
    raise last or RuntimeError("no model available")


# ── Groq ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing: add it to devoir2/.env or use a local model.")
    return Groq(api_key=config.GROQ_API_KEY, max_retries=0, timeout=60)


def _groq_chat(model_id: str, messages, max_tokens, temperature, retries: int = 6) -> str:
    extra = {}
    if model_id.startswith("openai/gpt-oss"):
        extra["reasoning_effort"] = "low"
        max_tokens += 600  # reasoning tokens count against max_tokens
    elif model_id.startswith("qwen/qwen3"):
        extra["reasoning_effort"] = "none"  # no hidden thinking: answer directly
    max_tokens = min(max_tokens, _OUTPUT_CAP.get(model_id, max_tokens))
    delay = 2.0
    for attempt in range(retries):
        try:
            r = _groq_client().chat.completions.create(
                model=model_id, messages=messages, max_tokens=max_tokens,
                temperature=temperature, **extra)
            return _strip_think(r.choices[0].message.content or "")
        except Exception as e:  # rate limit / transient errors: back off
            msg = str(e)
            if "tokens per day" in msg or "requests per day" in msg or "(TPD)" in msg or "(RPD)" in msg:
                raise QuotaExceeded(msg) from e
            retryable = any(s in msg for s in ("429", "rate_limit", "503", "502", "timeout", "Timeout", "Connection"))
            if not retryable or attempt == retries - 1:
                raise
            m = re.search(r"try again in ([\d.]+)s", msg)
            time.sleep(float(m.group(1)) + 0.5 if m else delay)
            delay = min(delay * 2, 30)
    return ""


# ── Gemini (REST, no extra dependency) ──────────────────────────────────────
def _gemini_chat(model_id: str, messages, max_tokens, temperature, retries: int = 5) -> str:
    import json as _json
    import urllib.error
    import urllib.request

    if not config.GEMINI_API_KEY:
        raise QuotaExceeded("GEMINI_API_KEY missing")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    body = {
        "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                     for m in messages if m["role"] != "system"],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens + 4096},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    req = urllib.request.Request(url, data=_json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY})
    delay = 3.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = _json.loads(r.read())
            parts = data["candidates"][0]["content"].get("parts", [])
            return _strip_think("".join(p.get("text", "") for p in parts))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")
            if e.code == 429 and ("PerDay" in msg or "per day" in msg.lower()):
                raise QuotaExceeded(msg[:300]) from e
            if e.code not in (429, 500, 503) or attempt == retries - 1:
                raise RuntimeError(f"Gemini {e.code}: {msg[:300]}") from e
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return ""


# ── MLX (local) ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def _mlx_load(model_id: str):
    from mlx_lm import load

    return load(model_id)


_MLX_LOCK = threading.Lock()  # MLX generation is not thread-safe


def _mlx_chat(model_id: str, messages, max_tokens) -> str:
    with _MLX_LOCK:
        return _mlx_chat_unlocked(model_id, messages, max_tokens)


def _mlx_chat_unlocked(model_id: str, messages, max_tokens) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = _mlx_load(model_id)
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0), verbose=False)
    return _strip_think(out)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
