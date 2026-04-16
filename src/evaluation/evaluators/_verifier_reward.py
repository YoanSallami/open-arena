"""Logprob-based reward helper for the LLM-as-a-Verifier evaluator.

Prompts the grader to emit one score letter between <score> tags, then
computes the expected scalar value under its `top_logprobs` distribution
at that position, averaged over K repeats. Implements G (granularity)
and K (repeats).
"""

import logging
import math
import os
from typing import Any

import litellm

_logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_OPENAI_BASE = "http://localhost:11434/v1"


def _coerce_ollama_to_openai_compat(cfg: dict[str, Any]) -> dict[str, Any]:
    """Route Ollama calls through the OpenAI-compatible endpoint.

    LiteLLM's dedicated `ollama`/`ollama_chat` providers currently reject
    `logprobs`/`top_logprobs` (see BerriAI/litellm#17165), but Ollama itself
    (>= 0.12.11) returns them on `/v1/chat/completions`. Rewriting the
    provider prefix to `openai/` with `api_base=<ollama>/v1` lets the
    params flow through untouched.
    """
    model = cfg.get("model")
    if not isinstance(model, str):
        return cfg
    for prefix in ("ollama_chat/", "ollama/"):
        if model.startswith(prefix):
            bare = model[len(prefix) :]
            rewritten = dict(cfg)
            rewritten["model"] = f"openai/{bare}"
            rewritten.setdefault(
                "api_base", os.getenv("OLLAMA_API_BASE", _DEFAULT_OLLAMA_OPENAI_BASE)
            )
            rewritten.setdefault("api_key", "ollama")
            _logger.info(
                "verifier: routing %s via OpenAI-compat endpoint %s (logprobs support)",
                model,
                rewritten["api_base"],
            )
            return rewritten
    return cfg


def build_score_tokens(granularity: int) -> list[str]:
    """Score letters ordered best → worst (A is best)."""
    if granularity < 2 or granularity > 26:
        raise ValueError(f"granularity must be in [2, 26]; got {granularity}")
    return [chr(ord("A") + i) for i in range(granularity)]


async def verifier_reward(
    llm_config: dict[str, Any],
    system_prompt: str,
    user_payload: str,
    granularity: int = 8,
    repeats: int = 1,
) -> tuple[float | None, str | None]:
    """Sample the verifier `repeats` times and return the average expected score.

    Returns `(reward, error)`. If every sample fails (no logprobs, API
    error, no score token found), `reward` is None and `error` holds the
    most useful message.
    """
    score_tokens = build_score_tokens(granularity)
    n = len(score_tokens)
    value_map = {tok: float(n - i) for i, tok in enumerate(score_tokens)}

    completion_kwargs = {k: v for k, v in llm_config.items() if v is not None}
    completion_kwargs = _coerce_ollama_to_openai_compat(completion_kwargs)
    completion_kwargs.pop("tools", None)
    completion_kwargs.pop("tool_choice", None)
    completion_kwargs["logprobs"] = True
    completion_kwargs["top_logprobs"] = granularity
    completion_kwargs["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    rewards: list[float] = []
    errors: list[str] = []
    for _ in range(max(1, repeats)):
        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as e:
            errors.append(f"completion failed: {e}")
            continue
        dist = _extract_score_distribution(response, set(score_tokens))
        if not dist:
            errors.append("no score-token logprobs in response")
            continue
        expected = sum(prob * value_map[tok] for tok, prob in dist.items())
        rewards.append(expected)

    if not rewards:
        return None, "; ".join(dict.fromkeys(errors)) or "no usable samples"
    return sum(rewards) / len(rewards), None


def _score_letter(token: str, score_tokens: set[str]) -> str | None:
    """Return the score letter carried by a token, or None.

    Handles three tokenizer patterns we see in practice:
    * a bare letter (`"A"`),
    * a letter fused with surrounding punctuation (`">A"`, `":A"`, `"\tA"`),
    * a letter with trailing whitespace (`"A "`, `"A\n"`).
    Anything where the letter is adjacent to another letter (e.g. `"Alpha"`)
    is rejected to avoid false positives.
    """
    stripped = (token or "").strip()
    if not stripped:
        return None
    if stripped in score_tokens:
        return stripped
    last = stripped[-1]
    if last in score_tokens and (len(stripped) == 1 or not stripped[-2].isalpha()):
        return last
    return None


def _extract_score_distribution(
    response: Any, score_tokens: set[str]
) -> dict[str, float] | None:
    """Find the first response token that carries a score letter, then
    build a normalized probability distribution from its `top_logprobs`.

    Because tokenizers often fuse the score letter with adjacent
    punctuation (e.g. `">A"`), we group `top_logprobs` entries by the
    score letter they carry and sum their probabilities per letter.
    """

    choices = _get(response, "choices", default=[]) or []
    if not choices:
        return None
    logprobs = _get(choices[0], "logprobs", default=None)
    if logprobs is None:
        return None
    content = _get(logprobs, "content", default=None) or []

    for token_entry in content:
        tok_text = _get(token_entry, "token", default="") or ""
        if _score_letter(tok_text, score_tokens) is None:
            continue
        top = _get(token_entry, "top_logprobs", default=[]) or []
        probs: dict[str, float] = {}
        for entry in top:
            t = _get(entry, "token", default="") or ""
            lp = _get(entry, "logprob", default=None)
            letter = _score_letter(t, score_tokens)
            if letter is None or lp is None:
                continue
            probs[letter] = probs.get(letter, 0.0) + math.exp(float(lp))
        if not probs:
            continue
        total = sum(probs.values())
        if total <= 0:
            continue
        return {k: v / total for k, v in probs.items()}
    return None


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
