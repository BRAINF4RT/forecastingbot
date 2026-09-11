"""
Client for the bot's "main brain" forecasting model.

Calls WeiboAI/VibeThinker-3B through Hugging Face Inference Providers,
routed to the Featherless AI backend:
https://huggingface.co/WeiboAI/VibeThinker-3B?inference_provider=featherless-ai

HF's router exposes an OpenAI-compatible /v1/chat/completions endpoint at
https://router.huggingface.co/v1/chat/completions. The provider is selected
with a ":<provider>" suffix on the model id (e.g. "WeiboAI/VibeThinker-3B:featherless-ai").

ARCHITECTURAL RULE
-------------------
This model is the bot's *forecasting* brain only. It must never be asked to
generate web-search queries or summarize search results — those jobs belong
to clients/openrouter_helper.py. Only call `generate_forecast_reasoning` from
the forecast-writing code in bot.py; never from research/pipeline.py.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


class HuggingFaceBrainError(RuntimeError):
    pass


def _model_id() -> str:
    model = os.getenv("HF_MAIN_BRAIN_MODEL", "WeiboAI/VibeThinker-3B")
    provider = os.getenv("HF_INFERENCE_PROVIDER", "featherless-ai")
    return f"{model}:{provider}"


async def generate_forecast_reasoning(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4000,
    timeout: float = 180.0,
    max_retries: int = 3,
) -> str:
    """Call VibeThinker-3B and return its raw text output.

    VibeThinker-3B is a small reasoning model that tends to produce long
    chain-of-thought before its final answer, so max_tokens defaults high
    and the timeout is generous.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise HuggingFaceBrainError(
            "HF_TOKEN is not set. Create a fine-grained token with the "
            "'Make calls to Inference Providers' permission at "
            "https://huggingface.co/settings/tokens"
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": _model_id(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(HF_ROUTER_URL, headers=headers, json=payload)
                if resp.status_code == 429:
                    wait = min(2**attempt, 30)
                    logger.warning(
                        "VibeThinker-3B rate limited (attempt %d/%d), retrying in %ds",
                        attempt,
                        max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content") or ""
                # Some reasoning-model backends split chain-of-thought into a
                # separate 'reasoning' field from the final 'content'. Stitch
                # them together so downstream parsing can find the final
                # "Probability: ZZ%" (or equivalent) line either way.
                reasoning_field = choice.get("reasoning")
                if reasoning_field and reasoning_field not in content:
                    content = f"{reasoning_field}\n\n{content}"
                if not content.strip():
                    raise HuggingFaceBrainError("Empty response from VibeThinker-3B")
                return content
            except (httpx.HTTPStatusError, httpx.TransportError, HuggingFaceBrainError) as exc:
                last_error = exc
                logger.warning(
                    "VibeThinker-3B call failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(min(2**attempt, 30))

    raise HuggingFaceBrainError(
        f"VibeThinker-3B call failed after {max_retries} attempts: {last_error}"
    )
