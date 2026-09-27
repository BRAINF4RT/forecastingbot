"""Direct OpenRouter client used by the Metaculus forecasting bot."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
import weakref

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Forecast/research fallback chain. All three are free OpenRouter endpoints.
PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL = "poolside/laguna-s-2.1:free"
THIRD_MODEL = "qwen/qwen3.8-27b:free"

_OPENROUTER_CONCURRENCY = 1


class _LoopBoundSemaphore:
    """Keep one semaphore per asyncio event loop.

    asyncio primitives can become associated with the first loop that waits on
    them. The bot may be invoked more than once by tests, so a per-loop pool
    avoids cross-event-loop binding errors.
    """

    def __init__(self, value: int) -> None:
        self._value = value
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()

    def _get(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._value)
            self._semaphores[loop] = semaphore
        return semaphore

    async def __aenter__(self) -> None:
        await self._get().acquire()

    async def __aexit__(self, *_: object) -> None:
        self._get().release()


_OPENROUTER_SEMAPHORE = _LoopBoundSemaphore(_OPENROUTER_CONCURRENCY)


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails."""


def _require_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")
    return api_key


def _normalise_model(model: str) -> str:
    """Normalise a LiteLLM-style OpenRouter model name for direct API use."""
    model = (model or "").strip()
    # `openrouter/free` is OpenRouter's own router ID; the prefix is part of
    # the real model ID and must NOT be stripped.
    while model.startswith("openrouter/openrouter/"):
        model = model[len("openrouter/") :]
    if model == "openrouter/free":
        return model
    if model.startswith("openrouter/"):
        return model[len("openrouter/") :]
    return model


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _extract_error_message(data: Any) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
            return str(error)
        if error:
            return str(error)
    return str(data)


def _extract_content(data: dict[str, Any], model: str) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(
            f"Unexpected OpenRouter response from {model}: {data!r}"
        ) from exc

    if isinstance(content, list):
        # Be tolerant of OpenAI/OpenRouter content-part responses.
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        content = "\n".join(parts)

    if not content or not str(content).strip():
        raise OpenRouterError(
            f"OpenRouter returned an empty response from {model}."
        )
    return str(content).strip()


async def _generate_with_model(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: float = 180.0,
    max_retries: int = 3,
) -> str:
    """Make a direct OpenRouter request to one model with bounded retries."""
    api_key = _require_api_key()
    api_model = _normalise_model(model)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/BRAINF4RT/forecastingbot",
        "X-Title": "BRAINF4RT Metaculus Forecasting Bot",
    }

    last_error: Exception | None = None
    async with _OPENROUTER_SEMAPHORE:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    response = await client.post(
                        OPENROUTER_URL,
                        headers=headers,
                        json=payload,
                    )
                    try:
                        data = response.json()
                    except ValueError:
                        data = None

                    if response.status_code != 200:
                        message = _extract_error_message(
                            data if data is not None else response.text
                        )
                        error = OpenRouterError(
                            f"OpenRouter returned HTTP {response.status_code} "
                            f"from {api_model}: {message}"
                        )
                        last_error = error
                        if (
                            not _is_retryable_status(response.status_code)
                            or attempt >= max_retries
                        ):
                            raise error
                        wait = min(2**attempt, 30)
                        logger.warning(
                            "OpenRouter request failed on %s (attempt %d/%d): %s. "
                            "Retrying in %ds.",
                            api_model,
                            attempt,
                            max_retries,
                            message,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    if not isinstance(data, dict):
                        raise OpenRouterError(
                            f"Unexpected OpenRouter response from {api_model}: {data!r}"
                        )
                    if data.get("error"):
                        message = _extract_error_message(data)
                        error = OpenRouterError(
                            f"Unexpected OpenRouter response from {api_model}: {message}"
                        )
                        last_error = error
                        if attempt >= max_retries:
                            raise error
                        wait = min(2**attempt, 30)
                        logger.warning(
                            "OpenRouter returned an error on %s (attempt %d/%d): %s. "
                            "Retrying in %ds.",
                            api_model,
                            attempt,
                            max_retries,
                            message,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    content = _extract_content(data, api_model)
                    logger.info(
                        "OpenRouter generation succeeded using %s", api_model
                    )
                    return content

                except httpx.TransportError as exc:
                    last_error = exc
                    logger.warning(
                        "OpenRouter transport failure on %s (attempt %d/%d): %s",
                        api_model,
                        attempt,
                        max_retries,
                        exc,
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(min(2**attempt, 30))
                        continue
                    raise

    raise OpenRouterError(
        f"OpenRouter request failed on {api_model} after {max_retries} attempts: "
        f"{last_error}"
    )


async def generate(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: float = 180.0,
    max_retries: int = 3,
) -> str:
    """Generate using Nemotron -> Laguna -> Qwen."""
    errors: list[tuple[str, Exception]] = []
    for model in (PRIMARY_MODEL, FALLBACK_MODEL, THIRD_MODEL):
        try:
            return await _generate_with_model(
                prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as exc:
            errors.append((model, exc))
            logger.warning(
                "OpenRouter model %s failed after %d attempts: %s",
                model,
                max_retries,
                exc,
            )

    details = "\n".join(f"{model}: {error!r}" for model, error in errors)
    raise OpenRouterError(
        "All OpenRouter reasoning models failed.\n" + details
    ) from errors[-1][1]


async def summarize_research(
    question_text: str,
    resolution_criteria: str,
    background: str,
    raw_research: str,
    max_tokens: int = 32000,
) -> str:
    """Summarise retrieved web evidence into a forecasting brief."""
    if not raw_research.strip():
        return ""

    prompt = f"""
You are the research-analysis specialist for a professional forecasting bot.

FORECASTING QUESTION:
{question_text}

RESOLUTION CRITERIA (this defines exactly what counts as relevant):
{resolution_criteria}

BACKGROUND:
{background}

WEB RESEARCH:
{raw_research[:30000]}

Create a concise factual research brief for another forecaster.

Requirements:
- Judge relevance against the exact resolution criteria, not merely the topic.
- Discard material that does not help determine this question's outcome.
- Preserve important dates, numbers, percentages, estimates and named sources.
- Distinguish established facts from uncertainty.
- Highlight recent developments that materially affect resolution.
- Mention source domains when possible.
- Explicitly identify disagreements between sources.
- Never invent facts or unsupported predictions.
- Keep the briefing under approximately 900 words.
"""

    return await generate(
        prompt,
        system_prompt=(
            "You are an evidence-focused research analyst. Extract only "
            "information that bears on how the exact Metaculus question "
            "will resolve. Never invent facts."
        ),
        temperature=0.15,
        max_tokens=max_tokens,
        timeout=240.0,
        max_retries=3,
    )


async def generate_forecast_reasoning(
    prompt: str,
    *,
    temperature: float = 0.15,
    max_tokens: int = 10000,
) -> str:
    """Generate the forecasting model's reasoning and final formatted answer."""
    return await generate(
        prompt,
        system_prompt=(
            "You are an expert probabilistic forecaster. Produce calibrated "
            "forecasts, carefully follow the resolution criteria, distinguish "
            "known facts from uncertainty, consider base rates and plausible "
            "alternatives, and obey the requested final-answer format exactly."
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=240.0,
        max_retries=3,
    )
