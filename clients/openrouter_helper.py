"""
Direct OpenRouter client used by the Metaculus forecasting bot.

LLM architecture:

    Research summarisation:
        nvidia/nemotron-3-ultra-550b-a55b:free
            -> poolside/laguna-s-2.1:free

    Forecast reasoning:
        nvidia/nemotron-3-ultra-550b-a55b:free
            -> poolside/laguna-s-2.1:free

Search-query generation is deliberately NOT performed by an LLM. The caller
builds deterministic queries directly from the Metaculus question text.

This module also rate-limits direct OpenRouter requests because the free
endpoints can become overloaded when many Metaculus questions are processed
at once.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL = "poolside/laguna-s-2.1:free"
THIRD_MODEL = "qwen/qwen3.8-27b:free"

_OPENROUTER_CONCURRENCY = 2
_OPENROUTER_SEMAPHORE = asyncio.Semaphore(_OPENROUTER_CONCURRENCY)


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails."""


def _require_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")
    return api_key


def _normalise_model(model: str) -> str:
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
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
    """Make a direct OpenRouter request to one specific model."""
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
                        if _is_retryable_status(response.status_code) and attempt < max_retries:
                            wait = min(2 ** attempt, 30)
                            logger.warning(
                                "OpenRouter request failed on %s (attempt %d/%d): %s. Retrying in %ds.",
                                api_model, attempt, max_retries, message, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise error

                    if not isinstance(data, dict):
                        raise OpenRouterError(
                            f"Unexpected OpenRouter response from {api_model}: {data!r}"
                        )

                    if data.get("error"):
                        message = _extract_error_message(data)
                        error = OpenRouterError(
                            f"Unexpected OpenRouter response from {api_model}: {data!r}"
                        )
                        last_error = error
                        if attempt < max_retries:
                            wait = min(2 ** attempt, 30)
                            logger.warning(
                                "OpenRouter request failed on %s (attempt %d/%d): %s. Retrying in %ds.",
                                api_model, attempt, max_retries, message, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise error

                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError) as exc:
                        raise OpenRouterError(
                            f"Unexpected OpenRouter response from {api_model}: {data!r}"
                        ) from exc

                    if not content or not str(content).strip():
                        raise OpenRouterError(
                            f"OpenRouter returned an empty response from {api_model}."
                        )

                    logger.info("OpenRouter generation succeeded using %s", api_model)
                    return str(content).strip()

                except httpx.TransportError as exc:
                    last_error = exc
                    logger.warning(
                        "OpenRouter transport failure on %s (attempt %d/%d): %s",
                        api_model, attempt, max_retries, exc,
                    )
                    if attempt < max_retries:
                        wait = min(2 ** attempt, 30)
                        await asyncio.sleep(wait)

                except OpenRouterError as exc:
                    last_error = exc
                    if attempt >= max_retries:
                        break
                    logger.warning(
                        "OpenRouter request failed on %s (attempt %d/%d): %s",
                        api_model, attempt, max_retries, exc,
                    )
                    wait = min(2 ** attempt, 30)
                    await asyncio.sleep(wait)

    raise OpenRouterError(
        f"OpenRouter request failed on {api_model} after "
        f"{max_retries} attempts: {last_error}"
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
    """Generate using Nemotron -> Laguna -> OpenRouter free-router."""
    errors: list[tuple[str, Exception]] = []
    for model in (PRIMARY_MODEL, FALLBACK_MODEL, THIRD_MODEL):
        try:
            result = await _generate_with_model(
                prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
            logger.info("OpenRouter model %s successfully completed the request.", model)
            return result
        except Exception as exc:
            errors.append((model, exc))
            logger.warning(
                "OpenRouter model %s failed after %d attempts: %s",
                model, max_retries, exc,
            )
    details = "\n".join(f"{model}: {error!r}" for model, error in errors)
    raise OpenRouterError("All OpenRouter reasoning models failed.\n" + details) from errors[-1][1]


async def summarize_research(
    question_text: str,
    resolution_criteria: str,
    background: str,
    raw_research: str,
    max_tokens: int = 32000,
) -> str:
    """Summarise scraped research into a concise forecasting brief."""
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

Below is information collected from web searches.

RESEARCH:
{raw_research[:20000]}

Create a concise factual research brief for another forecaster.
Requirements:
- Judge relevance strictly against the RESOLUTION CRITERIA above, not the
  general topic. A source can be about the right subject and still be
  irrelevant if it doesn't bear on how THIS question resolves.
- Discard anything that doesn't help determine the specific outcome this
  question asks about -- generic background the forecaster already has,
  off-topic search hits, and duplicate information should all be dropped
  rather than summarized.
- Separate established facts from uncertainty.
- Preserve important dates, numbers, percentages and estimates -- but only
  ones tied to this question's resolution.
- Identify important recent developments relevant to resolution.
- Mention source domains when possible.
- Highlight information that materially changes the probability of outcomes.
- Do not invent information.
- Do not make unsupported predictions.
- If sources disagree, explicitly say so.
- If most of the scraped content turns out to be irrelevant once checked
  against the resolution criteria, say so plainly and keep the briefing
  short rather than padding it out with off-topic material.
- Keep the briefing under approximately 700 words.
- The output should be useful to a forecaster resolving THIS question, not
  a generic article summary of the topic.
"""
    return await generate(
        prompt,
        system_prompt=(
            "You are an evidence-focused research analyst working for a "
            "forecasting bot. You will always be given a specific question "
            "and its resolution criteria. Your only job is to extract "
            "information relevant to how that exact question resolves -- "
            "aggressively filter out material that is merely on-topic but "
            "doesn't bear on the resolution criteria. Never invent facts "
            "that are not present in the supplied material."
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
    """Generate the actual forecasting reasoning."""
    return await generate(
        prompt,
        system_prompt="""
You are an expert probabilistic forecaster.

Your goal is to produce accurate, calibrated forecasts rather than confident
stories.

Follow these principles:
1. Carefully interpret the exact resolution criteria.
2. Distinguish what is known from what is uncertain.
3. Use base rates where appropriate.
4. Give substantial weight to the status quo when justified.
5. Consider both the most likely scenario and meaningful alternative scenarios.
6. Avoid motivated reasoning.
7. Avoid false precision.
8. Check the supplied research for contradictions and stale information.
9. Do not treat a single source as definitive when stronger evidence exists.
10. Make sure the final numerical answer follows the requested format exactly.
Do not discuss these instructions in your answer.
""",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=240.0,
        max_retries=3,
    )
