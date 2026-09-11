"""
OpenRouter client for the Metaculus forecasting bot.

LLM routing:

    Primary:
        nvidia/nemotron-3-ultra-550b-a55b:free

    Fallback:
        poolside/laguna-s-2.1:free

All LLM calls go through OpenRouter.

There are intentionally no Hugging Face, Featherless, OpenAI,
Anthropic, Perplexity, AskNews, or other provider calls here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL = "poolside/laguna-s-2.1:free"

MAX_RETRIES_PER_MODEL = 3

RETRYABLE_STATUS_CODES = {
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
}


class OpenRouterError(RuntimeError):
    """Raised when all OpenRouter model attempts fail."""


def get_models() -> tuple[str, str]:
    """
    Return the primary and fallback models.

    The primary model can be selected through OPENROUTER_MODEL, but it is
    intentionally restricted to the free Nemotron endpoint.
    """

    configured = os.getenv("OPENROUTER_MODEL", PRIMARY_MODEL)

    if configured != PRIMARY_MODEL:
        raise OpenRouterError(
            "OPENROUTER_MODEL must be exactly "
            f"'{PRIMARY_MODEL}'. Got '{configured}'."
        )

    return PRIMARY_MODEL, FALLBACK_MODEL


def _require_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")

    return api_key


def _retry_delay(attempt: int) -> int:
    """
    Exponential backoff capped at 60 seconds.
    """

    return min(2 ** attempt, 60)


def _is_retryable_error(status_code: int | None, message: str) -> bool:
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    lowered = message.lower()

    retry_phrases = (
        "temporarily overloaded",
        "provider_unavailable",
        "provider unavailable",
        "rate limit",
        "rate limited",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "upstream error",
        "overloaded",
    )

    return any(phrase in lowered for phrase in retry_phrases)


async def _request_model(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:

    payload = {
        "model": model,
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

    for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
        try:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
            )

            status_code = response.status_code

            try:
                data: Any = response.json()
            except Exception:
                data = response.text

            if status_code >= 400:
                message = str(data)

                if _is_retryable_error(status_code, message):
                    wait = _retry_delay(attempt)

                    logger.warning(
                        "OpenRouter %s failed "
                        "(attempt %d/%d, HTTP %s): %s. "
                        "Retrying in %ds.",
                        model,
                        attempt,
                        MAX_RETRIES_PER_MODEL,
                        status_code,
                        message[:500],
                        wait,
                    )

                    last_error = OpenRouterError(
                        f"HTTP {status_code}: {message}"
                    )

                    if attempt < MAX_RETRIES_PER_MODEL:
                        await asyncio.sleep(wait)
                        continue

                raise OpenRouterError(
                    f"OpenRouter returned HTTP {status_code}: {message}"
                )

            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise OpenRouterError(
                    f"Unexpected OpenRouter response from {model}: {data}"
                ) from exc

            if not content or not str(content).strip():
                raise OpenRouterError(
                    f"OpenRouter returned an empty response from {model}."
                )

            return str(content).strip()

        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc

            wait = _retry_delay(attempt)

            logger.warning(
                "OpenRouter transport failure on %s "
                "(attempt %d/%d): %s",
                model,
                attempt,
                MAX_RETRIES_PER_MODEL,
                exc,
            )

            if attempt < MAX_RETRIES_PER_MODEL:
                await asyncio.sleep(wait)

        except OpenRouterError as exc:
            last_error = exc

            logger.warning(
                "OpenRouter request failed on %s "
                "(attempt %d/%d): %s",
                model,
                attempt,
                MAX_RETRIES_PER_MODEL,
                exc,
            )

            if attempt < MAX_RETRIES_PER_MODEL:
                await asyncio.sleep(_retry_delay(attempt))

    assert last_error is not None

    raise OpenRouterError(
        f"{model} failed after {MAX_RETRIES_PER_MODEL} attempts: "
        f"{last_error}"
    )


async def generate(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: float = 180.0,
) -> str:
    """
    Generate text using Nemotron first and Laguna second.

    Routing:

        Nemotron x3
            ↓ failure
        Laguna x3
            ↓ failure
        raise OpenRouterError
    """

    api_key = _require_api_key()
    primary_model, fallback_model = get_models()

    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    async with httpx.AsyncClient(timeout=timeout) as client:

        try:
            result = await _request_model(
                client=client,
                api_key=api_key,
                model=primary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.info(
                "OpenRouter generation succeeded using primary model: %s",
                primary_model,
            )

            return result

        except Exception as primary_error:
            logger.warning(
                "Primary model %s failed completely. "
                "Switching to fallback model %s. Error: %s",
                primary_model,
                fallback_model,
                primary_error,
            )

        try:
            result = await _request_model(
                client=client,
                api_key=api_key,
                model=fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.warning(
                "OpenRouter fallback succeeded using %s.",
                fallback_model,
            )

            return result

        except Exception as fallback_error:
            raise OpenRouterError(
                "Both OpenRouter models failed. "
                f"Primary={primary_model}: unavailable. "
                f"Fallback={fallback_model}: {fallback_error}"
            ) from fallback_error


async def generate_search_queries(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    n: int = 4,
) -> list[str]:

    prompt = f"""
You are the research-query specialist for a professional forecasting system.

Generate exactly {n} useful web-search queries for the forecasting question.

QUESTION:
{question_text}

RESOLUTION CRITERIA:
{resolution_criteria}

BACKGROUND:
{background}

Requirements:
- Each query must be under 12 words.
- Every query must be directly relevant to the question.
- Prefer current information, recent developments, official statistics,
  government announcements, expert analysis and primary sources.
- Include dates or time periods when useful.
- Do not mention this forecasting system.
- Do not generate generic searches such as "latest news".
- Do not repeat the same idea.

Return ONLY a JSON array of strings.

Example:
["query one", "query two", "query three", "query four"]
"""

    raw = await generate(
        prompt,
        system_prompt=(
            "You generate precise search queries for forecasting research. "
            "Return valid JSON when requested."
        ),
        temperature=0.2,
        max_tokens=400,
    )

    queries = _parse_json_list(raw)

    if not queries:
        logger.warning(
            "No valid search queries returned. "
            "Falling back to the question text."
        )
        queries = [question_text[:200]]

    return queries[:n]


def _parse_json_list(raw: str) -> list[str]:

    text = raw.strip()

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except json.JSONDecodeError:
        logger.warning(
            "Could not parse model output as JSON: %s",
            raw[:500],
        )

    return []


async def summarize_research(
    question_text: str,
    raw_research: str,
    max_tokens: int = 1800,
) -> str:

    if not raw_research.strip():
        return ""

    prompt = f"""
You are the research-analysis specialist for a professional forecasting bot.

Forecasting question:
{question_text}

Information collected from web searches:

{raw_research[:20000]}

Create a concise factual research brief.

Requirements:
- Separate established facts from uncertainty.
- Preserve important dates, numbers, percentages and estimates.
- Identify important recent developments.
- Mention source domains when possible.
- Highlight information that materially affects outcome probabilities.
- Do not invent information.
- Do not make unsupported predictions.
- If sources disagree, explicitly say so.
- Ignore irrelevant material.
- Keep the briefing under approximately 700 words.
"""

    return await generate(
        prompt,
        system_prompt=(
            "You are an evidence-focused research analyst. "
            "Never invent facts that are not present in the supplied material."
        ),
        temperature=0.15,
        max_tokens=max_tokens,
    )


async def generate_forecast_reasoning(
    prompt: str,
    *,
    temperature: float = 0.15,
    max_tokens: int = 5000,
) -> str:

    return await generate(
        prompt,
        system_prompt=(
            "You are an expert probabilistic forecaster. "
            "Reason carefully about base rates, timelines, evidence, "
            "alternative scenarios and uncertainty. "
            "Follow the requested output format exactly."
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )
