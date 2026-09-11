"""
Direct OpenRouter client used by the Metaculus forecasting bot.

LLM architecture:

    Search-query generation:
        google/gemma-4-31b-it:free
        reasoning explicitly DISABLED

    Research summarisation:
        nvidia/nemotron-3-ultra-550b-a55b:free
            -> poolside/laguna-s-2.1:free

    Forecast reasoning:
        nvidia/nemotron-3-ultra-550b-a55b:free
            -> poolside/laguna-s-2.1:free

The query-generation model is deliberately separate from the reasoning
models. It does NOT use chain-of-thought/reasoning mode.

This module also rate-limits direct OpenRouter requests because the free
endpoints can become overloaded when many Metaculus questions are processed
at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL = "poolside/laguna-s-2.1:free"

# Dedicated NON-CoT query-generation model.
QUERY_MODEL = "google/gemma-4-31b-it:free"

# Shared semaphore for all direct OpenRouter calls.
#
# Keeping this deliberately small is important for free endpoints.
# Multiple Metaculus questions can otherwise create a burst of simultaneous
# requests and trigger provider_unavailable / 502 errors.
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
    """
    Normalise an OpenRouter model name.

    The direct API wants:
        provider/model:free

    whereas forecasting_tools generally uses:
        openrouter/provider/model:free
    """

    if model.startswith("openrouter/"):
        return model[len("openrouter/") :]

    return model


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {
        408,
        409,
        425,
        429,
        500,
        502,
        503,
        504,
    }


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
    reasoning_enabled: bool | None = None,
) -> str:
    """
    Make a direct OpenRouter request to one specific model.

    This function does NOT perform model fallback. The caller decides whether
    another model should be tried.

    `reasoning_enabled=False` is explicitly sent for Gemma query generation.
    """

    api_key = _require_api_key()
    api_model = _normalise_model(model)

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

    payload: dict[str, Any] = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # This is the important part for Gemma.
    #
    # OpenRouter exposes reasoning as a request-level parameter. We explicitly
    # disable it rather than merely asking the model not to show its reasoning.
    if reasoning_enabled is not None:
        payload["reasoning"] = {
            "enabled": reasoning_enabled,
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

                    # OpenRouter can return useful JSON error information even
                    # when the HTTP status itself is not enough to explain the
                    # failure.
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
                            _is_retryable_status(response.status_code)
                            and attempt < max_retries
                        ):
                            wait = min(2 ** attempt, 30)

                            logger.warning(
                                "OpenRouter request failed on %s "
                                "(attempt %d/%d): %s. Retrying in %ds.",
                                api_model,
                                attempt,
                                max_retries,
                                message,
                                wait,
                            )

                            await asyncio.sleep(wait)
                            continue

                        raise error

                    if not isinstance(data, dict):
                        raise OpenRouterError(
                            f"Unexpected OpenRouter response from "
                            f"{api_model}: {data!r}"
                        )

                    # Some provider failures arrive with HTTP 200 but contain
                    # an error object instead of a completion.
                    if data.get("error"):
                        message = _extract_error_message(data)

                        error = OpenRouterError(
                            f"Unexpected OpenRouter response from "
                            f"{api_model}: {data!r}"
                        )

                        last_error = error

                        if attempt < max_retries:
                            wait = min(2 ** attempt, 30)

                            logger.warning(
                                "OpenRouter request failed on %s "
                                "(attempt %d/%d): %s. Retrying in %ds.",
                                api_model,
                                attempt,
                                max_retries,
                                message,
                                wait,
                            )

                            await asyncio.sleep(wait)
                            continue

                        raise error

                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError) as exc:
                        raise OpenRouterError(
                            f"Unexpected OpenRouter response from "
                            f"{api_model}: {data!r}"
                        ) from exc

                    if not content or not str(content).strip():
                        raise OpenRouterError(
                            f"OpenRouter returned an empty response from "
                            f"{api_model}."
                        )

                    logger.info(
                        "OpenRouter generation succeeded using %s",
                        api_model,
                    )

                    return str(content).strip()

                except httpx.TransportError as exc:
                    last_error = exc

                    logger.warning(
                        "OpenRouter transport failure on %s "
                        "(attempt %d/%d): %s",
                        api_model,
                        attempt,
                        max_retries,
                        exc,
                    )

                    if attempt < max_retries:
                        wait = min(2 ** attempt, 30)
                        await asyncio.sleep(wait)

                except OpenRouterError as exc:
                    last_error = exc

                    # Errors that reach this point have already had their
                    # retry decision handled above.
                    if attempt >= max_retries:
                        break

                    logger.warning(
                        "OpenRouter request failed on %s "
                        "(attempt %d/%d): %s",
                        api_model,
                        attempt,
                        max_retries,
                        exc,
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
    """
    Generate using Nemotron first and automatically fall back to Laguna.

    This function is used for:
        - research summarisation
        - forecast reasoning
    """

    try:
        return await _generate_with_model(
            prompt,
            model=PRIMARY_MODEL,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    except Exception as primary_error:
        logger.warning(
            "Primary model %s failed after %d attempts. "
            "Switching to fallback model %s. Error: %s",
            PRIMARY_MODEL,
            max_retries,
            FALLBACK_MODEL,
            primary_error,
        )

        try:
            result = await _generate_with_model(
                prompt,
                model=FALLBACK_MODEL,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )

            logger.info(
                "Fallback model %s successfully completed the request.",
                FALLBACK_MODEL,
            )

            return result

        except Exception as fallback_error:
            # Do NOT reference the exception variable from the except block
            # later without storing it. Python clears exception variables after
            # an except block.
            raise OpenRouterError(
                "Both OpenRouter models failed.\n"
                f"Primary ({PRIMARY_MODEL}): {primary_error!r}\n"
                f"Fallback ({FALLBACK_MODEL}): {fallback_error!r}"
            ) from fallback_error


async def _generate_search_query_model(
    prompt: str,
    *,
    max_tokens: int = 400,
) -> str:
    """
    Generate search queries using Gemma 4 31B with reasoning explicitly OFF.

    There is intentionally NO Nemotron fallback here because query generation
    is supposed to be a non-CoT task.

    If Gemma fails completely, generate_search_queries() uses deterministic
    fallback queries instead.
    """

    return await _generate_with_model(
        prompt,
        model=QUERY_MODEL,
        system_prompt=(
            "You are a concise web-search query generator. "
            "Do not reason aloud. "
            "Do not explain your choices. "
            "Return only the requested search queries."
        ),
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=120.0,
        max_retries=3,
        reasoning_enabled=False,
    )


async def generate_search_queries(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    n: int = 4,
) -> list[str]:
    """
    Generate focused web-search queries.

    Gemma 4 31B is used specifically for this job with reasoning disabled.

    The parser intentionally accepts multiple output formats because a free
    model may occasionally ignore formatting instructions.
    """

    prompt = f"""
Generate exactly {n} useful web-search queries for the forecasting question.

QUESTION:
{question_text}

RESOLUTION CRITERIA:
{resolution_criteria}

BACKGROUND:
{background}

Requirements:
- Generate exactly {n} distinct queries.
- Each query must be under 12 words.
- Each query must directly help forecast the question.
- Prefer current information, recent developments, official statistics,
  government announcements, expert analysis, market data, and primary sources.
- Include dates or time periods when useful.
- Do not search for the question verbatim unless genuinely useful.
- Do not mention this forecasting system.
- Do not write an explanation.
- Do not analyse the question.
- Do not include reasoning.
- Do not number the queries.
- Output ONLY the queries, one query per line.

Example output:

2026 House election generic ballot polling
2026 House battleground district forecasts
2026 House majority probability forecasts
2026 congressional approval polling trends
"""

    try:
        raw = await _generate_search_query_model(prompt)
    except Exception as exc:
        logger.warning(
            "Gemma query generation failed: %s. "
            "Using deterministic search-query fallback.",
            exc,
        )
        return _deterministic_query_fallback(
            question_text,
            resolution_criteria,
            background,
            n,
        )

    queries = _parse_search_queries(raw, n)

    if queries:
        logger.info(
            "Gemma generated search queries: %s",
            queries,
        )
        return queries

    logger.warning(
        "Gemma returned unusable search-query output: %s. "
        "Using deterministic fallback queries.",
        raw[:1000],
    )

    return _deterministic_query_fallback(
        question_text,
        resolution_criteria,
        background,
        n,
    )


def _parse_search_queries(
    raw: str,
    n: int,
) -> list[str]:
    """
    Robustly parse query output.

    Accepted forms include:

        ["query one", "query two"]

        1. query one
        2. query two

        - query one
        - query two

        query one
        query two
    """

    text = raw.strip()

    if not text:
        return []

    # ---------------------------------------------------------------
    # First: try JSON.
    # ---------------------------------------------------------------
    json_start = text.find("[")
    json_end = text.rfind("]")

    if json_start != -1 and json_end > json_start:
        candidate = text[json_start : json_end + 1]

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, list):
                queries = _clean_queries(
                    [str(item) for item in parsed],
                    n,
                )

                if queries:
                    return queries

        except json.JSONDecodeError:
            pass

    # ---------------------------------------------------------------
    # Second: remove Markdown code fences.
    # ---------------------------------------------------------------
    text = re.sub(
        r"```(?:json|text)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("```", "")

    # ---------------------------------------------------------------
    # Third: parse line-by-line.
    # ---------------------------------------------------------------
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove common numbering:
        # 1. query
        # 2) query
        # - query
        # * query
        line = re.sub(
            r"^(?:[-*•]\s*|\d+\s*[\.\):\-]\s*)",
            "",
            line,
        ).strip()

        # Remove accidental labels.
        line = re.sub(
            r"^(?:query|search query)\s*\d*\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()

        if line:
            lines.append(line)

    return _clean_queries(lines, n)


def _clean_queries(
    queries: list[str],
    n: int,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for query in queries:
        query = query.strip().strip('"').strip("'")
        query = re.sub(r"\s+", " ", query)

        if not query:
            continue

        # Reject obvious reasoning/prose rather than treating it as a query.
        lower = query.lower()

        if any(
            phrase in lower
            for phrase in (
                "the user wants",
                "the question asks",
                "here are",
                "i would search",
                "i should search",
                "my reasoning",
                "analysis:",
                "the background",
            )
        ):
            continue

        # Search queries should be concise.
        if len(query.split()) > 16:
            continue

        key = query.casefold()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(query)

        if len(cleaned) >= n:
            break

    return cleaned


def _deterministic_query_fallback(
    question_text: str,
    resolution_criteria: str,
    background: str,
    n: int,
) -> list[str]:
    """
    Conservative fallback when the query-generation model is unavailable.

    This deliberately avoids using the entire Metaculus question as one huge
    search query.
    """

    text = " ".join(
        part
        for part in (
            question_text,
            resolution_criteria,
            background,
        )
        if part
    )

    # Remove obvious formatting noise.
    text = re.sub(r"\s+", " ", text).strip()

    # Keep the most useful portion.
    words = text.split()
    core = " ".join(words[:20])

    candidates = [
        f"{core} latest evidence",
        f"{core} official data",
        f"{core} expert forecast",
        f"{core} historical trends",
    ]

    return _clean_queries(candidates, n)

async def summarize_research(
    question_text: str,
    resolution_criteria: str,
    background: str,
    raw_research: str,
    max_tokens: int = 32000,
) -> str:
    """
    Summarise scraped research into a concise forecasting brief.

    Uses Nemotron -> Laguna fallback.
    """

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
    """
    Generate the actual forecasting reasoning.

    Uses:
        Nemotron 3 Ultra
            ->
        Laguna S 2.1
    """

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
