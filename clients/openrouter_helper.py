"""
OpenRouter client used by the forecasting bot.

All LLM operations use the same free OpenRouter model:

    nvidia/nemotron-3-ultra-550b-a55b:free

The model is used for:
    - web-search query generation
    - research summarisation
    - forecast reasoning
    - structured-output parsing via forecasting_tools

No Hugging Face or VibeThinker dependency is used here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails."""


def get_model() -> str:
    """
    Return the configured OpenRouter model.

    The model must be a free-tier model. This check prevents an accidental
    configuration change from silently turning the bot into a paid bot.
    """
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if model != DEFAULT_MODEL:
        raise OpenRouterError(
            f"OPENROUTER_MODEL is set to '{model}'. "
            f"This bot is configured to use only the free model "
            f"'{DEFAULT_MODEL}'."
        )

    return DEFAULT_MODEL


def _require_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")

    return api_key


async def generate(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: float = 180.0,
    max_retries: int = 4,
) -> str:
    """
    Send a text-generation request to OpenRouter.

    Every request is forced to the configured free Nemotron model.
    """

    api_key = _require_api_key()
    model = get_model()

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

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 429:
                    wait = min(2**attempt, 60)

                    logger.warning(
                        "OpenRouter rate limited "
                        "(attempt %d/%d). Retrying in %ds.",
                        attempt,
                        max_retries,
                        wait,
                    )

                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()

                data = response.json()

                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise OpenRouterError(
                        f"Unexpected OpenRouter response: {data}"
                    ) from exc

                if not content or not content.strip():
                    raise OpenRouterError(
                        "OpenRouter returned an empty response."
                    )

                return content.strip()

            except (
                httpx.HTTPStatusError,
                httpx.TransportError,
                OpenRouterError,
            ) as exc:
                last_error = exc

                logger.warning(
                    "OpenRouter request failed "
                    "(attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    exc,
                )

                if attempt < max_retries:
                    await asyncio.sleep(min(2**attempt, 30))

    raise OpenRouterError(
        f"OpenRouter request failed after {max_retries} attempts: "
        f"{last_error}"
    )


async def generate_search_queries(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    n: int = 4,
) -> list[str]:
    """
    Generate focused web-search queries for a forecasting question.
    """

    prompt = f"""
You are the research-query specialist for a professional forecasting system.

Your job is to generate useful web-search queries for the forecasting
question below.

QUESTION:
{question_text}

RESOLUTION CRITERIA:
{resolution_criteria}

BACKGROUND:
{background}

Generate exactly {n} distinct search queries.

Requirements:

- Each query must be under 12 words.
- Queries must be directly relevant to the question.
- Prefer current information, recent developments, official statistics,
  government announcements, expert analysis, and primary sources.
- Include dates or time periods when they materially improve the search.
- Do not search for the question verbatim unless that is genuinely useful.
- Do not mention this forecasting system.
- Do not generate generic searches such as "latest news".
- Do not repeat the same idea using slightly different wording.

Return ONLY a JSON array of strings.

Example:
["query one", "query two", "query three", "query four"]
"""

    raw = await generate(
        prompt,
        system_prompt=(
            "You generate precise search queries for forecasting research. "
            "Output only valid JSON when requested."
        ),
        temperature=0.2,
        max_tokens=400,
    )

    queries = _parse_json_list(raw)

    if not queries:
        logger.warning(
            "Nemotron returned no valid search queries; "
            "falling back to the question text."
        )
        queries = [question_text[:200]]

    return queries[:n]


def _parse_json_list(raw: str) -> list[str]:
    """
    Extract a JSON list from model output.
    """

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
            "Could not parse Nemotron search-query output as JSON: %s",
            raw[:500],
        )

    return []


async def summarize_research(
    question_text: str,
    raw_research: str,
    max_tokens: int = 1800,
) -> str:
    """
    Summarise scraped research into a concise forecasting brief.
    """

    if not raw_research.strip():
        return ""

    prompt = f"""
You are the research-analysis specialist for a professional forecasting bot.

Forecasting question:
{question_text}

Below is information collected from web searches.

RESEARCH:
{raw_research[:20000]}

Create a concise factual research brief for another forecaster.

Requirements:

- Separate established facts from uncertainty.
- Preserve important dates, numbers, percentages and estimates.
- Identify important recent developments.
- Mention source domains when possible.
- Highlight information that materially changes the probability of outcomes.
- Do not invent information.
- Do not make unsupported predictions.
- If sources disagree, explicitly say so.
- Ignore irrelevant material.
- Keep the briefing under approximately 700 words.

The output should be useful to a forecaster, not a generic article summary.
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
    """
    Generate the actual forecasting reasoning.

    This is the main forecasting call. Nemotron 3 Ultra is now the main
    reasoning model instead of VibeThinker.
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
    )
