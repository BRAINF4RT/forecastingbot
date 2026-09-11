"""
OpenRouter client for helper tasks.

The main forecasting brain is VibeThinker-3B via Hugging Face.
OpenRouter is used for:

1. Search-query generation
2. Research summarization

Structured-output parsing of the final forecast is handled separately
in bot.py via forecasting_tools.structure_output + GeneralLlm.

All OpenRouter models used by this bot are intended to be FREE-tier models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default OpenRouter model for research summarisation.
DEFAULT_FREE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Default OpenRouter model for search-query generation.
DEFAULT_QUERY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


class OpenRouterError(RuntimeError):
    pass


def _require_free_model(model: str) -> str:
    """
    Make sure the configured model uses OpenRouter's free-tier suffix.

    This bot is intentionally configured to use only free OpenRouter models.
    """
    if not model.endswith(":free"):
        raise OpenRouterError(
            f"OpenRouter model '{model}' is not configured as a free-tier "
            "model. Expected a model ID ending in ':free'."
        )
    return model


async def generate(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    timeout: float = 90.0,
    max_retries: int = 3,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")

    model = _require_free_model(
        model or os.getenv("OPENROUTER_HELPER_MODEL", DEFAULT_FREE_MODEL)
    )

    messages = []

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
        "HTTP-Referer": "https://github.com/Metaculus/metac-bot-template",
        "X-Title": "VibeThinker Metaculus Bot",
    }

    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                )

                if resp.status_code == 429:
                    wait = min(2**attempt, 30)

                    logger.warning(
                        "OpenRouter rate limited (attempt %d/%d), retrying in %ds",
                        attempt,
                        max_retries,
                        wait,
                    )

                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()

                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                if not content or not content.strip():
                    raise OpenRouterError(
                        "Empty response from OpenRouter model"
                    )

                return content

            except (
                httpx.HTTPStatusError,
                httpx.TransportError,
                OpenRouterError,
            ) as exc:
                last_error = exc

                logger.warning(
                    "OpenRouter call failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    exc,
                )

                if attempt < max_retries:
                    await asyncio.sleep(min(2**attempt, 30))

    raise OpenRouterError(
        f"OpenRouter call failed after {max_retries} attempts: {last_error}"
    )


async def generate_search_queries(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    n: int = 4,
) -> list[str]:
    """
    Turn a forecasting question into a short list of web-search queries.

    This is the ONLY place search queries are generated.
    The main forecasting brain never sees this prompt.
    """

    prompt = f"""You are a research assistant helping a forecaster find relevant information.

Question: {question_text}

Resolution criteria: {resolution_criteria}

Background: {background}

Generate {n} short, distinct, high-quality web search queries
(each under 12 words) that would help find up-to-date, relevant
information for this question.

Favor queries that would surface:

- recent news
- official statements
- data releases
- expert analysis
- relevant statistics

Avoid redundant queries.

Respond with ONLY a JSON array of strings, nothing else.

Example:
["query one", "query two", "query three"]
"""

    query_model = os.getenv(
        "OPENROUTER_QUERY_MODEL",
        DEFAULT_QUERY_MODEL,
    )

    raw = await generate(
        prompt,
        model=query_model,
        temperature=0.4,
        max_tokens=300,
    )

    queries = _parse_json_list(raw)

    if not queries:
        queries = [question_text[:120]]

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
            "Could not parse JSON query list from OpenRouter output: %s",
            raw[:200],
        )

    return []


async def summarize_research(
    question_text: str,
    raw_research: str,
    max_tokens: int = 1200,
) -> str:
    """
    Condense scraped research into a concise, cited brief before handing
    it to the main forecasting brain.
    """

    if not raw_research.strip():
        return ""

    prompt = f"""You are a research assistant.

Summarize the following scraped web content into a concise,
factual briefing for a forecaster trying to answer this question.

Question:
{question_text}

Scraped content:
{raw_research[:12000]}

Write a concise briefing under 500 words.

Include:

- concrete facts
- dates
- figures
- relevant trends
- important uncertainty
- source domains inline where possible

Do not speculate beyond what the content supports.

If the content is thin or irrelevant, say so plainly.
"""

    return await generate(
        prompt,
        temperature=0.2,
        max_tokens=max_tokens,
    )
