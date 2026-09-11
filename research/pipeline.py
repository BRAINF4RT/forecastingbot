"""
End-to-end research pipeline.

Research flow:

    Gemma query generation
        ↓
    DDGS multi-backend search
        ↓
    scrape / snippet extraction
        ↓
    if empty:
        Gemma generates replacement queries
        ↓
    second search
        ↓
    if still empty:
        deterministic queries
        ↓
    final search
        ↓
    only then summarise with Nemotron → Laguna

The forecasting model is never allowed to turn an empty search result
into a fake "research brief".
"""

from __future__ import annotations

import logging
import re

from clients import openrouter_helper
from research.scraper import (
    ScrapedSource,
    format_sources_as_markdown,
    gather_sources,
)

logger = logging.getLogger(__name__)

MAX_RESEARCH_ATTEMPTS = 3


def _deterministic_queries(
    question_text: str,
    resolution_criteria: str,
    background: str,
) -> list[str]:
    """
    Deterministic search queries used when Gemma-generated searches
    produce no evidence.

    These deliberately use short fragments rather than the entire
    question verbatim.
    """

    question = re.sub(
        r"\s+",
        " ",
        question_text,
    ).strip()

    resolution = re.sub(
        r"\s+",
        " ",
        resolution_criteria or "",
    ).strip()

    background = re.sub(
        r"\s+",
        " ",
        background or "",
    ).strip()

    # Extract a useful question fragment.
    words = question.split()

    if len(words) > 12:
        question_fragment = " ".join(words[:12])
    else:
        question_fragment = question

    queries = [
        f'"{question_fragment}"',
        question_fragment,
    ]

    if resolution:
        resolution_words = resolution.split()

        if len(resolution_words) > 12:
            resolution_fragment = " ".join(
                resolution_words[:12]
            )
        else:
            resolution_fragment = resolution

        queries.append(
            f'"{question_fragment}" {resolution_fragment}'
        )

    if background:
        background_words = background.split()

        if len(background_words) > 10:
            background_fragment = " ".join(
                background_words[:10]
            )
        else:
            background_fragment = background

        queries.append(
            f'"{question_fragment}" {background_fragment}'
        )

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(queries))


async def _generate_queries(
    question_text: str,
    resolution_criteria: str,
    background: str,
    n: int,
) -> list[str]:
    queries = await openrouter_helper.generate_search_queries(
        question_text,
        resolution_criteria,
        background,
        n=n,
    )

    cleaned = []

    for query in queries:
        if not isinstance(query, str):
            continue

        query = query.strip()

        if not query:
            continue

        if query not in cleaned:
            cleaned.append(query)

    return cleaned[:n]


async def run_research_pipeline(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    num_queries: int = 4,
    results_per_query: int = 3,
    summarize: bool = True,
) -> str:
    logger.info(
        "[RESEARCH] Starting research for question: %s",
        question_text,
    )

    all_sources: list[ScrapedSource] = []
    attempted_queries: list[str] = []

    # ------------------------------------------------------------
    # Attempt 1: Gemma-generated queries
    # ------------------------------------------------------------

    try:
        queries = await _generate_queries(
            question_text,
            resolution_criteria,
            background,
            num_queries,
        )

    except Exception as exc:
        logger.warning(
            "[RESEARCH] Initial query generation failed: %s",
            exc,
        )
        queries = []

    if queries:
        logger.info(
            "[RESEARCH] Attempt 1 queries: %s",
            queries,
        )

        attempted_queries.extend(queries)

        all_sources = gather_sources(
            queries,
            results_per_query=results_per_query,
        )

    else:
        logger.warning(
            "[RESEARCH] Attempt 1 produced no usable queries",
        )

    # ------------------------------------------------------------
    # Attempt 2: ask Gemma for DIFFERENT queries
    # ------------------------------------------------------------

    if not all_sources:
        logger.warning(
            "[RESEARCH] Attempt 1 returned ZERO sources. "
            "Requesting replacement queries."
        )

        retry_background = (
            f"{background}\n\n"
            "Previous search attempt returned zero usable sources. "
            "Generate substantially different search queries. "
            "Prefer specific entities, organisations, dates, "
            "statistics, official sources, and exact phrases. "
            "Do not repeat previous queries."
        )

        try:
            retry_queries = await _generate_queries(
                question_text,
                resolution_criteria,
                retry_background,
                num_queries,
            )

        except Exception as exc:
            logger.warning(
                "[RESEARCH] Replacement query generation failed: %s",
                exc,
            )
            retry_queries = []

        retry_queries = [
            q
            for q in retry_queries
            if q not in attempted_queries
        ]

        if retry_queries:
            logger.info(
                "[RESEARCH] Attempt 2 queries: %s",
                retry_queries,
            )

            attempted_queries.extend(retry_queries)

            all_sources = gather_sources(
                retry_queries,
                results_per_query=results_per_query,
            )

    # ------------------------------------------------------------
    # Attempt 3: deterministic fallback
    # ------------------------------------------------------------

    if not all_sources:
        logger.warning(
            "[RESEARCH] Attempt 2 returned ZERO sources. "
            "Using deterministic search queries."
        )

        fallback_queries = _deterministic_queries(
            question_text,
            resolution_criteria,
            background,
        )

        fallback_queries = [
            q
            for q in fallback_queries
            if q not in attempted_queries
        ]

        logger.info(
            "[RESEARCH] Attempt 3 queries: %s",
            fallback_queries,
        )

        if fallback_queries:
            all_sources = gather_sources(
                fallback_queries,
                results_per_query=results_per_query,
            )

    # ------------------------------------------------------------
    # Hard failure: don't hallucinate research
    # ------------------------------------------------------------

    if not all_sources:
        logger.error(
            "[RESEARCH] NO_RESEARCH_AVAILABLE after %d attempts. "
            "No web evidence will be presented as research.",
            MAX_RESEARCH_ATTEMPTS,
        )

        return (
            "RESEARCH STATUS: NO_RESEARCH_AVAILABLE\n\n"
            "All web-search attempts returned zero usable sources. "
            "Do not treat this as evidence that no information exists. "
            "The forecaster may use its general knowledge and base rates, "
            "but must explicitly recognise that fresh web evidence "
            "was unavailable."
        )

    logger.info(
        "[RESEARCH] Successfully collected %d sources",
        len(all_sources),
    )

    raw_research = format_sources_as_markdown(
        all_sources
    )

    if not summarize:
        return raw_research

    # ------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------

    try:
        summary = await openrouter_helper.summarize_research(
            question_text,
            raw_research,
        )

    except Exception as exc:
        logger.warning(
            "[RESEARCH] Research summarisation failed: %s",
            exc,
        )

        summary = ""

    if not summary:
        summary = (
            "Research sources were successfully retrieved, "
            "but the research summariser failed. "
            "Use the raw sources below."
        )

    sources_list = "\n".join(
        f"- {source.url}"
        for source in all_sources
    )

    return (
        f"{summary}\n\n"
        f"**Sources consulted:**\n"
        f"{sources_list}"
    )
