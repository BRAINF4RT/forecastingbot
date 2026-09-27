"""End-to-end research pipeline compatibility wrapper."""
from __future__ import annotations

import asyncio
import logging

from clients import openrouter_helper
from research.scraper import web_search

logger = logging.getLogger(__name__)


def build_search_queries(question_text: str, num_queries: int = 3) -> list[str]:
    """Build the same deterministic search queries used by bot.py."""
    words = question_text.rstrip("?").split()
    queries = [question_text]
    if len(words) > 6:
        queries.append(" ".join(words[:8]))
    queries.append(" ".join(words[:6]) + " latest news")
    deduped_queries = list(dict.fromkeys(queries))
    return deduped_queries[:num_queries]


async def _free_web_research(question_text: str, num_queries: int = 3) -> str:
    queries = build_search_queries(question_text, num_queries=num_queries)

    async def run_query(query: str) -> str:
        try:
            return await asyncio.to_thread(web_search, query, 4, True)
        except Exception as exc:
            logger.warning("[RESEARCH] Web search failed for %r: %s", query, exc)
            return ""

    results = await asyncio.gather(*(run_query(query) for query in queries))
    return "\n\n===\n\n".join(result for result in results if result)


async def run_research_pipeline(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    num_queries: int = 3,
    results_per_query: int = 4,
    summarize: bool = True,
) -> str:
    """Compatibility entry point using deterministic queries and parallel search."""
    if results_per_query != 4:
        logger.info(
            "[RESEARCH] results_per_query=%d requested; deterministic web_search path uses 4 sources per query.",
            results_per_query,
        )

    raw_research = await _free_web_research(
        question_text,
        num_queries=num_queries,
    )
    if not raw_research:
        return (
            "RESEARCH STATUS: NO_RESEARCH_AVAILABLE\n\n"
            "All web-search attempts returned zero usable sources. "
            "Do not treat this as evidence that no information exists."
        )

    if not summarize:
        return raw_research

    try:
        summary = await openrouter_helper.summarize_research(
            question_text,
            resolution_criteria,
            background,
            raw_research,
        )
    except TypeError:
        summary = await openrouter_helper.summarize_research(
            question_text,
            raw_research,
        )
    except Exception as exc:
        logger.warning("[RESEARCH] Research summarisation failed: %s", exc)
        summary = ""

    if not summary:
        summary = "Research was retrieved, but summarisation failed. Raw sources follow."

    return summary
