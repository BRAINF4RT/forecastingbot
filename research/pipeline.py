"""Research orchestration for the forecasting bot.

Search-query construction is deliberately deterministic. There is no LLM
query generation, expansion, retry-query generation, or query rewriting here.
"""
from __future__ import annotations

import asyncio
import logging

from clients import openrouter_helper
from research.scraper import web_search

logger = logging.getLogger(__name__)


def _dedupe_queries(values: list[str]) -> list[str]:
    """Deduplicate queries case-insensitively while preserving exact text/order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def build_search_queries(
    question_text: str,
    resolution_criteria: str = "",
    background: str = "",
    max_queries: int = 3,
    retry: bool = False,
) -> list[str]:
    """Build the fixed deterministic search-query set.

    Rules:
      1. Query 1 is ``question_text`` verbatim.
      2. If the question has more than six whitespace-delimited words,
         Query 2 is the first eight words, after removing a trailing ``?``
         only for tokenization.
      3. If the question has more than six words, Query 3 is the first six
         words followed by ``latest news``.

    ``resolution_criteria``, ``background`` and ``retry`` remain in the
    signature for compatibility with older callers, but are intentionally not
    used. There is no LLM or deterministic query expansion beyond the three
    rules above.
    """
    del resolution_criteria, background, retry

    if not isinstance(question_text, str) or not question_text:
        return []

    # IMPORTANT: query 1 is the exact question text. Do not strip, normalize,
    # or otherwise modify it.
    queries = [question_text]

    # Only remove a trailing question mark for tokenization. The original
    # question string above remains untouched.
    token_source = (
        question_text[:-1]
        if question_text.endswith("?")
        else question_text
    )
    words = token_source.split()

    if len(words) > 6:
        queries.append(" ".join(words[:8]))
        queries.append(" ".join(words[:6]) + " latest news")

    deduped_queries = _dedupe_queries(queries)
    return deduped_queries[: max(0, max_queries)]


async def _run_search(
    query: str,
    *,
    results_per_query: int,
    relevance_text: str,
) -> str:
    try:
        return await asyncio.to_thread(
            web_search,
            query,
            results_per_query,
            True,
            relevance_text,
        )
    except Exception as exc:
        logger.warning("[RESEARCH] Web search failed for %r: %s", query, exc)
        return ""


async def _parallel_search(
    queries: list[str],
    *,
    results_per_query: int,
    relevance_text: str,
) -> str:
    """Run every deduplicated query concurrently and combine nonempty results."""
    if not queries:
        return ""

    results = await asyncio.gather(
        *(
            _run_search(
                query,
                results_per_query=results_per_query,
                relevance_text=relevance_text,
            )
            for query in queries
        )
    )

    usable = [
        result.strip()
        for result in results
        if result and result.strip() and result.strip() != "NO_RESEARCH_AVAILABLE"
    ]
    logger.info(
        "[RESEARCH] Completed %d/%d searches.",
        len(usable),
        len(queries),
    )
    return "\n\n===\n\n".join(usable)


async def run_research_pipeline(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    fine_print: str = "",
    question_context: str = "",
    num_queries: int = 3,
    results_per_query: int = 4,
    summarize: bool = True,
) -> str:
    """Run the fixed three-query web-research pipeline exactly once."""
    if not isinstance(question_text, str) or not question_text:
        return "RESEARCH STATUS: NO_RESEARCH_AVAILABLE"

    queries = build_search_queries(
        question_text=question_text,
        resolution_criteria=resolution_criteria,
        background=background,
        max_queries=num_queries,
    )
    logger.info("[RESEARCH] Deterministic queries: %s", queries)

    raw_research = await _parallel_search(
        queries,
        results_per_query=results_per_query,
        relevance_text=question_text,
    )

    if not raw_research:
        return (
            "RESEARCH STATUS: NO_RESEARCH_AVAILABLE\n\n"
            "All web searches returned zero usable sources. "
            "Do not treat this as evidence that no information exists."
        )

    if not summarize:
        return raw_research

    try:
        summary = await openrouter_helper.summarize_research(
            question_text=question_text,
            resolution_criteria=resolution_criteria or "",
            background=background or "",
            fine_print=fine_print or "",
            question_context=question_context or "",
            raw_research=raw_research,
        )
    except Exception as exc:
        logger.warning("[RESEARCH] Research summarisation failed: %s", exc)
        summary = ""

    if not summary or not summary.strip():
        summary = (
            "Research was retrieved, but summarisation failed. "
            "The raw research is supplied below for the forecaster."
        )

    return f"{summary}\n\nRAW RESEARCH:\n{raw_research[:30000]}"
