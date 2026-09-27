"""Research orchestration for the forecasting bot."""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable

from clients import openrouter_helper
from research.scraper import web_search

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "about", "after", "against", "all", "an", "and", "are", "as",
    "at", "be", "before", "by", "can", "could", "does", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "its", "may", "might",
    "more", "most", "of", "on", "or", "that", "the", "their", "there",
    "this", "to", "under", "until", "what", "when", "where", "which", "who",
    "will", "with", "would", "year", "years", "than", "then", "whether",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = _clean(value)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _signal_terms(text: str, limit: int = 18) -> list[str]:
    """Extract useful entity/topic terms without requiring an LLM."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'./%-]*", text or "")
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lower = token.casefold()
        if lower in _STOPWORDS or len(lower) < 3:
            continue
        # Preserve years, percentages, acronyms and proper-looking tokens.
        if not (len(lower) >= 4 or any(ch.isdigit() for ch in token)):
            continue
        if lower not in seen:
            seen.add(lower)
            terms.append(token.strip(".,;:!?()[]{}"))
        if len(terms) >= limit:
            break
    return terms


def build_search_queries(
    question_text: str,
    resolution_criteria: str = "",
    background: str = "",
    max_queries: int = 5,
    retry: bool = False,
) -> list[str]:
    """Build deterministic, targeted search queries.

    The exact question is always the first query. Other queries emphasize
    entities/signals and resolution-specific language instead of simply using
    the first few words of the question.
    """
    question = _clean(question_text)
    question_terms = _signal_terms(question, 18)
    criteria_terms = _signal_terms(resolution_criteria, 10)
    background_terms = _signal_terms(background, 8)

    queries: list[str] = [question]
    signal = " ".join(question_terms[:12])
    criteria_signal = " ".join(criteria_terms[:6])
    background_signal = " ".join(background_terms[:5])

    if signal:
        queries.append(f"{signal} latest news")
        queries.append(f"{signal} official data results")
    if criteria_signal:
        queries.append(f"{signal} {criteria_signal} evidence")
    if background_signal:
        queries.append(f"{signal} {background_signal} recent developments")

    if retry:
        if signal:
            queries.extend(
                [
                    f"{signal} current status September 2026",
                    f"{signal} announcement update outcome",
                    f"{signal} statistics report filing",
                ]
            )
        if criteria_signal:
            queries.append(f"{criteria_signal} official source latest")

    return _unique(queries)[:max_queries]


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
        result
        for result in results
        if result.strip() and result.strip() != "NO_RESEARCH_AVAILABLE"
    ]
    return "\n\n=== SEARCH QUERY ===\n\n".join(usable)


async def run_research_pipeline(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    num_queries: int = 5,
    results_per_query: int = 4,
    summarize: bool = True,
) -> str:
    """Run parallel web research, retrying with alternate deterministic queries."""
    question = _clean(question_text)
    if not question:
        return "RESEARCH STATUS: NO_RESEARCH_AVAILABLE"

    first_queries = build_search_queries(
        question,
        resolution_criteria,
        background,
        max_queries=num_queries,
    )
    logger.info("[RESEARCH] Attempt 1 queries: %s", first_queries)
    raw_research = await _parallel_search(
        first_queries,
        results_per_query=results_per_query,
        relevance_text=question,
    )

    if not raw_research:
        retry_queries = build_search_queries(
            question,
            resolution_criteria,
            background,
            max_queries=num_queries,
            retry=True,
        )
        attempted = {q.casefold() for q in first_queries}
        retry_queries = [q for q in retry_queries if q.casefold() not in attempted]
        if retry_queries:
            logger.info("[RESEARCH] Attempt 2 queries: %s", retry_queries)
            raw_research = await _parallel_search(
                retry_queries,
                results_per_query=results_per_query,
                relevance_text=question,
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
            question,
            _clean(resolution_criteria),
            _clean(background),
            raw_research,
        )
    except Exception as exc:
        logger.warning("[RESEARCH] Research summarisation failed: %s", exc)
        summary = ""

    if not summary.strip():
        summary = (
            "Research was retrieved, but summarisation failed. The raw research "
            "is supplied below for the forecaster."
        )

    return f"{summary}\n\nRAW RESEARCH:\n{raw_research[:30000]}"
