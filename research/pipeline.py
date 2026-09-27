"""End-to-end research pipeline."""
from __future__ import annotations
import logging
import re
from clients import openrouter_helper
from research.scraper import ScrapedSource, format_sources_as_markdown, gather_sources

logger = logging.getLogger(__name__)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _unique(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for query in queries:
        query = _clean(query)
        if query and query.casefold() not in seen:
            seen.add(query.casefold())
            result.append(query)
    return result


def _deterministic_queries(question_text: str, resolution_criteria: str, background: str) -> list[str]:
    question = _clean(question_text)
    fragment = " ".join(question.split()[:16])
    return _unique([
        question,
        f"{fragment} latest evidence",
        f"{fragment} official data",
        f"{fragment} expert forecast",
        f"{fragment} historical trends",
        f"{fragment} {_clean(resolution_criteria)[:120]}" if resolution_criteria else "",
        f"{fragment} {_clean(background)[:120]}" if background else "",
    ])


async def _generate_queries(question_text: str, resolution_criteria: str, background: str, n: int) -> list[str]:
    queries = await openrouter_helper.generate_search_queries(
        question_text, resolution_criteria, background, n=n
    )
    return _unique(queries)[:n]


async def run_research_pipeline(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    num_queries: int = 4,
    results_per_query: int = 3,
    summarize: bool = True,
) -> str:
    logger.info("[RESEARCH] Starting research for question: %s", question_text)
    original_query = _clean(question_text)
    attempted: list[str] = []

    try:
        generated = await _generate_queries(
            question_text, resolution_criteria, background, num_queries
        )
    except Exception as exc:
        logger.warning("[RESEARCH] Query generation failed: %s", exc)
        generated = []

    # The original question is ALWAYS searched alongside generated queries.
    queries = _unique(generated + [original_query])
    attempted.extend(queries)
    logger.info("[RESEARCH] Attempt 1 queries: %s", queries)
    all_sources = gather_sources(queries, results_per_query=results_per_query)

    if not all_sources:
        retry_background = (
            f"{background}\n\nPrevious search returned zero usable sources. "
            "Generate substantially different, specific queries. Do not repeat "
            "previous queries."
        )
        try:
            retry_generated = await _generate_queries(
                question_text, resolution_criteria, retry_background, num_queries
            )
        except Exception as exc:
            logger.warning("[RESEARCH] Replacement query generation failed: %s", exc)
            retry_generated = []

        # Original question is also retained on retry, in addition to new queries.
        attempted_keys = {q.casefold() for q in attempted}
        retry_queries = [
            q for q in _unique(retry_generated + [original_query])
            if q.casefold() not in attempted_keys
        ]
        attempted.extend(retry_queries)
        if retry_queries:
            logger.info("[RESEARCH] Attempt 2 queries: %s", retry_queries)
            all_sources = gather_sources(
                retry_queries, results_per_query=results_per_query
            )

    if not all_sources:
        fallback_queries = [
            q for q in _deterministic_queries(
                question_text, resolution_criteria, background
            )
            if q.casefold() not in {x.casefold() for x in attempted}
        ]
        if fallback_queries:
            logger.info("[RESEARCH] Attempt 3 queries: %s", fallback_queries)
            all_sources = gather_sources(
                fallback_queries, results_per_query=results_per_query
            )

    if not all_sources:
        return (
            "RESEARCH STATUS: NO_RESEARCH_AVAILABLE\n\n"
            "All web-search attempts returned zero usable sources. "
            "Do not treat this as evidence that no information exists."
        )

    raw_research = format_sources_as_markdown(all_sources)
    if not summarize:
        return raw_research

    try:
        summary = await openrouter_helper.summarize_research(
            question_text, resolution_criteria, background, raw_research
        )
    except TypeError:
        summary = await openrouter_helper.summarize_research(
            question_text, raw_research
        )
    except Exception as exc:
        logger.warning("[RESEARCH] Research summarisation failed: %s", exc)
        summary = ""

    if not summary:
        summary = "Research was retrieved, but summarisation failed. Raw sources follow."

    sources_list = "\n".join(f"- {source.url}" for source in all_sources)
    return f"{summary}\n\n**Sources consulted:**\n{sources_list}"
