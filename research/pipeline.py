"""
End-to-end research pipeline used by bot.py's run_research.

Flow:
  1. clients.openrouter_helper generates search queries (never the main brain).
  2. research.scraper runs DDGS searches and scrapes each result:
     trafilatura -> BeautifulSoup -> DDGS snippet (last resort).
  3. clients.openrouter_helper condenses the raw scraped text into a brief.

The output of this pipeline is the *only* research context handed to
VibeThinker-3B (the main brain) when it writes its forecast.
"""

from __future__ import annotations

import logging

from clients import openrouter_helper
from research.scraper import format_sources_as_markdown, gather_sources

logger = logging.getLogger(__name__)


async def run_research_pipeline(
    question_text: str,
    resolution_criteria: str,
    background: str = "",
    num_queries: int = 4,
    results_per_query: int = 3,
    summarize: bool = True,
) -> str:
    try:
        queries = await openrouter_helper.generate_search_queries(
            question_text, resolution_criteria, background, n=num_queries
        )
    except Exception as exc:
        logger.warning("Query generation failed, falling back to raw question text: %s", exc)
        queries = [question_text[:120]]

    logger.info("Generated search queries: %s", queries)

    sources = gather_sources(queries, results_per_query=results_per_query)
    raw_research = format_sources_as_markdown(sources)

    if not summarize:
        return raw_research

    try:
        summary = await openrouter_helper.summarize_research(question_text, raw_research)
    except Exception as exc:
        logger.warning("Research summarization failed, using raw research instead: %s", exc)
        return raw_research

    if not summary:
        return raw_research

    sources_list = "\n".join(f"- {s.url}" for s in sources) or "- (no sources found)"
    return f"{summary}\n\n**Sources consulted:**\n{sources_list}"
