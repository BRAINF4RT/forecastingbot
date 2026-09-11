"""
Search + scrape cascade:

  1. DDGS text search finds candidate URLs (+ snippets) for each query.
  2. trafilatura fetches and extracts the article body from each URL.
  3. If trafilatura fails/returns nothing, fall back to requests + BeautifulSoup.
  4. If that also fails, fall back to the DDGS search-result snippet itself
     (last resort -- no live page fetch involved, just what DDGS already gave us).

Queries fed into `gather_sources` must come from clients.openrouter_helper
(or a human), never from the main-brain model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

MAX_CHARS_PER_SOURCE = 3000


@dataclass
class ScrapedSource:
    query: str
    title: str
    url: str
    snippet: str
    content: str = ""
    method: str = ""


def ddgs_search(query: str, max_results: int = 5) -> list[dict]:
    """Run a DDGS text search. Returns [] on failure so the pipeline can move
    on to the next query instead of crashing the whole research step."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.warning("DDGS search failed for query '%s': %s", query, exc)
        return []


def scrape_with_trafilatura(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text.strip() if text else None
    except Exception as exc:
        logger.debug("trafilatura failed for %s: %s", url, exc)
        return None


def scrape_with_bs4(url: str) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(p for p in paragraphs if len(p) > 40)
        return text.strip() if text else None
    except Exception as exc:
        logger.debug("bs4 fallback failed for %s: %s", url, exc)
        return None


def scrape_url(url: str) -> tuple[str, str]:
    """Try trafilatura, then BeautifulSoup. Returns (content, method_used)."""
    text = scrape_with_trafilatura(url)
    if text:
        return text, "trafilatura"
    text = scrape_with_bs4(url)
    if text:
        return text, "bs4"
    return "", "failed"


def gather_sources(queries: list[str], results_per_query: int = 3) -> list[ScrapedSource]:
    sources: list[ScrapedSource] = []
    seen_urls: set[str] = set()

    for query in queries:
        results = ddgs_search(query, max_results=results_per_query)
        if not results:
            continue

        for result in results:
            url = result.get("href") or result.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = result.get("title", "")
            snippet = result.get("body", "")

            content, method = scrape_url(url)
            if not content:
                # Final fallback: the DDGS search-result snippet itself,
                # since direct fetch + BS4 parsing both failed (paywalls,
                # JS-rendered pages, blocked bots, etc).
                content = snippet
                method = "ddgs_snippet_fallback"

            sources.append(
                ScrapedSource(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content[:MAX_CHARS_PER_SOURCE],
                    method=method,
                )
            )
    return sources


def format_sources_as_markdown(sources: list[ScrapedSource]) -> str:
    if not sources:
        return "No search results were found or scraped for this question."

    blocks = []
    for src in sources:
        blocks.append(
            f"### {src.title or src.url}\n"
            f"Source: {src.url}\n"
            f'(query: "{src.query}", extraction: {src.method})\n\n'
            f"{src.content}\n"
        )
    return "\n---\n".join(blocks)
