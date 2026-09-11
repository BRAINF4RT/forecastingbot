"""
Robust web search + scraping layer.

Search strategy:
    1. Try multiple DDGS backends independently.
    2. A failed/rate-limited backend never kills the entire search.
    3. Deduplicate URLs across backends.
    4. Block Metaculus URLs entirely.
    5. Try trafilatura for page extraction.
    6. Fall back to requests + BeautifulSoup.
    7. Fall back to the search-result snippet.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)

MAX_CHARS_PER_SOURCE = 5000

# Keep this deliberately small.
# The important thing is diversity, not hammering every search engine.
SEARCH_BACKENDS = (
    "brave",
    "google",
    "bing",
    "duckduckgo",
    "yahoo",
    "wikipedia",
)

SEARCH_DELAY_SECONDS = 0.25
SCRAPE_TIMEOUT = 12

# Metaculus must never be accessed by the research scraper.
# This blocks the main domain and all subdomains.
BLOCKED_HOSTS = {
    "metaculus.com",
}


@dataclass
class ScrapedSource:
    query: str
    title: str
    url: str
    snippet: str
    content: str = ""
    method: str = ""
    backend: str = ""


def is_blocked_url(url: str) -> bool:
    """Return True if the URL belongs to a blocked host."""
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return True

    return hostname in BLOCKED_HOSTS or hostname.endswith(".metaculus.com")


def _normalise_url(url: str) -> str:
    """Normalise URLs enough for useful deduplication."""
    url = url.strip()

    if not url:
        return ""

    # Remove common tracking parameters.
    if "?" in url:
        base, query = url.split("?", 1)

        keep = []
        for item in query.split("&"):
            key = item.split("=", 1)[0].lower()

            if key.startswith("utm_"):
                continue
            if key in {
                "fbclid",
                "gclid",
                "ref",
                "ref_src",
            }:
                continue

            keep.append(item)

        url = base + (("?" + "&".join(keep)) if keep else "")

    return url.rstrip("/")


def _valid_result(result: dict) -> bool:
    url = result.get("href") or result.get("url") or ""
    title = result.get("title") or ""
    body = result.get("body") or ""
    return bool(
        isinstance(url, str)
        and url.strip()
        and (title.strip() or body.strip())
    )


def ddgs_search(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search multiple DDGS backends independently.

    This is intentionally NOT one DDGS().text(..., backend="auto") call.

    A single backend being unavailable must not turn the whole research
    operation into zero results.
    """

    query = query.strip()

    if not query:
        return []

    all_results: list[dict] = []
    seen_urls: set[str] = set()

    logger.info(
        "[SEARCH] Searching %r across %d backends",
        query,
        len(SEARCH_BACKENDS),
    )

    for backend in SEARCH_BACKENDS:
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        max_results=max_results,
                        backend=backend,
                    )
                )

            valid = 0

            for result in results:
                if not _valid_result(result):
                    continue

                url = _normalise_url(
                    result.get("href")
                    or result.get("url")
                    or ""
                )

                # Never allow Metaculus results into the research pipeline.
                if is_blocked_url(url):
                    logger.info(
                        "[SEARCH] Blocked Metaculus result: %s",
                        url,
                    )
                    continue

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                result = dict(result)
                result["_backend"] = backend
                result["_normalised_url"] = url

                all_results.append(result)
                valid += 1

            logger.info(
                "[SEARCH] backend=%s returned=%d usable=%d",
                backend,
                len(results),
                valid,
            )

            if len(all_results) >= max_results:
                break

        except Exception as exc:
            logger.warning(
                "[SEARCH] backend=%s failed for %r: %s",
                backend,
                query,
                exc,
            )

        time.sleep(SEARCH_DELAY_SECONDS)

    logger.info(
        "[SEARCH] query=%r produced %d unique results",
        query,
        len(all_results),
    )

    return all_results[:max_results]


def scrape_with_trafilatura(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        )

        if not text:
            return None

        text = text.strip()

        return text if len(text) >= 100 else None

    except Exception as exc:
        logger.debug(
            "[SCRAPE] trafilatura failed for %s: %s",
            url,
            exc,
        )
        return None


def scrape_with_bs4(url: str) -> str | None:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml" not in content_type
        ):
            return None

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "noscript",
                "svg",
                "form",
            ]
        ):
            tag.decompose()

        paragraphs = []

        for paragraph in soup.find_all("p"):
            text = paragraph.get_text(
                " ",
                strip=True,
            )

            if len(text) >= 40:
                paragraphs.append(text)

        text = "\n".join(paragraphs).strip()

        return text if len(text) >= 100 else None

    except Exception as exc:
        logger.debug(
            "[SCRAPE] BeautifulSoup failed for %s: %s",
            url,
            exc,
        )
        return None


def scrape_url(url: str) -> tuple[str, str]:
    """Try multiple extraction methods."""

    # Safety boundary: Metaculus must never be fetched.
    if is_blocked_url(url):
        logger.info(
            "[SCRAPE] Blocked Metaculus URL: %s",
            url,
        )
        return "", "blocked"

    text = scrape_with_trafilatura(url)

    if text:
        return text, "trafilatura"

    text = scrape_with_bs4(url)

    if text:
        return text, "bs4"

    return "", "failed"


def gather_sources(
    queries: Iterable[str],
    results_per_query: int = 3,
) -> list[ScrapedSource]:
    """
    Search all supplied queries and return usable sources.

    Search snippets count as usable evidence. This is important because
    many legitimate sites block automated page fetching while still
    appearing in search results.
    """

    sources: list[ScrapedSource] = []
    seen_urls: set[str] = set()

    queries = [
        q.strip()
        for q in queries
        if isinstance(q, str) and q.strip()
    ]

    logger.info(
        "[RESEARCH] Starting source gathering: %d queries",
        len(queries),
    )

    for query_index, query in enumerate(queries, start=1):
        logger.info(
            "[RESEARCH] Query %d/%d: %s",
            query_index,
            len(queries),
            query,
        )

        results = ddgs_search(
            query,
            max_results=results_per_query,
        )

        if not results:
            logger.warning(
                "[RESEARCH] Query produced no search results: %s",
                query,
            )
            continue

        for result in results:
            url = _normalise_url(
                result.get("href")
                or result.get("url")
                or ""
            )

            # Second safety boundary: even if a blocked URL somehow
            # reaches gather_sources(), it is discarded here.
            if is_blocked_url(url):
                logger.info(
                    "[RESEARCH] Skipping blocked Metaculus result: %s",
                    url,
                )
                continue

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            title = (
                result.get("title")
                or url
            )

            snippet = (
                result.get("body")
                or ""
            ).strip()

            backend = (
                result.get("_backend")
                or "unknown"
            )

            content, method = scrape_url(url)

            # Critical fallback:
            # if the actual page blocks us, the search engine's snippet
            # is still usable evidence.
            if not content and len(snippet) >= 40:
                content = snippet
                method = "search_snippet"

            if not content:
                logger.debug(
                    "[RESEARCH] Discarding unusable result: %s",
                    url,
                )
                continue

            sources.append(
                ScrapedSource(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content[:MAX_CHARS_PER_SOURCE],
                    method=method,
                    backend=backend,
                )
            )

            logger.info(
                "[SOURCE] accepted backend=%s method=%s url=%s",
                backend,
                method,
                url,
            )

    logger.info(
        "[RESEARCH] Source gathering complete: %d usable sources",
        len(sources),
    )

    return sources


def format_sources_as_markdown(
    sources: list[ScrapedSource],
) -> str:
    if not sources:
        return "NO_RESEARCH_AVAILABLE"

    blocks = []

    for source in sources:
        blocks.append(
            f"### {source.title or source.url}\n"
            f"Source: {source.url}\n"
            f"(query: {source.query!r}; "
            f"backend: {source.backend}; "
            f"extraction: {source.method})\n\n"
            f"{source.content}\n"
        )

    return "\n---\n".join(blocks)
