"""Web search and scraping helpers."""
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
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
MAX_CHARS_PER_SOURCE = 100000
SEARCH_BACKENDS = ("brave", "google", "bing", "duckduckgo", "yahoo", "wikipedia")
SEARCH_DELAY_SECONDS = 0.25
SCRAPE_TIMEOUT = 12
BLOCKED_HOSTS = {"metaculus.com"}


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
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return True
    return hostname in BLOCKED_HOSTS or hostname.endswith(".metaculus.com")


def _normalise_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "?" in url:
        base, query = url.split("?", 1)
        keep = []
        for item in query.split("&"):
            key = item.split("=", 1)[0].lower()
            if key.startswith("utm_") or key in {"fbclid", "gclid", "ref", "ref_src"}:
                continue
            keep.append(item)
        url = base + (("?" + "&".join(keep)) if keep else "")
    return url.rstrip("/")


def _valid_result(result: dict) -> bool:
    url = result.get("href") or result.get("url") or ""
    title = result.get("title") or ""
    body = result.get("body") or ""
    return bool(isinstance(url, str) and url.strip() and (title.strip() or body.strip()))


def ddgs_search(query: str, max_results: int = 5) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    results_out: list[dict] = []
    seen_urls: set[str] = set()
    for backend in SEARCH_BACKENDS:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query, region="us-en", safesearch="moderate",
                    max_results=max_results, backend=backend,
                ))
            for result in results:
                if not _valid_result(result):
                    continue
                url = _normalise_url(result.get("href") or result.get("url") or "")
                if is_blocked_url(url) or not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                result = dict(result)
                result["_backend"] = backend
                result["_normalised_url"] = url
                results_out.append(result)
                if len(results_out) >= max_results:
                    break
            if len(results_out) >= max_results:
                break
        except Exception as exc:
            logger.warning("[SEARCH] backend=%s failed for %r: %s", backend, query, exc)
        time.sleep(SEARCH_DELAY_SECONDS)
    return results_out[:max_results]


def scrape_with_trafilatura(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded, include_comments=False, include_tables=False
        )
        if not text:
            return None
        text = text.strip()
        return text if len(text) >= 100 else None
    except Exception as exc:
        logger.debug("[SCRAPE] trafilatura failed for %s: %s", url, exc)
        return None


def scrape_with_bs4(url: str) -> str | None:
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT, allow_redirects=True,
        )
        response.raise_for_status()
        if is_blocked_url(response.url):
            return None
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(" ", strip=True)) >= 40
        ]
        text = "\n".join(paragraphs).strip()
        return text if len(text) >= 100 else None
    except Exception as exc:
        logger.debug("[SCRAPE] BeautifulSoup failed for %s: %s", url, exc)
        return None


def scrape_with_ddgs_snippet(
    url: str,
    title: str = "",
    original_snippet: str = "",
) -> str | None:
    """
    FINAL scraper layer.

    DDGS can return indexed search-result snippets even when the live URL
    cannot be downloaded. Search the exact URL first, then the page title.
    """
    if is_blocked_url(url):
        return None

    target = _normalise_url(url)
    searches = [f'"{url}"']
    if title.strip():
        searches.append(f'"{title.strip()}"')
    for query in searches:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=8,
                    backend="auto",
                ))
            for result in results:
                result_url = _normalise_url(
                    result.get("href") or result.get("url") or ""
                )
                body = (result.get("body") or "").strip()
                if is_blocked_url(result_url):
                    continue
                if result_url == target and len(body) >= 40:
                    return body
        except Exception as exc:
            logger.debug("[SCRAPE] DDGS snippet fallback failed for %s: %s", url, exc)
    # The original DDGS result snippet is still a valid final source of text.
    return original_snippet.strip() if len(original_snippet.strip()) >= 40 else None


def scrape_url(url: str, title: str = "", snippet: str = "") -> tuple[str, str]:
    if is_blocked_url(url):
        return "", "blocked"

    text = scrape_with_trafilatura(url)
    if text:
        return text, "trafilatura"
    text = scrape_with_bs4(url)
    if text:
        return text, "bs4"
    text = scrape_with_ddgs_snippet(
        url, title=title, original_snippet=snippet
    )
    if text:
        return text, "ddgs_snippet"

    return "", "failed"


def gather_sources(
    queries: Iterable[str],
    results_per_query: int = 3,
) -> list[ScrapedSource]:
    sources: list[ScrapedSource] = []
    seen_urls: set[str] = set()
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    for query_index, query in enumerate(queries, start=1):
        logger.info(
            "[RESEARCH] Query %d/%d: %s",
            query_index, len(queries), query,
        )
        results = ddgs_search(query, max_results=results_per_query)

        for result in results:
            url = _normalise_url(result.get("href") or result.get("url") or "")
            if is_blocked_url(url) or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = (result.get("title") or url).strip()
            snippet = (result.get("body") or "").strip()
            backend = result.get("_backend") or "unknown"

            content, method = scrape_url(
                url, title=title, snippet=snippet
            )
            if not content:
                continue
            sources.append(ScrapedSource(
                query=query,
                title=title,
                url=url,
                snippet=snippet,
                content=content[:MAX_CHARS_PER_SOURCE],
                method=method,
                backend=backend,
            ))
            logger.info(
                "[SOURCE] accepted backend=%s method=%s url=%s",
                backend, method, url,
            )
    logger.info(
        "[RESEARCH] Source gathering complete: %d usable sources",
        len(sources),
    )
    return sources


def format_sources_as_markdown(sources: list[ScrapedSource]) -> str:
    if not sources:
        return "NO_RESEARCH_AVAILABLE"
    return "\n---\n".join(
        f"### {source.title or source.url}\n"
        f"Source: {source.url}\n"
        f"(query: {source.query!r}; backend: {source.backend}; extraction: {source.method})\n\n"
        f"{source.content}\n"
        for source in sources
    )


def web_search(
    query: str,
    max_results: int = 4,
    scrape: bool = True,
) -> str:
    """Run one blocking web search and return its scraped source text."""
    query = query.strip()
    if not query:
        return ""

    if scrape:
        sources = gather_sources(
            [query],
            results_per_query=max_results,
        )
        return format_sources_as_markdown(sources)

    results = ddgs_search(query, max_results=max_results)
    sources = [
        ScrapedSource(
            query=query,
            title=(result.get("title") or result.get("href") or "").strip(),
            url=_normalise_url(result.get("href") or result.get("url") or ""),
            snippet=(result.get("body") or "").strip(),
            content=(result.get("body") or "").strip(),
            method="ddgs_snippet",
            backend=result.get("_backend") or "unknown",
        )
        for result in results
        if not is_blocked_url(
            _normalise_url(result.get("href") or result.get("url") or "")
        )
    ]
    return format_sources_as_markdown(sources)
