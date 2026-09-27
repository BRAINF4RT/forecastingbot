"""Web search, relevance filtering, and resilient page scraping helpers."""
from __future__ import annotations

import logging
import re
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
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/131.0 Safari/537.36"
)
MAX_CHARS_PER_SOURCE = 100000
SEARCH_BACKENDS = ("brave", "google", "bing", "duckduckgo", "yahoo", "wikipedia")
SEARCH_DELAY_SECONDS = 0.25
SCRAPE_TIMEOUT = 12
BLOCKED_HOSTS = {"metaculus.com"}

# Search engines can return stale, generic, or duplicated results. We collect
# several candidates per backend, then rank them against the *full forecasting
# question* before scraping.
SEARCH_CANDIDATE_MULTIPLIER = 3
MIN_METADATA_RELEVANCE_SCORE = 3.0
MIN_CONTENT_RELEVANCE_SCORE = 4.0

# Text that strongly suggests an HTTP/application error page rather than a
# usable source. We only apply body-text markers to relatively short pages so
# that a legitimate article discussing a 404 does not get rejected.
DEAD_TITLE_MARKERS = (
    "404",
    "page not found",
    "file not found",
    "content not found",
    "page doesn't exist",
    "page does not exist",
)

DEAD_PAGE_MARKERS = (
    "page not found",
    "file not found",
    "content not found",
    "page doesn't exist",
    "page does not exist",
    "content is unavailable",
    "this page is unavailable",
    "the page you requested could not be found",
    "requested page could not be found",
    "the requested page could not be found",
    "we couldn't find the page",
    "we could not find the page",
    "sorry, this page is unavailable",
)

STOPWORDS = {
    "a",
    "about",
    "after",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "being",
    "between",
    "but",
    "by",
    "can",
    "could",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "more",
    "most",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "should",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
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
    relevance_score: float = 0.0


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


def _tokenize(text: str) -> list[str]:
    """Tokenise text deterministically for lightweight lexical relevance scoring."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in _tokenize(text)
        if token not in STOPWORDS and len(token) >= 3
    }


def _relevance_score(
    question_text: str,
    title: str,
    text: str,
) -> float:
    """
    Score a candidate against the full forecasting question.

    This is intentionally non-LLM and inexpensive. Title matches carry more
    weight than body/snippet matches, and adjacent meaningful word pairs get
    an additional phrase-match bonus.
    """
    question_terms = _meaningful_tokens(question_text)
    title_terms = _meaningful_tokens(title)
    text_terms = _meaningful_tokens(text)

    if not question_terms:
        return 0.0

    title_hits = question_terms & title_terms
    text_hits = question_terms & text_terms

    score = len(title_hits) * 3.0 + len(text_hits) * 1.0

    question_words = _tokenize(question_text)
    combined_words = _tokenize(f"{title} {text}")
    combined_bigrams = set(zip(combined_words, combined_words[1:]))

    for index in range(len(question_words) - 1):
        first = question_words[index]
        second = question_words[index + 1]
        if first in STOPWORDS or second in STOPWORDS:
            continue
        if len(first) < 3 or len(second) < 3:
            continue
        if (first, second) in combined_bigrams:
            score += 4.0

    return score


def _looks_like_dead_page(text: str, title: str = "") -> bool:
    """Detect common short error/document-missing pages."""
    clean_text = (text or "").strip()
    clean_title = (title or "").strip().lower()

    if any(marker in clean_title for marker in DEAD_TITLE_MARKERS):
        return True

    # Do not reject long legitimate articles merely because they mention an
    # error-page phrase somewhere in their prose.
    if len(clean_text) <= 3000:
        haystack = f"{clean_title}\n{clean_text.lower()}"
        return any(marker in haystack for marker in DEAD_PAGE_MARKERS)

    return False


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
    relevance_text: str | None = None,
) -> list[dict]:
    """
    Search several DDGS backends, collect extra candidates, and rank them.

    ``max_results`` is the number of candidates returned after filtering, not
    the number fetched from the first backend. This avoids letting one search
    backend dominate the final source set.
    """
    query = query.strip()
    if not query:
        return []

    target_count = max(1, max_results)
    candidate_limit = max(target_count * SEARCH_CANDIDATE_MULTIPLIER, target_count + 4)
    backend_result_limit = max(target_count, 4)

    results_out: list[dict] = []
    seen_urls: set[str] = set()

    for backend in SEARCH_BACKENDS:
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        max_results=backend_result_limit,
                        backend=backend,
                    )
                )

            for result in results:
                if not _valid_result(result):
                    continue

                url = _normalise_url(result.get("href") or result.get("url") or "")
                if is_blocked_url(url) or not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                candidate = dict(result)
                title = (candidate.get("title") or "").strip()
                snippet = (candidate.get("body") or "").strip()
                candidate["_backend"] = backend
                candidate["_normalised_url"] = url
                candidate["_relevance_score"] = _relevance_score(
                    relevance_text or query,
                    title,
                    snippet,
                )
                results_out.append(candidate)

            # Stop only after at least two backends have had a chance to
            # contribute. This preserves some backend diversity without
            # forcing every query to hit every DDGS backend when results are
            # already plentiful.
            if len(results_out) >= candidate_limit and backend != SEARCH_BACKENDS[0]:
                break

        except Exception as exc:
            logger.warning(
                "[SEARCH] backend=%s failed for %r: %s",
                backend,
                query,
                exc,
            )

        time.sleep(SEARCH_DELAY_SECONDS)

    results_out.sort(
        key=lambda result: result.get("_relevance_score", 0.0),
        reverse=True,
    )

    if relevance_text:
        before = len(results_out)
        results_out = [
            result
            for result in results_out
            if result.get("_relevance_score", 0.0) >= MIN_METADATA_RELEVANCE_SCORE
        ]
        logger.info(
            "[SEARCH] Relevance filter kept %d/%d candidates for %r",
            len(results_out),
            before,
            query,
        )

    return results_out[:target_count]


def _fetch_html(url: str) -> tuple[str | None, str, str]:
    """
    Fetch HTML once and return ``(html, final_url, status)``.

    ``status`` distinguishes confirmed dead URLs from transient request
    failures. A confirmed 404/410 must never fall through to the stale DDGS
    snippet fallback.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT,
            allow_redirects=True,
        )

        final_url = response.url or url

        if response.status_code in {404, 410}:
            return None, final_url, "dead_http"
        if response.status_code >= 400:
            return None, final_url, f"http_{response.status_code}"
        if is_blocked_url(final_url):
            return None, final_url, "redirected_blocked"

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None, final_url, "non_html"

        return response.text, final_url, "ok"

    except requests.RequestException as exc:
        logger.debug("[SCRAPE] request failed for %s: %s", url, exc)
        return None, url, "request_failed"


def scrape_with_trafilatura(downloaded: str, title: str = "") -> str | None:
    """Extract article text from already-downloaded HTML."""
    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        )
        if not text:
            return None
        text = text.strip()
        if len(text) < 100 or _looks_like_dead_page(text, title):
            return None
        return text
    except Exception as exc:
        logger.debug("[SCRAPE] trafilatura failed: %s", exc)
        return None


def scrape_with_bs4(
    downloaded: str,
    title: str = "",
) -> str | None:
    """Fallback paragraph extraction from already-downloaded HTML."""
    try:
        soup = BeautifulSoup(downloaded, "html.parser")
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]
        ):
            tag.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(" ", strip=True)) >= 40
        ]
        text = "\n".join(paragraphs).strip()

        if len(text) < 100 or _looks_like_dead_page(text, title):
            return None
        return text
    except Exception as exc:
        logger.debug("[SCRAPE] BeautifulSoup failed: %s", exc)
        return None


def scrape_with_ddgs_snippet(
    url: str,
    title: str = "",
    original_snippet: str = "",
) -> str | None:
    """
    LAST-RESORT scraper layer for pages that are unavailable to direct HTTP.

    The exact URL is searched first, then the page title. The old behaviour of
    blindly accepting the original search-result snippet has intentionally
    been removed because that snippet can survive after a URL becomes dead.
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
                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        max_results=8,
                        backend="auto",
                    )
                )

            for result in results:
                result_url = _normalise_url(
                    result.get("href") or result.get("url") or ""
                )
                result_title = (result.get("title") or "").strip()
                body = (result.get("body") or "").strip()

                if is_blocked_url(result_url):
                    continue
                if result_url != target or len(body) < 40:
                    continue
                if _looks_like_dead_page(body, result_title):
                    logger.debug(
                        "[SCRAPE] rejected dead-page snippet for %s",
                        url,
                    )
                    continue
                return body

        except Exception as exc:
            logger.debug(
                "[SCRAPE] DDGS snippet fallback failed for %s: %s",
                url,
                exc,
            )

    # Deliberately do not return original_snippet here. It is stale-index
    # metadata, not proof that the URL is alive or that the page was retrieved.
    return None


def scrape_url(
    url: str,
    title: str = "",
    snippet: str = "",
    relevance_text: str | None = None,
) -> tuple[str, str, float]:
    """Fetch, extract, validate, and relevance-check one source."""
    if is_blocked_url(url):
        return "", "blocked", 0.0

    downloaded, final_url, fetch_status = _fetch_html(url)

    # A 404/410 is definitive. Never resurrect it with a cached search snippet.
    if fetch_status == "dead_http":
        logger.info("[SOURCE] rejected dead HTTP page status=%s url=%s", fetch_status, url)
        return "", "dead_http", 0.0

    if downloaded:
        # Some sites return a soft-404 as HTTP 200. Check the rendered page
        # text before allowing the indexed-snippet fallback to revive it.
        try:
            error_soup = BeautifulSoup(downloaded, "html.parser")
            page_title = (
                error_soup.title.get_text(" ", strip=True)
                if error_soup.title
                else ""
            )
            visible_text = error_soup.get_text(" ", strip=True)[:3000]
            if _looks_like_dead_page(visible_text, f"{title} {page_title}"):
                logger.info(
                    "[SOURCE] rejected soft-dead page url=%s",
                    final_url,
                )
                return "", "dead_content", 0.0
        except Exception:
            pass

        text = scrape_with_trafilatura(downloaded, title=title)
        method = "trafilatura" if text else ""
        if not text:
            text = scrape_with_bs4(downloaded, title=title)
            method = "bs4" if text else ""

        if text:
            relevance_score = _relevance_score(
                relevance_text or title,
                title,
                text[:30000],
            )
            if relevance_text and relevance_score < MIN_CONTENT_RELEVANCE_SCORE:
                logger.info(
                    "[SOURCE] rejected low content relevance score=%.2f url=%s",
                    relevance_score,
                    final_url,
                )
                return "", "low_relevance", relevance_score
            return text, method, relevance_score

    # For transient failures / anti-bot pages, an exact-URL indexed snippet can
    # still be useful. Confirmed HTTP errors are already excluded above.
    if fetch_status not in {"non_html", "redirected_blocked"}:
        text = scrape_with_ddgs_snippet(
            final_url or url,
            title=title,
            original_snippet=snippet,
        )
        if text:
            relevance_score = _relevance_score(
                relevance_text or title,
                title,
                text,
            )
            if relevance_text and relevance_score < MIN_CONTENT_RELEVANCE_SCORE:
                logger.info(
                    "[SOURCE] rejected low snippet relevance score=%.2f url=%s",
                    relevance_score,
                    final_url or url,
                )
                return "", "low_relevance", relevance_score
            return text, "ddgs_snippet", relevance_score

    return "", "failed", 0.0


def gather_sources(
    queries: Iterable[str],
    results_per_query: int = 3,
    relevance_text: str | None = None,
) -> list[ScrapedSource]:
    """Search, filter, scrape, and validate sources for each query."""
    sources: list[ScrapedSource] = []
    seen_urls: set[str] = set()
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

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
            relevance_text=relevance_text,
        )

        for result in results:
            url = _normalise_url(
                result.get("href") or result.get("url") or ""
            )
            if is_blocked_url(url) or not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = (result.get("title") or url).strip()
            snippet = (result.get("body") or "").strip()
            backend = result.get("_backend") or "unknown"

            content, method, relevance_score = scrape_url(
                url,
                title=title,
                snippet=snippet,
                relevance_text=relevance_text,
            )

            if not content:
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
                    relevance_score=relevance_score,
                )
            )
            logger.info(
                "[SOURCE] accepted score=%.2f backend=%s method=%s url=%s",
                relevance_score,
                backend,
                method,
                url,
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
        f"(query: {source.query!r}; backend: {source.backend}; extraction: {source.method}; relevance: {source.relevance_score:.2f})\n\n"
        f"{source.content}\n"
        for source in sources
    )


def web_search(
    query: str,
    max_results: int = 4,
    scrape: bool = True,
    relevance_text: str | None = None,
) -> str:
    """Run one blocking web search and return its validated source text."""
    query = query.strip()
    if not query:
        return ""

    relevance_text = relevance_text.strip() if relevance_text else None

    if scrape:
        sources = gather_sources(
            [query],
            results_per_query=max_results,
            relevance_text=relevance_text,
        )
        return format_sources_as_markdown(sources)

    results = ddgs_search(
        query,
        max_results=max_results,
        relevance_text=relevance_text,
    )
    sources = [
        ScrapedSource(
            query=query,
            title=(result.get("title") or result.get("href") or "").strip(),
            url=_normalise_url(result.get("href") or result.get("url") or ""),
            snippet=(result.get("body") or "").strip(),
            content=(result.get("body") or "").strip(),
            method="ddgs_snippet",
            backend=result.get("_backend") or "unknown",
            relevance_score=float(result.get("_relevance_score", 0.0)),
        )
        for result in results
        if not is_blocked_url(
            _normalise_url(result.get("href") or result.get("url") or "")
        )
    ]
    return format_sources_as_markdown(sources)
