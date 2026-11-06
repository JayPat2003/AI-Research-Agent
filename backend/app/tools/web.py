from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SCHOLAR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _clean_scholar_url(href: str | None) -> str:
    if not href:
        return ""
    if href.startswith("/"):
        href = "https://scholar.google.com" + href
    if "scholar.google." in href and "/scholar_url" in href:
        qs = parse_qs(urlparse(href).query)
        target = qs.get("url") or qs.get("q")
        if target:
            return unquote(target[0])
    return href


def _parse_scholar_html(html: str, max_results: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, Any]] = []
    for block in soup.select("div.gs_r.gs_or.gs_scl"):
        if len(results) >= max_results:
            break
        title_el = block.select_one("h3.gs_rt a") or block.select_one("h3.gs_rt")
        if not title_el:
            continue
        title = title_el.get_text(" ", strip=True)
        title = re.sub(r"^\[(?:PDF|HTML|CITATION)\]\s*", "", title, flags=re.I).strip()
        if not title:
            continue
        url = _clean_scholar_url(title_el.get("href") if title_el.name == "a" else None)
        snippet_el = block.select_one("div.gs_rs")
        meta_el = block.select_one("div.gs_a")
        meta = meta_el.get_text(" ", strip=True) if meta_el else ""
        year_match = re.search(r"\b(19|20)\d{2}\b", meta)
        authors = meta.split("-")[0].strip() if meta else ""
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": (snippet_el.get_text(" ", strip=True) if snippet_el else "")[:800],
                "source": "google_scholar",
                "authors": authors,
                "published": year_match.group(0) if year_match else "",
            }
        )
    return results


def search_google_scholar(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search Google Scholar for academic papers. No API key required."""
    results = _search_scholar_http(query, max_results)
    if results:
        return results
    return _search_scholar_library(query, max_results)


def _search_scholar_http(query: str, max_results: int) -> list[dict[str, Any]]:
    url = f"https://scholar.google.com/scholar?q={quote_plus(query)}&hl=en&as_sdt=0,5"
    try:
        with httpx.Client(follow_redirects=True, timeout=25.0, headers=_SCHOLAR_HEADERS) as client:
            resp = client.get(url)
        if resp.status_code >= 400:
            logger.warning("Google Scholar HTTP %s", resp.status_code)
            return []
        lower = resp.text.lower()
        if "captcha" in lower or "unusual traffic" in lower:
            logger.warning("Google Scholar returned a CAPTCHA / bot check page")
            return []
        return _parse_scholar_html(resp.text, max_results)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Google Scholar HTTP search failed: %s", exc)
        return []


def _search_scholar_library(query: str, max_results: int) -> list[dict[str, Any]]:
    """Optional fallback via the scholarly package if installed."""
    try:
        from scholarly import scholarly
    except ImportError:
        return []
    try:
        results: list[dict[str, Any]] = []
        for i, pub in enumerate(scholarly.search_pubs(query)):
            if i >= max_results:
                break
            bib = pub.get("bib") or {}
            results.append(
                {
                    "title": bib.get("title") or "",
                    "url": pub.get("pub_url") or pub.get("eprint_url") or "",
                    "snippet": (bib.get("abstract") or "")[:800],
                    "source": "google_scholar",
                    "authors": bib.get("author") or "",
                    "published": str(bib.get("pub_year") or ""),
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Google Scholar library search failed: %s", exc)
        return []


def search_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        for paper in client.results(search):
            papers.append(
                {
                    "title": paper.title,
                    "url": paper.entry_id,
                    "snippet": paper.summary.replace("\n", " ")[:800],
                    "source": "arxiv",
                    "authors": ", ".join(a.name for a in paper.authors[:6]),
                    "published": str(paper.published.date()),
                }
            )
        return papers
    except Exception as exc:  # noqa: BLE001
        logger.warning("arXiv search failed: %s", exc)
        return []


def search_semantic_scholar(query: str, max_results: int = 5, api_key: str | None = None) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,url,authors,externalIds",
    }
    headers = {"User-Agent": "ai-research-agent/0.1"}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers,
            timeout=20.0,
        )
        if resp.status_code >= 400:
            logger.warning("Semantic Scholar HTTP %s", resp.status_code)
            return []
        data = resp.json()
        results = []
        for paper in data.get("data") or []:
            authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or [])[:6])
            results.append(
                {
                    "title": paper.get("title") or "",
                    "url": paper.get("url") or "",
                    "snippet": (paper.get("abstract") or "")[:800],
                    "source": "semantic_scholar",
                    "authors": authors,
                    "published": str(paper.get("year") or ""),
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic Scholar search failed: %s", exc)
        return []
