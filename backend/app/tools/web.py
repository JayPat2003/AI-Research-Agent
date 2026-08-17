from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """DuckDuckGo text search. No API key required."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        results = []
        for hit in hits:
            results.append(
                {
                    "title": hit.get("title") or "",
                    "url": hit.get("href") or hit.get("url") or "",
                    "snippet": hit.get("body") or hit.get("snippet") or "",
                    "source": "web",
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed: %s", exc)
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
