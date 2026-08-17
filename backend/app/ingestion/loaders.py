from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import SentenceSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: str | Path) -> list[LIDocument]:
    path = Path(path)
    suffix = path.suffix.lower()
    extra: dict[str, Any] = {"filename": path.name, "url": None}
    if suffix in {".md", ".txt", ".html", ".htm"}:
        text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
        return [LIDocument(text=text, metadata={**extra, "title": path.stem, "source": suffix.lstrip(".")})]
    if suffix == ".pdf":
        from llama_index.readers.file import PDFReader

        docs = PDFReader().load_data(file=path)
        for d in docs:
            d.text = _clean(d.text)
            d.metadata = {**extra, **d.metadata, "title": path.stem, "source": "pdf"}
        return docs
    if suffix == ".docx":
        from llama_index.readers.file import DocxReader

        docs = DocxReader().load_data(file=path)
        for d in docs:
            d.text = _clean(d.text)
            d.metadata = {**extra, **d.metadata, "title": path.stem, "source": "docx"}
        return docs
    raise ValueError(f"Unsupported file type: {suffix}")


def load_url(url: str) -> list[LIDocument]:
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch URL: {url}")
    text = trafilatura.extract(downloaded, include_comments=False) or ""
    text = _clean(text)
    if not text:
        raise ValueError(f"No extractable text at {url}")
    title = url
    match = re.search(r"<title>(.*?)</title>", downloaded, re.I | re.S)
    if match:
        title = _clean(re.sub(r"<[^>]+>", "", match.group(1)))
    return [LIDocument(text=text, metadata={"title": title, "source": "web", "url": url})]


def load_arxiv(arxiv_id: str) -> list[LIDocument]:
    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id.strip()])
    paper = next(client.results(search))
    abstract = _clean(paper.summary)
    authors = ", ".join(a.name for a in paper.authors[:8])
    body = f"# {paper.title}\n\nAuthors: {authors}\nPublished: {paper.published.date()}\nURL: {paper.entry_id}\n\n{abstract}"
    return [
        LIDocument(
            text=body,
            metadata={
                "title": paper.title,
                "author": authors,
                "source": "arxiv",
                "url": paper.entry_id,
                "publication_date": str(paper.published.date()),
                "topic": ", ".join(paper.categories[:4]),
            },
        )
    ]


def load_markdown_text(text: str, title: str = "Pasted note") -> list[LIDocument]:
    return [LIDocument(text=_clean(text), metadata={"title": title, "source": "markdown", "url": None})]


def chunk_documents(docs: list[LIDocument]) -> list[dict[str, Any]]:
    settings = get_settings()
    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    chunks: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        meta = dict(node.metadata or {})
        pub = meta.get("publication_date")
        pub_date = None
        if isinstance(pub, date):
            pub_date = pub
        elif isinstance(pub, str) and pub:
            try:
                pub_date = date.fromisoformat(pub[:10])
            except ValueError:
                pub_date = None
        chunks.append(
            {
                "chunk_index": i,
                "content": node.get_content(),
                "page_number": meta.get("page_label") or meta.get("page") or meta.get("page_number"),
                "title": meta.get("title") or "Untitled",
                "author": meta.get("author"),
                "source": meta.get("source") or "upload",
                "url": meta.get("url"),
                "domain": meta.get("domain"),
                "topic": meta.get("topic"),
                "publication_date": pub_date,
                "extra": meta,
            }
        )
    return chunks
