"""Load seed documents into Chroma (dense index)."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.config import get_settings
from app.ingestion.loaders import chunk_documents, load_file
from app.pipeline.chroma_kb import reset, upsert_chunks


def ingest_seed(reset_index: bool = True) -> int:
    settings = get_settings()
    root = Path(settings.seed_dir)
    if reset_index:
        reset()
    rows: list[dict] = []
    for path in sorted(root.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf", ".docx", ".html"}:
            continue
        docs = load_file(path)
        chunks = chunk_documents(docs)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path.stem))
        for chunk in chunks:
            rows.append(
                {
                    "chunk_id": f"{doc_id}-{chunk['chunk_index']}",
                    "document_id": doc_id,
                    "title": path.stem,
                    "source": chunk.get("source") or path.suffix.lstrip("."),
                    "url": chunk.get("url") or "",
                    "page_number": chunk.get("page_number"),
                    "content": chunk["content"],
                }
            )
    return upsert_chunks(rows)


if __name__ == "__main__":
    n = ingest_seed()
    print(f"Indexed {n} chunks into Chroma at {get_settings().chroma_path}")
