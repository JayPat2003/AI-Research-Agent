"""Document ingestion pipeline.

Loaders normalize PDF/DOCX/Markdown/HTML/URL/arXiv sources into LlamaIndex
documents, chunk them, embed them locally, and persist ``Document`` / ``Chunk``
rows. API requests enqueue Redis jobs; :mod:`app.worker` executes them
off the request thread.
"""
