"""RAG ingest/query modules (Step 2+: chunk contract, embed, store).

Parsing stays in ``document_parser``. This package consumes
``CanonicalDocument`` / Elements JSON and produces storable chunks.
"""

from .schemas import Chunk

__all__ = ["Chunk"]
