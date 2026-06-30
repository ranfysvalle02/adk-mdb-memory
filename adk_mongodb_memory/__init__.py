"""adk_mongodb_memory - MongoDB Atlas Vector Search memory for Google ADK.

A light, production-oriented wrapper that implements Google ADK's
``BaseMemoryService`` on top of MongoDB Atlas Vector Search, with Voyage AI
embeddings.

Public API::

    from adk_mongodb_memory import MongoAtlasMemoryService, VoyageEmbedder

    embedder = VoyageEmbedder(model="voyage-3.5", output_dimension=1024)
    service = MongoAtlasMemoryService(
        connection_string="mongodb+srv://...",
        embedding_fn=embedder,
        embedding_dimensions=1024,
    )
"""

from __future__ import annotations

from .embeddings import (
    DEFAULT_VOYAGE_DIMENSIONS,
    DEFAULT_VOYAGE_MODEL,
    VoyageEmbedder,
    build_voyage_embedder,
)
from .service import (
    DOCUMENT_INPUT,
    QUERY_INPUT,
    DistillFn,
    EmbeddingFn,
    MongoAtlasMemoryService,
)

__version__ = "0.1.0"

__all__ = [
    "MongoAtlasMemoryService",
    "VoyageEmbedder",
    "build_voyage_embedder",
    "EmbeddingFn",
    "DistillFn",
    "DOCUMENT_INPUT",
    "QUERY_INPUT",
    "DEFAULT_VOYAGE_MODEL",
    "DEFAULT_VOYAGE_DIMENSIONS",
    "__version__",
]
