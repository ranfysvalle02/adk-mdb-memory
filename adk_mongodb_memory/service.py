"""MongoDB Atlas backend for the Google ADK ``BaseMemoryService`` abstraction.

This module provides :class:`MongoAtlasMemoryService`, a production-oriented,
fully asynchronous long-term memory provider for the Google Agent Development
Kit (ADK). It stores conversation transcripts in a MongoDB Atlas collection and
retrieves them with `Atlas Vector Search
<https://www.mongodb.com/docs/atlas/atlas-vector-search/>`_ (``$vectorSearch``).

Design goals
------------
* **Exact ADK compliance** - subclasses
  :class:`google.adk.memory.base_memory_service.BaseMemoryService` and matches
  the framework's *keyword-only* ``search_memory`` signature so the built-in
  ``load_memory`` / ``PreloadMemoryTool`` wiring works unchanged.
* **No event-loop blocking** - uses :class:`pymongo.AsyncMongoClient` and awaits
  every database operation. Async embedding/distill callables - including
  *callable objects* whose ``__call__`` is ``async def`` (e.g. the bundled
  Voyage embedder built on ``voyageai.AsyncClient``) - are awaited directly on
  the running loop; genuinely synchronous callables are off-loaded with
  :func:`asyncio.to_thread` so a blocking HTTP call never stalls the loop.
* **Retrieval-tuned embeddings** - the embedder contract is ``input_type``-aware
  (``"document"`` for stored memories, ``"query"`` for searches). Voyage (and
  most modern retrieval embedders) produce materially better recall when the
  query and the document are embedded with the correct intent.
* **Operable** - programmatic creation of the supporting lookup index, the Atlas
  Vector Search index, and an optional compliance TTL index, plus lifecycle
  management (``close`` / async context manager).
* **Multi-tenant by default** - every search is constrained to a single
  ``app_name`` + ``user_id``.

The service is intentionally a *light wrapper*: it leans on native PyMongo async
features rather than third-party glue, and exposes documented extension points
(distillation, CSFLE via an injected ``client``) instead of fragile
half-implementations.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from google.adk.events import Event
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions import Session
from google.genai.types import Content, Part
from pymongo import AsyncMongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing private paths at runtime
    from pymongo.asynchronous.collection import AsyncCollection
    from pymongo.asynchronous.database import AsyncDatabase

logger = logging.getLogger(__name__)

# An embedding function maps ``(text, input_type)`` -> a dense vector, where
# ``input_type`` is ``"document"`` (text being stored) or ``"query"`` (a search
# query). Retrieval embedders such as Voyage use this to tailor the vector to
# its role, which noticeably improves recall. A bring-your-own embedder that
# does not care about the distinction can simply accept and ignore the second
# argument. The callable may be synchronous or asynchronous.
EmbeddingFn = Callable[[str, str], Union[Sequence[float], Awaitable[Sequence[float]]]]
# A distillation function compresses a raw transcript -> stored text. Sync or async.
DistillFn = Callable[[str], Union[str, Awaitable[str]]]

#: Allowed ``input_type`` values passed to the embedder, mirroring Voyage's API.
DOCUMENT_INPUT = "document"
QUERY_INPUT = "query"

_VALID_SIMILARITIES = frozenset({"cosine", "euclidean", "dotProduct"})
# Atlas/Mongo error codes that mean "this search index already exists".
_INDEX_EXISTS_CODES = frozenset({68, 11000})


class MongoAtlasMemoryService(BaseMemoryService):
    """Long-term memory for ADK agents backed by MongoDB Atlas Vector Search.

    A single document is stored per ``(app_name, user_id, session_id)`` tuple.
    Each document holds the joined conversation transcript, its embedding vector,
    and bookkeeping timestamps. ``search_memory`` performs an approximate
    nearest-neighbour ``$vectorSearch`` pre-filtered to the calling tenant.

    Args:
        connection_string:
            MongoDB Atlas SRV connection string (``mongodb+srv://...``) or any
            MongoDB URI. Required unless a pre-built ``client`` is supplied.
        db_name:
            Target database name.
        embedding_fn:
            Callable that turns text into a dense vector. It is invoked as
            ``embedding_fn(text, input_type)`` where ``input_type`` is
            ``"document"`` for stored memories and ``"query"`` for searches (see
            :data:`EmbeddingFn`). May be synchronous (run via
            :func:`asyncio.to_thread`) or asynchronous - a coroutine function or
            a callable object with an ``async def __call__`` such as the bundled
            Voyage embedder (awaited directly). **Required.**
        collection_name:
            Collection that stores memory documents. Defaults to
            ``"agent_memories"``.
        vector_index_name:
            Name of the Atlas Vector Search index. Defaults to ``"vector_index"``.
        embedding_field:
            Document field holding the embedding vector. Defaults to
            ``"embedding"``.
        transcript_field:
            Document field holding the stored (optionally distilled) transcript.
            Defaults to ``"transcript"``.
        embedding_dimensions:
            Dimensionality of the embedding vectors. Must match both your
            embedding model's output and the vector index definition. Defaults to
            ``1024`` (Voyage ``voyage-3.5`` and the ``voyage-4`` family default,
            which also support 256 / 512 / 1024 / 2048).
        similarity:
            Vector similarity metric: ``"cosine"`` (default), ``"euclidean"`` or
            ``"dotProduct"``. Voyage float embeddings are unit-normalized and
            cosine is scale-invariant, so no manual normalization is needed.
        default_search_limit:
            Number of results returned by ``search_memory``. The ADK framework
            never passes a limit, so it is fixed here. Defaults to ``5``.
        num_candidates_multiplier:
            ``numCandidates`` for ``$vectorSearch`` is computed as
            ``default_search_limit * num_candidates_multiplier`` (ANN recall vs.
            latency trade-off). Defaults to ``20`` per MongoDB's guidance of
            "at least 20x the limit", which also helps recall when the tenant
            pre-filter is selective.
        ttl_seconds:
            If set, :meth:`setup_indexes` creates a TTL index so documents are
            auto-purged this many seconds after ``updated_at`` (GDPR/HIPAA-style
            retention). Defaults to ``None`` (no auto-purge).
        distill_fn:
            Optional callable that compresses the raw transcript before embedding
            and storage (memory compression / fact extraction). Sync or async. If
            ``None`` (default), the full transcript is stored.
        memory_author:
            ``author`` recorded on transcript-derived memories and surfaced on
            returned :class:`MemoryEntry` objects. Defaults to ``"memory"``.
        client:
            An existing :class:`pymongo.AsyncMongoClient` to reuse instead of
            constructing one. Supplying your own client is also the supported
            extension point for **CSFLE** (configure ``AutoEncryptionOpts`` on the
            client for transparent at-rest field encryption). When provided, the
            service will not close it on ``close()``.

    Raises:
        ValueError: If ``embedding_fn`` is missing, ``similarity`` is invalid, or
            neither ``connection_string`` nor ``client`` is supplied.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        db_name: str = "adk_memory",
        *,
        embedding_fn: EmbeddingFn,
        collection_name: str = "agent_memories",
        vector_index_name: str = "vector_index",
        embedding_field: str = "embedding",
        transcript_field: str = "transcript",
        embedding_dimensions: int = 1024,
        similarity: str = "cosine",
        default_search_limit: int = 5,
        num_candidates_multiplier: int = 20,
        ttl_seconds: Optional[int] = None,
        distill_fn: Optional[DistillFn] = None,
        memory_author: str = "memory",
        client: Optional[AsyncMongoClient] = None,
    ) -> None:
        if embedding_fn is None:
            raise ValueError(
                "embedding_fn is required - memory recall depends on vector "
                "embeddings. Pass a sync or async callable mapping "
                "(text, input_type) -> list[float]."
            )
        if similarity not in _VALID_SIMILARITIES:
            raise ValueError(
                f"similarity must be one of {sorted(_VALID_SIMILARITIES)}, got {similarity!r}."
            )
        if default_search_limit < 1:
            raise ValueError("default_search_limit must be >= 1.")
        if embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be >= 1.")

        if client is not None:
            self.client: AsyncMongoClient = client
            self._owns_client = False
        else:
            if not connection_string:
                raise ValueError(
                    "Provide a connection_string (mongodb+srv://...) or an "
                    "AsyncMongoClient via the `client` argument."
                )
            # Native async client - bound cleanly to the running event loop.
            self.client = AsyncMongoClient(connection_string)
            self._owns_client = True

        self.db: "AsyncDatabase" = self.client[db_name]
        self.collection: "AsyncCollection" = self.db[collection_name]

        self.embedding_fn = embedding_fn
        self.distill_fn = distill_fn

        self.collection_name = collection_name
        self.vector_index_name = vector_index_name
        self.embedding_field = embedding_field
        self.transcript_field = transcript_field
        self.embedding_dimensions = embedding_dimensions
        self.similarity = similarity
        self.default_search_limit = default_search_limit
        self.num_candidates_multiplier = max(1, num_candidates_multiplier)
        self.ttl_seconds = ttl_seconds
        self.memory_author = memory_author

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def close(self) -> None:
        """Close the underlying client (only if this service created it)."""
        if self._owns_client:
            await self.client.close()
            logger.debug("Closed AsyncMongoClient owned by MongoAtlasMemoryService.")

    async def __aenter__(self) -> "MongoAtlasMemoryService":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #
    async def setup_indexes(
        self,
        *,
        create_vector_index: bool = True,
        wait_for_vector_index: bool = False,
        timeout_seconds: float = 180.0,
    ) -> None:
        """Create all supporting indexes. Safe to call repeatedly (idempotent).

        This creates, in order:

        1. A unique compound index on ``(app_name, user_id, session_id)`` for fast
           tenant/session lookups and to enforce one document per session.
        2. An optional TTL index on ``updated_at`` when ``ttl_seconds`` is set.
        3. The Atlas Vector Search index (when ``create_vector_index`` is true).

        Args:
            create_vector_index: Whether to create the ``$vectorSearch`` index.
            wait_for_vector_index: Block until the vector index reports queryable.
            timeout_seconds: Max seconds to wait when ``wait_for_vector_index``.
        """
        await self.create_lookup_index()
        if self.ttl_seconds is not None:
            await self.create_ttl_index()
        if create_vector_index:
            await self.create_vector_search_index(
                wait_for_ready=wait_for_vector_index,
                timeout_seconds=timeout_seconds,
            )

    async def create_lookup_index(self) -> str:
        """Create the unique compound index used for tenant/session upserts."""
        name = await self.collection.create_index(
            [("app_name", 1), ("user_id", 1), ("session_id", 1)],
            unique=True,
            name="idx_tenant_session_lookup",
        )
        logger.info("Ensured compound lookup index %r on %s.", name, self.collection_name)
        return name

    async def create_ttl_index(self) -> Optional[str]:
        """Create a TTL index on ``updated_at`` when ``ttl_seconds`` is configured.

        MongoDB's background TTL monitor purges documents ``ttl_seconds`` after
        their ``updated_at`` time, providing zero-cron compliance retention.
        """
        if self.ttl_seconds is None:
            logger.debug("ttl_seconds is None; skipping TTL index.")
            return None
        name = await self.collection.create_index(
            [("updated_at", 1)],
            expireAfterSeconds=int(self.ttl_seconds),
            name="idx_ttl_updated_at",
        )
        logger.info(
            "Ensured TTL index %r (expireAfterSeconds=%s) on %s.",
            name,
            self.ttl_seconds,
            self.collection_name,
        )
        return name

    def _vector_index_definition(self) -> dict[str, Any]:
        """Build the Atlas Vector Search index definition document.

        Declares the embedding vector field plus ``app_name``/``user_id`` as
        ``filter`` fields so the tenant pre-filter in :meth:`search_memory` is
        index-backed (a hard requirement for filtered ``$vectorSearch``).
        """
        return {
            "fields": [
                {
                    "type": "vector",
                    "path": self.embedding_field,
                    "numDimensions": self.embedding_dimensions,
                    "similarity": self.similarity,
                },
                {"type": "filter", "path": "app_name"},
                {"type": "filter", "path": "user_id"},
            ]
        }

    async def create_vector_search_index(
        self,
        *,
        wait_for_ready: bool = False,
        timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 5.0,
    ) -> bool:
        """Create the Atlas Vector Search index idempotently.

        Args:
            wait_for_ready: If true, poll until the index is queryable.
            timeout_seconds: Max seconds to wait when ``wait_for_ready``.
            poll_interval_seconds: Delay between readiness polls.

        Returns:
            ``True`` if the index was newly created, ``False`` if it already
            existed.

        Raises:
            OperationFailure: For non-"already exists" Atlas errors (e.g. the
                cluster tier does not support Vector Search).
        """
        model = SearchIndexModel(
            definition=self._vector_index_definition(),
            name=self.vector_index_name,
            type="vectorSearch",
        )
        created = True
        try:
            await self.collection.create_search_index(model)
            logger.info(
                "Submitted Atlas Vector Search index %r (dims=%s, similarity=%s). "
                "Atlas builds it asynchronously.",
                self.vector_index_name,
                self.embedding_dimensions,
                self.similarity,
            )
        except OperationFailure as exc:
            if self._is_already_exists(exc):
                logger.info(
                    "Atlas Vector Search index %r already exists; leaving it in place.",
                    self.vector_index_name,
                )
                created = False
            else:
                logger.error(
                    "Failed to create Atlas Vector Search index %r: %s. "
                    "Vector Search requires an Atlas cluster that supports it "
                    "(e.g. M10+ or Flex), or the MongoDB Atlas Local image for dev.",
                    self.vector_index_name,
                    exc,
                )
                raise

        if wait_for_ready:
            await self.wait_for_vector_index_ready(
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        return created

    async def wait_for_vector_index_ready(
        self,
        *,
        timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 5.0,
    ) -> bool:
        """Poll ``listSearchIndexes`` until the vector index is queryable.

        Returns ``True`` once the index reports queryable, or ``False`` if the
        timeout elapses first (a warning is logged - the caller may still try to
        query, it will just error until the build completes).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            status = await self._vector_index_status()
            if status is not None and (
                status.get("queryable") is True or status.get("status") == "READY"
            ):
                logger.info("Atlas Vector Search index %r is queryable.", self.vector_index_name)
                return True
            if loop.time() >= deadline:
                logger.warning(
                    "Timed out after %.0fs waiting for vector index %r to become "
                    "queryable (last status=%s). Searches may fail until it finishes building.",
                    timeout_seconds,
                    self.vector_index_name,
                    None if status is None else status.get("status"),
                )
                return False
            await asyncio.sleep(poll_interval_seconds)

    async def _vector_index_status(self) -> Optional[dict[str, Any]]:
        """Return the raw status doc for the vector index, or ``None`` if absent."""
        try:
            cursor = await self.collection.list_search_indexes(self.vector_index_name)
            indexes = await cursor.to_list(length=None)
        except OperationFailure as exc:
            logger.debug("listSearchIndexes failed (index may not exist yet): %s", exc)
            return None
        for index in indexes:
            if index.get("name") == self.vector_index_name:
                return index
        return None

    @staticmethod
    def _is_already_exists(exc: OperationFailure) -> bool:
        """Heuristically detect an "index already exists" Atlas error."""
        if getattr(exc, "code", None) in _INDEX_EXISTS_CODES:
            return True
        return "already exists" in str(exc).lower()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def add_session_to_memory(self, session: Session) -> None:
        """Ingest a (usually completed) session into long-term memory.

        Joins every text part across the session's events into a transcript,
        optionally distills it, embeds it (as a ``"document"``), and upserts a
        single document keyed by ``(app_name, user_id, session.id)``. Sessions
        with no text content are skipped.
        """
        segments = self._segments_from_events(session.events)
        if not segments:
            logger.info("Session %s contains no text content; nothing to store.", session.id)
            return

        transcript = self._format_segments(segments)
        stored_text = await self._distill(transcript)
        embedding = await self._embed(stored_text, DOCUMENT_INPUT)
        event_ids = [seg[0] for seg in segments if seg[0]]

        await self._upsert_session_document(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.id,
            stored_text=stored_text,
            embedding=embedding,
            event_ids=event_ids,
            source="add_session_to_memory",
        )
        logger.info(
            "Stored memory for session %s (app=%s, user=%s, %d chars, %d dims).",
            session.id,
            session.app_name,
            session.user_id,
            len(stored_text),
            len(embedding),
        )

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: Optional[str] = None,
        custom_metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Incrementally append a delta of events to a session's memory document.

        New events (deduplicated by ``event.id``) are appended to the stored
        transcript, then the combined transcript is re-embedded (as a
        ``"document"``) and upserted. When ``distill_fn`` is set, the *delta* is
        distilled before being appended.

        Note:
            Without a ``session_id`` the events are written under a stable
            ``"_default"`` bucket for the tenant.
        """
        segments = self._segments_from_events(events)
        if not segments:
            logger.info("add_events_to_memory: no text content in delta; skipping.")
            return

        scoped_session_id = session_id or "_default"
        existing = await self.collection.find_one(
            {"app_name": app_name, "user_id": user_id, "session_id": scoped_session_id}
        )
        existing_ids: set[str] = set(existing.get("event_ids", [])) if existing else set()

        new_segments = [seg for seg in segments if not seg[0] or seg[0] not in existing_ids]
        if not new_segments:
            logger.info("add_events_to_memory: all events already stored; skipping.")
            return

        delta_text = self._format_segments(new_segments)
        delta_stored = await self._distill(delta_text)

        prior_text = (existing or {}).get(self.transcript_field, "") if existing else ""
        combined_text = f"{prior_text}\n{delta_stored}".strip() if prior_text else delta_stored
        embedding = await self._embed(combined_text, DOCUMENT_INPUT)

        combined_event_ids = list(existing_ids) + [seg[0] for seg in new_segments if seg[0]]
        await self._upsert_session_document(
            app_name=app_name,
            user_id=user_id,
            session_id=scoped_session_id,
            stored_text=combined_text,
            embedding=embedding,
            event_ids=combined_event_ids,
            source="add_events_to_memory",
            custom_metadata=custom_metadata,
        )
        logger.info(
            "Appended %d event(s) to memory for session %s (app=%s, user=%s).",
            len(new_segments),
            scoped_session_id,
            app_name,
            user_id,
        )

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Insert pre-built :class:`MemoryEntry` items directly.

        Each entry's text parts are joined, embedded (as a ``"document"``), and
        inserted as an independent memory document (not keyed by session).
        Entries with no text are skipped. ``custom_metadata`` is merged onto
        every inserted document - this is a convenient hook for storing
        non-vectorized side fields (including CSFLE ciphertext; see the CSFLE
        example).
        """
        documents: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for entry in memories:
            text = self._extract_entry_text(entry)
            if not text:
                continue
            embedding = await self._embed(text, DOCUMENT_INPUT)
            merged_metadata: dict[str, Any] = {}
            if entry.custom_metadata:
                merged_metadata.update(entry.custom_metadata)
            if custom_metadata:
                merged_metadata.update(custom_metadata)
            documents.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": None,
                    self.transcript_field: text,
                    self.embedding_field: embedding,
                    "author": entry.author or self.memory_author,
                    "custom_metadata": merged_metadata,
                    "source": "add_memory",
                    "created_at": now,
                    "updated_at": now,
                }
            )

        if not documents:
            logger.info("add_memory: no non-empty memories to insert.")
            return
        await self.collection.insert_many(documents)
        logger.info(
            "Inserted %d memory document(s) for app=%s, user=%s.",
            len(documents),
            app_name,
            user_id,
        )

    async def _upsert_session_document(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        stored_text: str,
        embedding: list[float],
        event_ids: list[str],
        source: str,
        custom_metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Upsert the single memory document for a tenant/session key."""
        now = datetime.now(timezone.utc)
        key = {"app_name": app_name, "user_id": user_id, "session_id": session_id}
        set_fields: dict[str, Any] = {
            **key,
            self.transcript_field: stored_text,
            self.embedding_field: embedding,
            "event_ids": event_ids,
            "author": self.memory_author,
            "source": source,
            "updated_at": now,
        }
        if custom_metadata:
            set_fields["custom_metadata"] = dict(custom_metadata)
        await self.collection.update_one(
            key,
            {"$set": set_fields, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Semantic search over a tenant's memories via Atlas Vector Search.

        Matches the ADK base signature exactly (keyword-only, no ``limit``) so the
        framework's ``load_memory`` / ``PreloadMemoryTool`` call it correctly. The
        result count is controlled by ``default_search_limit``. The query is
        embedded with ``input_type="query"`` so it lands in the same space as the
        stored ``"document"`` vectors.

        Args:
            app_name: Tenant application identifier (pre-filter).
            user_id: Tenant user identifier (pre-filter).
            query: Natural-language search query.

        Returns:
            A :class:`SearchMemoryResponse` whose ``memories`` are ordered by
            descending similarity. Empty if the query is blank or nothing matches.

        Raises:
            RuntimeError: If the ``$vectorSearch`` aggregation fails (most often
                because the vector index does not exist yet or is still building).
        """
        response = SearchMemoryResponse()
        if not query or not query.strip():
            logger.debug("search_memory called with empty query; returning no results.")
            return response

        query_vector = await self._embed(query, QUERY_INPUT)
        limit = self.default_search_limit
        num_candidates = max(limit, limit * self.num_candidates_multiplier)

        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": self.embedding_field,
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit,
                    # Tenant isolation: only ever search this app + user.
                    "filter": {
                        "$and": [
                            {"app_name": {"$eq": app_name}},
                            {"user_id": {"$eq": user_id}},
                        ]
                    },
                }
            },
            # Surface the similarity score, then drop the bulky vector from results.
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
            {"$project": {self.embedding_field: 0}},
        ]

        try:
            cursor = await self.collection.aggregate(pipeline)
            docs = await cursor.to_list(length=limit)
        except OperationFailure as exc:
            raise RuntimeError(
                f"Atlas Vector Search query failed on index {self.vector_index_name!r}. "
                "Confirm the vector search index exists, is READY, and that "
                "'app_name'/'user_id' are declared as filter fields. "
                f"Underlying error: {exc}"
            ) from exc

        for doc in docs:
            entry = self._document_to_memory_entry(doc, app_name=app_name, user_id=user_id)
            if entry is not None:
                response.memories.append(entry)

        logger.info(
            "search_memory(app=%s, user=%s) returned %d result(s).",
            app_name,
            user_id,
            len(response.memories),
        )
        return response

    def _document_to_memory_entry(
        self, doc: Mapping[str, Any], *, app_name: str, user_id: str
    ) -> Optional[MemoryEntry]:
        """Convert a stored document into an ADK :class:`MemoryEntry`.

        Reads fields defensively and skips documents with no usable text.
        """
        text = doc.get(self.transcript_field)
        if not text:
            logger.debug("Skipping document %s with empty transcript.", doc.get("_id"))
            return None

        metadata: dict[str, Any] = {
            "session_id": doc.get("session_id"),
            "source": doc.get("source"),
        }
        if "_score" in doc:
            metadata["score"] = doc["_score"]
        stored_metadata = doc.get("custom_metadata")
        if isinstance(stored_metadata, Mapping):
            metadata.update(stored_metadata)

        return MemoryEntry(
            content=Content(role="model", parts=[Part(text=str(text))]),
            author=doc.get("author") or self.memory_author,
            # ADK expects an ISO-8601 *string* timestamp (not a datetime object).
            timestamp=self._to_iso(doc.get("updated_at")),
            custom_metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    async def _embed(self, text: str, input_type: str) -> list[float]:
        """Embed ``text`` using ``embedding_fn`` with the given ``input_type``.

        ``input_type`` is ``"document"`` for stored memories and ``"query"`` for
        searches; it is forwarded to ``embedding_fn`` (sync or async).
        """
        vector = await self._maybe_await(self.embedding_fn, text, input_type)
        vector_list = list(vector)
        if not vector_list:
            raise ValueError("embedding_fn returned an empty vector.")
        if len(vector_list) != self.embedding_dimensions:
            # Not fatal (some callers vary), but almost always a misconfiguration.
            logger.warning(
                "Embedding length %d != configured embedding_dimensions %d; "
                "this will not match the vector index and searches may fail.",
                len(vector_list),
                self.embedding_dimensions,
            )
        return vector_list

    async def _distill(self, transcript: str) -> str:
        """Apply ``distill_fn`` if configured; otherwise return the transcript."""
        if self.distill_fn is None:
            return transcript
        distilled = await self._maybe_await(self.distill_fn, transcript)
        if not distilled or not str(distilled).strip():
            logger.warning("distill_fn returned empty output; storing raw transcript instead.")
            return transcript
        return str(distilled)

    @staticmethod
    async def _maybe_await(fn: Callable[..., Any], *args: Any) -> Any:
        """Invoke ``fn`` correctly whether it is async or sync.

        Async callables - a coroutine function *or* a callable object whose
        ``__call__`` is ``async def`` (e.g. the Voyage embedder wrapping
        ``voyageai.AsyncClient``) - are awaited on the running event loop so the
        underlying client shares this loop and is never used across threads.
        Genuinely synchronous callables are off-loaded with
        :func:`asyncio.to_thread` so a blocking embedding/HTTP call cannot stall
        the loop.
        """
        if MongoAtlasMemoryService._is_async_callable(fn):
            return await fn(*args)
        result = await asyncio.to_thread(fn, *args)
        # Defensive: a sync wrapper might still hand back an awaitable.
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _is_async_callable(fn: Callable[..., Any]) -> bool:
        """Return ``True`` if calling ``fn`` yields a coroutine to await.

        :func:`inspect.iscoroutinefunction` returns ``False`` for a *callable
        object* whose ``__call__`` is ``async def`` (only the bound method is the
        coroutine function), so we inspect ``__call__`` too. ``functools.partial``
        wrappers are unwrapped first.
        """
        target = fn
        while isinstance(target, functools.partial):
            target = target.func
        if inspect.iscoroutinefunction(target):
            return True
        # A callable *object* (e.g. the Voyage embedder) carries its coroutine on
        # the class-level __call__; instance-level iscoroutinefunction misses it.
        # (We read __call__ directly to inspect its async-ness, not via
        # hasattr/callable - those only tell us *that* it is callable.)
        call = type(target).__call__ if callable(target) else None
        return inspect.iscoroutinefunction(call)

    @staticmethod
    def _segments_from_events(events: Iterable[Event]) -> list[tuple[str, str, str]]:
        """Extract ``(event_id, author, text)`` tuples from events with text content."""
        segments: list[tuple[str, str, str]] = []
        for event in events or []:
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            text = "".join(part.text for part in parts if getattr(part, "text", None))
            if not text.strip():
                continue
            author = getattr(event, "author", None) or "unknown"
            event_id = getattr(event, "id", None) or ""
            segments.append((event_id, author, text.strip()))
        return segments

    @staticmethod
    def _format_segments(segments: Sequence[tuple[str, str, str]]) -> str:
        """Render ``(event_id, author, text)`` tuples into an authored transcript."""
        return "\n".join(f"{author}: {text}" for _id, author, text in segments)

    @staticmethod
    def _extract_entry_text(entry: MemoryEntry) -> str:
        """Join all text parts of a :class:`MemoryEntry`'s content."""
        content = getattr(entry, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return ""
        return "".join(part.text for part in parts if getattr(part, "text", None)).strip()

    @staticmethod
    def _to_iso(value: Any) -> Optional[str]:
        """Coerce a stored timestamp into an ISO-8601 string (ADK's expected type)."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
