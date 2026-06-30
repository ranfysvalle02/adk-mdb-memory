"""Voyage AI embedding helper for :class:`MongoAtlasMemoryService`.

This module provides :class:`VoyageEmbedder`, an ``input_type``-aware embedding
callable backed by `Voyage AI <https://docs.voyageai.com/docs/embeddings>`_'s
asynchronous client. It satisfies the service's :data:`EmbeddingFn` contract:
it is invoked as ``embedder(text, input_type)`` and returns a dense vector.

Why ``input_type`` matters
--------------------------
Voyage (like most modern retrieval embedders) produces better recall when you
tell it whether the text is a stored *document* or a search *query*: it prepends
a small role-specific instruction before vectorizing. The service passes
``"document"`` on every write path and ``"query"`` from ``search_memory``, so
recall is tuned automatically. MongoDB's own Voyage guidance is explicit: do not
omit ``input_type`` for retrieval.

Why ``voyage-3.5`` / 1024 dims by default
-----------------------------------------
``voyage-3.5`` is a strong, cost-effective default that emits 1024-dim vectors
(it also supports 256 / 512 / 1024 / 2048 via ``output_dimension``). The current
top-end family - ``voyage-4-large`` (best quality), ``voyage-4`` (balanced) and
``voyage-4-lite`` (cheapest/fastest), plus domain models such as
``voyage-code-3`` - share the same dimension options, so switching model rarely
requires re-defining the vector index. Voyage float embeddings are
unit-normalized, so keep ``similarity="cosine"`` and do **not** normalize again.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import aiohttp

logger = logging.getLogger(__name__)

#: Sensible, cost-effective default model (1024-dim, multilingual, retrieval-tuned).
DEFAULT_VOYAGE_MODEL = "voyage-3.5"
#: Default output dimensionality (also the ``voyage-3.5`` / ``voyage-4`` default).
DEFAULT_VOYAGE_DIMENSIONS = 1024


class VoyageEmbedder:
    """Async, ``input_type``-aware embedder backed by ``voyageai.AsyncClient``.

    Instances are *callable objects* with an ``async def __call__``; the memory
    service awaits them directly on the running event loop (it never off-loads
    them to a thread), so the underlying async HTTP client stays bound to one
    loop.

    Args:
        api_key:
            Voyage API key. When ``None`` (default) the underlying client reads
            ``VOYAGE_API_KEY`` from the environment.
        model:
            Voyage embedding model. Defaults to :data:`DEFAULT_VOYAGE_MODEL`.
        output_dimension:
            Target vector dimensionality (256 / 512 / 1024 / 2048 on supported
            models). Must match the service's ``embedding_dimensions`` and the
            Atlas Vector Search index. Defaults to
            :data:`DEFAULT_VOYAGE_DIMENSIONS`. Pass ``None`` to use the model's
            native default.
        truncation:
            Forwarded to Voyage. ``None`` (default) uses Voyage's default
            behaviour (over-long inputs are truncated to the model's context).
        reuse_connections:
            For long-lived, high-throughput services. By default the Voyage SDK
            opens a short-lived ``aiohttp`` session *per request*. Set this to
            ``True`` to have the embedder create **one** shared
            ``aiohttp.ClientSession`` (registered on ``voyageai.aiosession``) so
            HTTP connections are pooled and reused; it is closed by
            :meth:`aclose`. Leave ``False`` for short scripts/examples.
        client:
            An existing ``voyageai.AsyncClient`` to reuse instead of
            constructing one (advanced).

    Lifecycle:
        Call :meth:`aclose` (or use the instance as an async context manager) on
        shutdown. It only has work to do when ``reuse_connections=True``; in the
        default per-request mode it is a safe no-op.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_VOYAGE_MODEL,
        output_dimension: Optional[int] = DEFAULT_VOYAGE_DIMENSIONS,
        truncation: Optional[bool] = None,
        reuse_connections: bool = False,
        client: Optional[object] = None,
    ) -> None:
        # Imported lazily so the core package does not hard-require voyageai
        # unless the Voyage embedder is actually used.
        import voyageai

        self.model = model
        self.output_dimension = output_dimension
        self.truncation = truncation
        self._client = client if client is not None else voyageai.AsyncClient(api_key=api_key)
        self._reuse_connections = reuse_connections
        self._session: Optional["aiohttp.ClientSession"] = None

    async def __call__(self, text: str, input_type: str) -> list[float]:
        """Embed a single ``text`` with the given ``input_type``.

        ``input_type`` is ``"document"`` or ``"query"`` (passed straight through
        to Voyage). Returns the embedding as a ``list[float]``.
        """
        if self._reuse_connections and self._session is None:
            await self._open_shared_session()
        result = await self._client.embed(
            [text],
            model=self.model,
            input_type=input_type,
            output_dimension=self.output_dimension,
            truncation=self.truncation,
        )
        return list(result.embeddings[0])

    async def _open_shared_session(self) -> None:
        """Create and register a process-wide aiohttp session for connection reuse."""
        import aiohttp
        import voyageai

        self._session = aiohttp.ClientSession()
        # voyageai.aiosession is a ContextVar; when set, the SDK reuses this
        # session for every request instead of remaking one each time.
        voyageai.aiosession.set(self._session)
        logger.debug("VoyageEmbedder opened a shared aiohttp session for connection reuse.")

    async def aclose(self) -> None:
        """Close the shared aiohttp session, if one was opened. Safe to call always."""
        if self._session is not None:
            import voyageai

            voyageai.aiosession.set(None)
            await self._session.close()
            self._session = None
            logger.debug("VoyageEmbedder closed its shared aiohttp session.")

    async def __aenter__(self) -> "VoyageEmbedder":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def build_voyage_embedder(
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_VOYAGE_MODEL,
    output_dimension: Optional[int] = DEFAULT_VOYAGE_DIMENSIONS,
    truncation: Optional[bool] = None,
    reuse_connections: bool = False,
) -> VoyageEmbedder:
    """Construct a :class:`VoyageEmbedder` (convenience factory).

    Reads ``VOYAGE_API_KEY`` from the environment when ``api_key`` is ``None``.
    The returned object is an ``input_type``-aware async embedding callable ready
    to pass as ``embedding_fn`` to :class:`MongoAtlasMemoryService`.
    """
    return VoyageEmbedder(
        api_key=api_key,
        model=model,
        output_dimension=output_dimension,
        truncation=truncation,
        reuse_connections=reuse_connections,
    )
