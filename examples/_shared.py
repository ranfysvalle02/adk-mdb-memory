"""Shared helpers for the runnable examples under ``examples/``.

This is the single source of example-only glue. It deliberately lives *outside*
the ``adk_mongodb_memory`` package so the library itself stays free of localhost
defaults and example-only conveniences.

It provides:

* A tiny ``sys.path`` bootstrap so ``import adk_mongodb_memory`` works when you
  run an example straight from the repo without installing anything
  (``python examples/01_quickstart.py``). ``pip install -e .`` also works.
* :class:`Config` - a typed view over the environment variables the examples
  read (loaded from ``.env`` via python-dotenv).
* Simplified MongoDB URI resolution: ``MONGODB_URI`` if set, otherwise a local
  Atlas default, plus :func:`redact_uri` for safe logging.
* A Voyage embedder factory bound to :class:`Config`, the offline-friendly
  :func:`trivial_distiller`, and presentation helpers (:func:`banner`,
  :func:`configure_logging`, :func:`wait_until_searchable`).
* :func:`require_voyage_key` - prints friendly guidance and exits cleanly when
  ``VOYAGE_API_KEY`` is missing (no raw tracebacks).
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import re
import sys
from typing import TYPE_CHECKING, Optional

# --- Make the repo root importable before importing the project package. ----- #
# (Running `python examples/01_quickstart.py` only puts `examples/` on the path,
# not the repo root. `pip install -e .` makes this unnecessary but harmless.)
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402  (import after the sys.path tweak)

from adk_mongodb_memory import (  # noqa: E402
    DEFAULT_VOYAGE_DIMENSIONS,
    DEFAULT_VOYAGE_MODEL,
    MongoAtlasMemoryService,
    VoyageEmbedder,
    build_voyage_embedder,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

logger = logging.getLogger("adk_mongodb_memory.examples")

# A MongoDB Atlas Local container started with the docker one-liner:
#   docker run -d -p 27017:27017 mongodb/mongodb-atlas-local
# ``directConnection=true`` matters: the Atlas Local image is a single-node
# replica set, and the flag tells the driver to connect directly instead of
# trying to reach the node by its (container-internal) replica-set hostname.
# The Atlas Local image bundles mongod + mongot, so $vectorSearch works locally.
LOCAL_DEFAULT_URI = "mongodb://localhost:27017/?directConnection=true"

# Where to get a Voyage API key (shown in the friendly missing-key guidance).
VOYAGE_KEYS_URL = "https://dash.voyageai.com/api-keys"


def redact_uri(uri: str) -> str:
    """Mask any ``user:password@`` credentials in a connection string for logs."""
    return re.sub(r"://([^:@/]+):([^@]+)@", r"://\1:****@", uri)


class Config:
    """Typed view over the environment variables the examples read.

    Loads ``.env`` (if present) on construction. ``mongodb_uri`` is always
    populated: it uses ``MONGODB_URI`` when set, otherwise the local Atlas
    default (:data:`LOCAL_DEFAULT_URI`); ``mongodb_uri_is_default`` records which.
    """

    def __init__(self) -> None:
        load_dotenv()  # reads a local .env file if present; no-op otherwise.

        explicit_uri = (os.getenv("MONGODB_URI") or "").strip()
        self.mongodb_uri: str = explicit_uri or LOCAL_DEFAULT_URI
        self.mongodb_uri_is_default: bool = not explicit_uri

        self.db_name: str = os.getenv("MONGODB_DB_NAME", "adk_memory")
        self.collection: str = os.getenv("MONGODB_COLLECTION_NAME", "agent_memories")
        self.vector_index: str = os.getenv("MONGODB_VECTOR_INDEX_NAME", "vector_index")

        # Voyage embeddings (the only embedder this project uses).
        self.voyage_api_key: Optional[str] = os.getenv("VOYAGE_API_KEY")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", DEFAULT_VOYAGE_MODEL)
        self.embedding_dimensions: int = int(
            os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_VOYAGE_DIMENSIONS))
        )

        # Gemini chat model - only example 02 needs this (for the LlmAgent).
        self.google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY") or os.getenv(
            "GEMINI_API_KEY"
        )
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        ttl = os.getenv("MEMORY_TTL_SECONDS")
        self.ttl_seconds: Optional[int] = int(ttl) if ttl else None
        self.distill: bool = os.getenv("MEMORY_DISTILL", "").lower() in {"1", "true", "yes"}

        self.app_name: str = os.getenv("APP_NAME", "memory_demo_app")
        self.user_id: str = os.getenv("USER_ID", "demo_user")


def require_voyage_key(cfg: Config) -> bool:
    """Return ``True`` if a Voyage key is configured; otherwise print guidance.

    Every example needs real Voyage embeddings - there is intentionally no
    offline embedding fallback. When the key is missing we print where to get one
    and return ``False`` so the caller can exit cleanly (no stack trace).
    """
    if cfg.voyage_api_key:
        return True
    print(
        "VOYAGE_API_KEY is not set, so embeddings can't be created.\n\n"
        f"  1. Get a key from the Voyage dashboard:  {VOYAGE_KEYS_URL}\n"
        "     (MongoDB Atlas customers can instead create one in the Atlas UI\n"
        "      under 'AI Models'. The voyageai client auto-routes either key.)\n"
        '  2. Add it to .env:  VOYAGE_API_KEY="pa-..."\n'
        "  3. Re-run this example.\n\n"
        "Voyage offers a free tier, so you can try this end-to-end at no cost."
    )
    return False


def make_embedder(cfg: Config, *, reuse_connections: bool = False) -> VoyageEmbedder:
    """Build a :class:`VoyageEmbedder` from :class:`Config`.

    Uses ``cfg.embedding_model`` and ``cfg.embedding_dimensions``, reading the
    key from ``VOYAGE_API_KEY`` (call :func:`require_voyage_key` first).
    """
    return build_voyage_embedder(
        api_key=cfg.voyage_api_key,
        model=cfg.embedding_model,
        output_dimension=cfg.embedding_dimensions,
        reuse_connections=reuse_connections,
    )


def make_service(
    cfg: Config,
    *,
    embedder: Optional["Callable[[str, str], object]"] = None,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
    **kwargs: object,
) -> MongoAtlasMemoryService:
    """Build a :class:`MongoAtlasMemoryService` from a resolved :class:`Config`.

    Defaults to a Voyage embedder built from ``cfg``. Pass your own ``embedder``
    to override. Extra keyword arguments (``ttl_seconds``, ``distill_fn``, ...)
    are forwarded to the service constructor.
    """
    dims = embedding_dimensions if embedding_dimensions is not None else cfg.embedding_dimensions
    if embedder is None:
        embedder = make_embedder(cfg)
    return MongoAtlasMemoryService(
        connection_string=cfg.mongodb_uri,
        db_name=db_name if db_name is not None else cfg.db_name,
        embedding_fn=embedder,
        collection_name=collection_name if collection_name is not None else cfg.collection,
        vector_index_name=cfg.vector_index,
        embedding_dimensions=dims,
        **kwargs,
    )


def trivial_distiller(transcript: str) -> str:
    """A toy 'distillation' hook: keep only user-authored lines, capped.

    Distillation compresses a raw transcript *before* it is embedded and stored
    (memory compression / fact extraction). This offline placeholder just shows
    where the hook plugs in; a real deployment would route the transcript through
    a fast model (e.g. ``gemini-2.5-flash``) to extract atomic facts. Enable it
    in the examples with ``MEMORY_DISTILL=true`` (or example 04 forces it on).
    """
    user_lines = [ln for ln in transcript.splitlines() if ln.lower().startswith("user:")]
    distilled = "\n".join(user_lines) if user_lines else transcript
    return distilled[:2000]


def configure_logging(level: int = logging.INFO) -> None:
    """Configure friendly, readable logging for the examples.

    Safe to call more than once (:func:`logging.basicConfig` only acts on the
    first call). Noisy third-party loggers are quieted so the narrative printed
    by each example stays readable.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("pymongo", "google_genai", "httpx", "httpcore", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def banner(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_connection_banner(cfg: Config, *, db_name: str, collection: str) -> None:
    """Print the resolved MongoDB target (and a local-dev hint when defaulted)."""
    banner("Connection")
    print(f"MongoDB URI    : {redact_uri(cfg.mongodb_uri)}")
    print(f"Database       : {db_name}")
    print(f"Collection     : {collection}")
    print(f"Embedding model: {cfg.embedding_model} ({cfg.embedding_dimensions} dims, Voyage)")
    if cfg.mongodb_uri_is_default:
        print(
            "\nNo MONGODB_URI set - targeting a local Atlas container. Start one with:\n"
            "    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local\n"
            "or set MONGODB_URI to a cloud Atlas cluster (M10+/Flex) that supports "
            "Vector Search."
        )


async def wait_until_searchable(
    service: MongoAtlasMemoryService,
    *,
    app_name: str,
    user_id: str,
    query: str,
    min_results: int = 1,
    timeout_seconds: float = 60.0,
    interval_seconds: float = 3.0,
) -> bool:
    """Poll ``search_memory`` until recently-written memories become retrievable.

    Atlas Vector Search is eventually consistent: there is a short lag between an
    insert and when the new document becomes searchable (and a longer one while
    the index first builds). Polling keeps the examples deterministic instead of
    racing the indexer. Returns ``True`` once at least ``min_results`` memories
    appear, else ``False`` after ``timeout_seconds``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            response = await service.search_memory(
                app_name=app_name, user_id=user_id, query=query
            )
            if len(response.memories) >= min_results:
                logger.info("Memory is searchable after %d attempt(s).", attempt)
                return True
        except RuntimeError as exc:
            # Index may still be building; surface at debug level, keep polling.
            logger.debug("search not ready yet: %s", exc)
        if loop.time() >= deadline:
            logger.warning(
                "Fewer than %d result(s) searchable within %.0fs; continuing anyway.",
                min_results,
                timeout_seconds,
            )
            return False
        await asyncio.sleep(interval_seconds)
