"""Example 01 - Quickstart.

The "hello world" for :class:`MongoAtlasMemoryService`. It:

1. Resolves the MongoDB connection string (``MONGODB_URI`` if set, otherwise a
   local Atlas Docker container).
2. Creates the supporting + Atlas Vector Search indexes and waits until the
   vector index is queryable.
3. Stores a couple of hand-built sessions as long-term memories. Each is
   embedded with Voyage using ``input_type="document"`` (handled inside the
   service).
4. Runs a semantic ``search_memory`` (the query is embedded with
   ``input_type="query"``) and prints the hits with similarity scores.

Requires ``VOYAGE_API_KEY`` for embeddings (no offline fallback). Run it from
anywhere once dependencies are installed (``pip install -e .``)::

    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local   # or set MONGODB_URI
    python examples/01_quickstart.py

Re-running is safe: memories are upserted per (app_name, user_id, session_id).
"""

from __future__ import annotations

import asyncio

from _shared import (
    Config,
    banner,
    configure_logging,
    make_service,
    print_connection_banner,
    require_voyage_key,
    wait_until_searchable,
)

# Keep this example's data isolated from the others.
DB_NAME = "adk_memory_examples"
COLLECTION = "quickstart"
APP_NAME = "quickstart_app"
USER_ID = "alice"


def build_session(session_id: str, user_text: str, model_text: str):
    """Build a minimal completed ADK :class:`Session` with one exchange."""
    from google.adk.events import Event
    from google.adk.sessions import Session
    from google.genai.types import Content, Part

    events = [
        Event(author="user", content=Content(role="user", parts=[Part(text=user_text)])),
        Event(author="assistant", content=Content(role="model", parts=[Part(text=model_text)])),
    ]
    return Session(id=session_id, app_name=APP_NAME, user_id=USER_ID, state={}, events=events)


async def main() -> None:
    configure_logging()
    cfg = Config()

    banner("Example 01 - Quickstart (Voyage embeddings)")
    if not require_voyage_key(cfg):
        return
    print_connection_banner(cfg, db_name=DB_NAME, collection=COLLECTION)

    async with make_service(cfg, db_name=DB_NAME, collection_name=COLLECTION) as service:
        banner("Step 1: create indexes (compound lookup + vector search)")
        print("Submitting the Atlas Vector Search index and waiting for it to build...")
        await service.setup_indexes(wait_for_vector_index=True, timeout_seconds=180.0)
        print("Indexes ready.")

        banner("Step 2: store a couple of memories (embedded as documents)")
        memories = [
            build_session(
                "fav_language",
                "My favorite programming language is Rust and I build Project Alpha in it.",
                "Got it - Rust on Project Alpha.",
            ),
            build_session(
                "morning_routine",
                "I usually start my day with an oat milk latte before standup.",
                "Noted your oat milk latte habit.",
            ),
        ]
        for session in memories:
            await service.add_session_to_memory(session)
            print(f"  stored session '{session.id}'")

        banner("Step 3: semantic search (query embedded as a query)")
        query = "Which coding language do I prefer?"
        print(f"Query: {query!r}")
        # Vector Search is eventually consistent - poll until both writes land so
        # the ranking is stable (falls through gracefully if only one indexes).
        await wait_until_searchable(
            service, app_name=APP_NAME, user_id=USER_ID, query=query, min_results=len(memories)
        )
        response = await service.search_memory(app_name=APP_NAME, user_id=USER_ID, query=query)

        print(f"\nTop {len(response.memories)} result(s) by similarity:")
        for i, entry in enumerate(response.memories, start=1):
            text = " ".join(p.text for p in (entry.content.parts or []) if p.text)
            score = entry.custom_metadata.get("score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"  {i}. (score={score_str}) {text}")
        if not response.memories:
            print("  (nothing found yet - on a cold index, wait a moment and re-run)")
        elif len(response.memories) >= 2:
            print(
                "\nNote how the Rust/Project Alpha memory ranks above the latte one: "
                "the query is semantically closer to it."
            )

    banner("Done")


if __name__ == "__main__":
    asyncio.run(main())
