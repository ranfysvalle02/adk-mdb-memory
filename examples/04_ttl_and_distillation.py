"""Example 04 - TTL compliance index + transcript distillation.

Two production-flavoured features of :class:`MongoAtlasMemoryService`, shown
together:

* **TTL retention** - constructing the service with ``ttl_seconds=...`` makes
  ``setup_indexes`` create a MongoDB TTL index on ``updated_at``. MongoDB then
  auto-purges memory documents that many seconds after their last update - zero
  cron jobs, handy for GDPR/HIPAA-style retention.
* **Distillation** - passing a ``distill_fn`` lets you compress / fact-extract a
  raw transcript *before* it is embedded and stored. Here we use the bundled
  ``trivial_distiller`` (keep user-authored lines only) so you can see the hook;
  in production you'd route the transcript through a fast model. Distillation
  shrinks what gets vectorized - it does not change that embeddings are real
  Voyage vectors.

Requires ``VOYAGE_API_KEY`` for embeddings. Run from anywhere::

    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local   # or set MONGODB_URI
    python examples/04_ttl_and_distillation.py
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
    trivial_distiller,
    wait_until_searchable,
)

DB_NAME = "adk_memory_examples"
COLLECTION = "ttl_distillation"
APP_NAME = "ttl_demo_app"
USER_ID = "dana"
TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


def build_session():
    """A multi-turn session whose transcript has both user and model lines."""
    from google.adk.events import Event
    from google.adk.sessions import Session
    from google.genai.types import Content, Part

    turns = [
        ("user", "I'm planning a trip to Japan in April."),
        ("assistant", "How exciting - that's cherry blossom season!"),
        ("user", "I want to visit Kyoto and Osaka, and I absolutely love ramen."),
        ("assistant", "Noted: Kyoto, Osaka, and plenty of ramen."),
    ]
    events = [
        Event(
            author=author,
            content=Content(
                role="user" if author == "user" else "model", parts=[Part(text=text)]
            ),
        )
        for author, text in turns
    ]
    return Session(id="japan_trip", app_name=APP_NAME, user_id=USER_ID, state={}, events=events)


async def main() -> None:
    configure_logging()
    cfg = Config()

    banner("Example 04 - TTL + distillation (Voyage embeddings)")
    if not require_voyage_key(cfg):
        return
    print_connection_banner(cfg, db_name=DB_NAME, collection=COLLECTION)

    # Construct WITH a TTL and a distill function. make_service forwards extras.
    async with make_service(
        cfg,
        db_name=DB_NAME,
        collection_name=COLLECTION,
        ttl_seconds=TTL_SECONDS,
        distill_fn=trivial_distiller,
    ) as service:
        banner("Step 1: setup_indexes also creates the TTL index")
        await service.setup_indexes(wait_for_vector_index=True, timeout_seconds=180.0)

        # Show the TTL index that was created on updated_at.
        index_info = await service.collection.index_information()
        ttl_indexes = {
            name: spec for name, spec in index_info.items() if "expireAfterSeconds" in spec
        }
        print("TTL index(es) on the collection:")
        for name, spec in ttl_indexes.items():
            print(f"  - {name}: expireAfterSeconds={spec['expireAfterSeconds']} on {spec['key']}")
        if not ttl_indexes:
            print("  (none found - unexpected)")

        banner("Step 2: store a multi-turn transcript (it gets distilled first)")
        session = build_session()
        raw_transcript = "\n".join(
            f"{e.author}: {''.join(p.text for p in e.content.parts if p.text)}"
            for e in session.events
        )
        print("Raw transcript that was sent in:")
        for line in raw_transcript.splitlines():
            print(f"    {line}")
        await service.add_session_to_memory(session)

        banner("Step 3: inspect what was actually stored")
        doc = await service.collection.find_one(
            {"app_name": APP_NAME, "user_id": USER_ID, "session_id": "japan_trip"}
        )
        stored_text = doc.get(service.transcript_field, "") if doc else ""
        print("Distilled text stored in the memory document (user lines only):")
        for line in stored_text.splitlines():
            print(f"    {line}")
        print(
            f"\nDistillation kept {len(stored_text)} of {len(raw_transcript)} characters "
            f"({len(stored_text) / max(1, len(raw_transcript)):.0%} of the original)."
        )

        banner("Step 4: the distilled memory is still searchable")
        query = "Where am I traveling and what food do I like?"
        await wait_until_searchable(service, app_name=APP_NAME, user_id=USER_ID, query=query)
        response = await service.search_memory(app_name=APP_NAME, user_id=USER_ID, query=query)
        print(f"Query: {query!r} -> {len(response.memories)} result(s)")
        for entry in response.memories:
            text = " ".join(p.text for p in (entry.content.parts or []) if p.text)
            score = entry.custom_metadata.get("score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"    (score={score_str}) {text}")

    banner("Done")


if __name__ == "__main__":
    asyncio.run(main())
