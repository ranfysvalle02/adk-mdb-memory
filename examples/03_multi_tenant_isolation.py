"""Example 03 - Multi-tenant isolation.

Every read and write in :class:`MongoAtlasMemoryService` is scoped to an
``(app_name, user_id)`` tenant, and ``search_memory`` enforces that with an
index-backed ``filter`` inside the ``$vectorSearch`` stage. This example proves
it: two different users store *near-identical* content (so vector similarity
alone would cross them), and each user's search only ever returns their own
memory - never the other tenant's.

Requires ``VOYAGE_API_KEY`` for embeddings. Run from anywhere::

    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local   # or set MONGODB_URI
    python examples/03_multi_tenant_isolation.py
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

DB_NAME = "adk_memory_examples"
COLLECTION = "multi_tenant"
APP_NAME = "support_bot"  # same app for both users -> isolation is purely per-user.

# Two tenants under the same app, with deliberately similar content so that
# vector similarity *alone* would cross them - only the tenant filter keeps
# them apart.
TENANTS = {
    "alice": "My account PIN hint is my dog's name, Rex.",
    "bob": "My account PIN hint is my cat's name, Mittens.",
}


def build_session(user_id: str, text: str):
    from google.adk.events import Event
    from google.adk.sessions import Session
    from google.genai.types import Content, Part

    events = [Event(author="user", content=Content(role="user", parts=[Part(text=text)]))]
    return Session(id="hint", app_name=APP_NAME, user_id=user_id, state={}, events=events)


async def main() -> None:
    configure_logging()
    cfg = Config()

    banner("Example 03 - Multi-tenant isolation (Voyage embeddings)")
    if not require_voyage_key(cfg):
        return
    print_connection_banner(cfg, db_name=DB_NAME, collection=COLLECTION)

    async with make_service(cfg, db_name=DB_NAME, collection_name=COLLECTION) as service:
        banner("Setup: indexes + per-tenant writes")
        await service.setup_indexes(wait_for_vector_index=True, timeout_seconds=180.0)
        for user_id, text in TENANTS.items():
            await service.add_session_to_memory(build_session(user_id, text))
            print(f"  stored memory for user_id={user_id!r}: {text}")

        # Wait until both writes are searchable (poll using one tenant).
        await wait_until_searchable(
            service, app_name=APP_NAME, user_id="alice", query="what is my PIN hint?"
        )

        banner("Each tenant searches the SAME query")
        query = "What is my PIN hint?"
        leaked = False
        for user_id, own_text in TENANTS.items():
            response = await service.search_memory(app_name=APP_NAME, user_id=user_id, query=query)
            texts = [
                " ".join(p.text for p in (e.content.parts or []) if p.text)
                for e in response.memories
            ]
            print(f"\nuser_id={user_id!r} -> {len(texts)} result(s):")
            for t in texts:
                print(f"    - {t}")

            # Isolation check: results must contain ONLY this tenant's content.
            others = {uid: txt for uid, txt in TENANTS.items() if uid != user_id}
            for other_user, other_text in others.items():
                if any(other_text in t for t in texts):
                    leaked = True
                    print(f"    !! LEAK: saw {other_user!r}'s memory in {user_id!r}'s results")
            if not any(own_text in t for t in texts):
                print(f"    (note: {user_id!r} did not see their own memory yet - index lag?)")

        banner("Result")
        if leaked:
            print("FAIL: cross-tenant data leaked. (This should never happen.)")
        else:
            print(
                "PASS: every search returned only the calling tenant's memory.\n"
                "The $vectorSearch 'filter' on app_name + user_id guarantees isolation."
            )

    banner("Done")


if __name__ == "__main__":
    asyncio.run(main())
