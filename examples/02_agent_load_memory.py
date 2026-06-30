"""Example 02 - Agent recall via the built-in ``load_memory`` tool.

The full Google ADK flow, end to end:

    Turn 1  An ``LlmAgent`` (InfoCaptureAgent) chats with the user. An
            ``after_agent_callback`` then *auto-saves* the completed session to
            long-term memory - the enterprise pattern (no manual bookkeeping).
    Turn 2  In a brand-new session, a second ``LlmAgent`` (MemoryRecallAgent)
            armed with the built-in ``load_memory`` tool answers a question by
            semantically searching that stored memory - which calls our
            ``MongoAtlasMemoryService.search_memory`` under the hood.

This example needs two keys:

* ``VOYAGE_API_KEY``  - embeddings for storing/searching memory (always).
* ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` - the Gemini chat model for the
  agents. (ADK also supports many other models, e.g. via LiteLLM - swap the
  ``model=`` string.)

Missing keys produce clear guidance and a clean exit (no stack trace). Run::

    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local   # or set MONGODB_URI
    python examples/02_agent_load_memory.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from _shared import (
    Config,
    banner,
    configure_logging,
    make_service,
    print_connection_banner,
    require_voyage_key,
    wait_until_searchable,
)

logger = logging.getLogger("adk_mongodb_memory.examples.02")

DB_NAME = "adk_memory_examples"
COLLECTION = "agent_load_memory"
APP_NAME = "agent_memory_app"
USER_ID = "casey"


def _scrub(text: str, secret: Optional[str]) -> str:
    """Strip a secret (the API key) from a message before printing, just in case."""
    return text.replace(secret, "***REDACTED***") if secret and secret in text else text


async def run_turn(runner, *, user_id: str, session_id: str, text: str) -> str:
    """Send one user message through a Runner and return the final text reply."""
    from google.genai.types import Content, Part

    final = "(no response)"
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(role="user", parts=[Part(text=text)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text
    return final


async def main() -> None:
    configure_logging()
    cfg = Config()

    banner("Example 02 - Agent recall via load_memory (Voyage + Gemini)")

    # Embeddings are non-negotiable; the chat model is also required here.
    if not require_voyage_key(cfg):
        return
    if not cfg.google_api_key:
        print(
            "VOYAGE_API_KEY is set, but no GOOGLE_API_KEY / GEMINI_API_KEY was found, "
            "so the live LLM agents can't run.\n\n"
            "  1. Get a key at https://aistudio.google.com/apikey\n"
            '  2. Add it to .env:  GOOGLE_API_KEY="..."\n'
            "  3. Re-run this example.\n\n"
            "Prefer no LLM? Examples 01, 03, 04 and 05 exercise the memory service "
            "directly with only VOYAGE_API_KEY."
        )
        return

    print_connection_banner(cfg, db_name=DB_NAME, collection=COLLECTION)

    from google.adk.agents import LlmAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import load_memory

    async def auto_save_to_memory(callback_context: CallbackContext) -> None:
        """Enterprise pattern: persist the finished session to long-term memory.

        Wiring this as an ``after_agent_callback`` means every completed
        conversation is captured automatically (it calls the same
        ``add_session_to_memory`` the Runner's ``memory_service`` exposes) - no
        manual save calls scattered through your app.
        """
        await callback_context.add_session_to_memory()

    async with make_service(
        cfg,
        db_name=DB_NAME,
        collection_name=COLLECTION,
    ) as service:
        banner("Bootstrapping indexes")
        try:
            await service.setup_indexes(wait_for_vector_index=True, timeout_seconds=180.0)
        except Exception as exc:  # noqa: BLE001 - friendly guidance, not a trace
            print(f"\nIndex setup failed: {_scrub(str(exc), cfg.voyage_api_key)}")
            print(
                "Is a MongoDB with Vector Search reachable? Start one with:\n"
                "    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local\n"
                "or set MONGODB_URI to an Atlas cluster (M10+/Flex)."
            )
            return

        session_service = InMemorySessionService()

        try:
            # ---- Turn 1: capture a fact; the callback auto-saves it --------- #
            banner("Turn 1: capture a preference (InfoCaptureAgent + auto-save)")
            capture_agent = LlmAgent(
                model=cfg.gemini_model,
                name="InfoCaptureAgent",
                instruction="Acknowledge the user's statement in one short, friendly sentence.",
                after_agent_callback=auto_save_to_memory,
            )
            capture_runner = Runner(
                agent=capture_agent,
                app_name=APP_NAME,
                session_service=session_service,
                memory_service=service,
            )
            await session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id="capture"
            )
            fact = "Please remember I'm allergic to peanuts and I prefer window seats when I fly."
            print(f"User: {fact}")
            reply = await run_turn(capture_runner, user_id=USER_ID, session_id="capture", text=fact)
            print(f"InfoCaptureAgent: {reply}")
            print("(the after_agent_callback saved this session to MongoDB automatically)")

            print("Waiting for the memory to become searchable...")
            await wait_until_searchable(
                service, app_name=APP_NAME, user_id=USER_ID, query="dietary allergy seat preference"
            )

            # ---- Turn 2: recall it in a brand-new session ------------------ #
            banner("Turn 2: recall in a NEW session (MemoryRecallAgent + load_memory)")
            recall_agent = LlmAgent(
                model=cfg.gemini_model,
                name="MemoryRecallAgent",
                instruction=(
                    "Answer the user's question. Use the 'load_memory' tool to look "
                    "up facts from past conversations whenever it might help."
                ),
                # `load_memory` lets the model decide *when* to search memory.
                # For always-on retrieval, swap in PreloadMemoryTool() instead:
                #   from google.adk.tools.preload_memory_tool import PreloadMemoryTool
                #   tools=[PreloadMemoryTool()]
                # Both call MongoAtlasMemoryService.search_memory under the hood.
                tools=[load_memory],
            )
            recall_runner = Runner(
                agent=recall_agent,
                app_name=APP_NAME,
                session_service=session_service,
                memory_service=service,
            )
            await session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id="recall"
            )
            question = (
                "I'm booking a flight and ordering a snack box - "
                "anything you should flag for me?"
            )
            print(f"User: {question}")
            answer = await run_turn(
                recall_runner, user_id=USER_ID, session_id="recall", text=question
            )
            print(f"MemoryRecallAgent: {answer}")
            print(
                "\nExpected: the agent recalls the peanut allergy and window-seat "
                "preference it loaded from Atlas memory."
            )
        except Exception as exc:  # noqa: BLE001 - network/quota/model errors
            print(f"\nLLM run failed: {type(exc).__name__}: {_scrub(str(exc), cfg.google_api_key)}")
            print("Check your API key, model name, network, and quota, then re-run.")
            return

    banner("Done")


if __name__ == "__main__":
    asyncio.run(main())
