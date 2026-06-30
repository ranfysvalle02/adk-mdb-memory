# adk-mongodb-memory

A light, production-oriented wrapper that implements Google ADK's
`BaseMemoryService` on top of **MongoDB Atlas Vector Search**, with **Voyage AI**
embeddings.

---

# Why We Chose a Light Wrapper Over a Pull Request for Google ADK Memory

## TL;DR — Executive Level Summary

* **The Problem:** Google’s Agent Development Kit (ADK) features a robust `MemoryService` abstraction, but its out-of-the-box storage backends are limited to volatile, in-memory caches or native Google Cloud services (Vertex AI). For teams standardizing their operational and semantic data on MongoDB Atlas, a connector is missing.
* **The Decision:** Instead of opening an upstream Pull Request (PR) to integrate MongoDB directly into core ADK, we engineered a clean, localized **Light Wrapper**.
* **Why Avoid the PR?** It prevents structural dependency bloat inside core packages (forcing `pymongo` drivers on developers who don't need them), bypasses slow open-source governance timelines, and avoids rigid, lowest-common-denominator schemas that choke enterprise data partitioning.
* **The Technical Edge:** Writing a local wrapper allows us to tap directly into the **native asynchronous features of PyMongo (v4.13+)**, ensuring zero blocking on the asyncio event loop while integrating custom, domain-specific features like cryptographic tenant isolation, LLM session distillation, and compliance TTLs.

---

## Why a Light Wrapper Beats an Upstream PR (The Strategic Breakdown)

When deciding whether to submit code back to the main repository or shield it inside a local platform utility layer, three major architectural factors tip the scale in favor of the **Light Wrapper**:

### 1. Core Dependency Discipline

Open-source maintainers are highly conservative regarding their dependency footprint. If a PR introduces massive third-party client drivers directly into the core `google-adk` package, it forces weight on teams running lightweight, edge-based configurations. It would likely face extensive scrutiny or be wrapped in optional, clunky "extras" syntax. A local wrapper keeps your system dependencies modular and lightweight.

### 2. Timeline Decoupling & Engineering Velocity

Enterprise product roadmaps cannot stall while waiting for open-source project maintainers to debate, refactor, review, and merge a custom database adapter. Building a localized extension allows your platform team to ship production features *today*, completely isolated from external release schedules.

### 3. Exploiting cutting-edge PyMongo Async Features Natively

As an engineer, you can move faster than an upstream framework can update. PyMongo introduced fully native asynchronous capabilities (`AsyncMongoClient`) that eliminate the need for legacy third-party async wrappers or thread-pool delegations for database I/O. Authoring a local wrapper gives you the structural freedom to implement clean, asynchronous, awaitable database calls that blend directly with ADK’s modern asyncio architecture.

---

## High-Value Customization Patterns (Tailoring the Brain)

By choosing the wrapper path, you gain the freedom to embed advanced enterprise capabilities that would never be merged into a generic, general-purpose upstream PR:

* **LLM-Driven Distillation (Memory Compression):** Raw chat logs are noisy and exhaust vector context windows quickly. In your custom wrapper's `add_session_to_memory` interceptor, you can route the raw session stream through a fast `gemini-2.5-flash` context loop to extract *atomic facts and user preferences*. Vectorizing and storing these distilled facts cuts document sizing dramatically and boosts downstream semantic recall. (Plug it in via `distill_fn`; see example 04.)
* **Cryptographic Tenant Isolation:** In multi-tenant SaaS environments, securing customer records purely via an MQL query filter is often insufficient. A light wrapper lets you wrap sensitive fields in Client-Side Field-Level Encryption (CSFLE) using keys managed in a real KMS — encrypting the PII *side fields* while still embedding/searching the non-sensitive text. (See example 05.)
* **Compliance via Native Time-To-Live (TTL):** Data privacy frameworks (such as GDPR or HIPAA) require automatic data deletion policies. By ensuring a native MongoDB TTL index on an `updated_at` field within your wrapper initialization routine, MongoDB handles automated record purging implicitly, requiring no external cron jobs or scripts.

---

# Practical Guide: Setup, Usage & Atlas Vector Search

> The sections above explain *why* this is a light wrapper. The sections below are
> the *how* — everything you need to install it, wire it into your agent, and run
> it against a real MongoDB Atlas cluster.

## What's in this repo

| Path | Purpose |
| --- | --- |
| `adk_mongodb_memory/` | The installable Python package. |
| `adk_mongodb_memory/service.py` | `MongoAtlasMemoryService` — a `BaseMemoryService` backed by MongoDB Atlas Vector Search. |
| `adk_mongodb_memory/embeddings.py` | `VoyageEmbedder` — the input-type-aware Voyage embedding callable, plus `build_voyage_embedder()`. |
| `adk_mongodb_memory/__init__.py` | Re-exports the public API (`MongoAtlasMemoryService`, `VoyageEmbedder`, types). |
| `pyproject.toml` | PEP 621 packaging, dependencies, and the optional `encryption` extra (CSFLE). |
| `examples/` | Focused, runnable examples (incl. CSFLE). See [`examples/README.md`](examples/README.md). |
| `examples/_shared.py` | Example-only glue: `Config`, URI resolution, the Voyage embedder factory, helpers. |
| `examples/01_quickstart.py` | Connect → index → store → semantic search with scores. |
| `examples/02_agent_load_memory.py` | Full ADK flow with `LlmAgent` + the built-in `load_memory` tool (needs a Gemini key). |
| `examples/03_multi_tenant_isolation.py` | Proves per-tenant isolation: searches never cross `(app_name, user_id)`. |
| `examples/04_ttl_and_distillation.py` | TTL compliance index + transcript distillation hook. |
| `examples/05_csfle.py` | Atlas Vector Search + Client-Side Field-Level Encryption: search the safe text, decrypt the PII that rode along. |
| `.env.example` | Every environment variable the examples read, documented. |
| `requirements.txt` | Installs the project editable with the `encryption` extra. |

## Requirements

* **Python 3.10+**
* **A MongoDB endpoint with Vector Search**, either:
  * **Local (recommended for dev):** the **MongoDB Atlas Local** Docker image
    (`mongodb/mongodb-atlas-local`). It bundles `mongod` + `mongot`, so
    `$vectorSearch` works on your laptop with **no cloud account**:

    ```bash
    docker run -d -p 27017:27017 --name adk-mongo mongodb/mongodb-atlas-local
    ```

    The code auto-targets it at `mongodb://localhost:27017/?directConnection=true`
    when `MONGODB_URI` is unset. Remove it later with `docker rm -f adk-mongo`.
  * **Cloud:** a **MongoDB Atlas** cluster on a tier that supports Atlas Vector
    Search — **M10+ dedicated or a Flex cluster**. (Free `M0` tiers do **not**
    support `$vectorSearch`.) Set `MONGODB_URI` to its SRV string.
* **A Voyage AI API key** (`VOYAGE_API_KEY`) for embeddings — **required**. Get
  one from the [Voyage dashboard](https://dash.voyageai.com/api-keys) (free tier
  available), or, as a MongoDB Atlas customer, from the Atlas UI under *AI Models*.
* A **Google AI (Gemini) API key** *only* for the agent example (`02`). Get one at
  [Google AI Studio](https://aistudio.google.com/apikey).

## Setup

```bash
# 1. Create an isolated environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip

# 2. Install the package (editable) with the CSFLE extra (libmongocrypt)
pip install -e ".[encryption]"     # or: pip install -r requirements.txt

# 3. Configure your keys
cp .env.example .env
#   ...edit .env: set VOYAGE_API_KEY (required) and, for example 02, a Gemini key.
#   Leave MONGODB_URI unset to use the local Atlas container, or set it for cloud.

# 4. Start a local Atlas (skip if you set MONGODB_URI to a cloud cluster)
docker run -d -p 27017:27017 --name adk-mongo mongodb/mongodb-atlas-local
```

The connection string resolves simply: **`MONGODB_URI` if set, otherwise the
local default** `mongodb://localhost:27017/?directConnection=true`. The core
library has **no** localhost defaults — it always requires an explicit
`connection_string` (or an injected `client`); that convenience lives only in the
examples' `_shared.py`.

## Run the examples

```bash
python examples/01_quickstart.py            # store + semantic search (scores)
python examples/03_multi_tenant_isolation.py
python examples/04_ttl_and_distillation.py
python examples/05_csfle.py                 # Atlas Vector Search + field-level encryption of PII

# Needs a Gemini key in addition to VOYAGE_API_KEY:
python examples/02_agent_load_memory.py
```

On first run the service **creates the Atlas Vector Search index for you** and
waits for it to become queryable (a few seconds on local Atlas; up to ~a minute
on a cold cloud index). See **[`examples/README.md`](examples/README.md)** for a
guided tour and expected output.

## Wiring it into your own agent

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory

from adk_mongodb_memory import MongoAtlasMemoryService, VoyageEmbedder

# 1. A Voyage embedder. It is input_type-aware and async; it reads VOYAGE_API_KEY
#    from the environment. voyage-3.5 emits 1024-dim vectors by default.
embedder = VoyageEmbedder(model="voyage-3.5", output_dimension=1024)

# 2. The memory service
memory_service = MongoAtlasMemoryService(
    connection_string="mongodb+srv://...",
    db_name="adk_memory",
    embedding_fn=embedder,
    embedding_dimensions=1024,
)
await memory_service.setup_indexes(wait_for_vector_index=True)  # idempotent

# 3. Give an agent the load_memory tool and pass the service to the Runner
agent = LlmAgent(
    model="gemini-2.5-flash",
    name="MemoryRecallAgent",
    instruction="Use the 'load_memory' tool to recall facts from past chats.",
    tools=[load_memory],
)
runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=InMemorySessionService(),
    memory_service=memory_service,   # <-- MongoDB Atlas memory
)
```

### The embedder contract (`input_type`-aware)

`embedding_fn` is called as `embedding_fn(text, input_type)`, where `input_type`
is `"document"` for stored memories and `"query"` for searches. Voyage uses this
to tailor the vector to its role, which materially improves retrieval — MongoDB's
own guidance is to **not** omit `input_type` for search. The service threads it
through automatically.

Bringing your own embedder? Accept the two positional arguments; you can simply
ignore the second if your model doesn't distinguish roles:

```python
async def my_embedder(text: str, input_type: str) -> list[float]:
    # input_type is "document" or "query"; ignore it if you don't need it.
    return await my_model.embed(text)
```

Sync callables work too — the service off-loads them to a worker thread with
`asyncio.to_thread` so they never block the event loop. Async callables (a
coroutine function *or* a callable object with `async def __call__`, like
`VoyageEmbedder`) are awaited directly on the running loop.

> **Voyage client lifecycle:** by default the Voyage SDK opens a short-lived HTTP
> session per request. For a long-lived, high-throughput service, construct the
> embedder with `VoyageEmbedder(reuse_connections=True)` to pool connections
> across requests, and `await embedder.aclose()` on shutdown.

### Auto-saving sessions to memory

Persist a completed conversation either explicitly
(`await service.add_session_to_memory(session)`) or, more cleanly, via an
`after_agent_callback` (this is what example 02 does):

```python
from google.adk.agents.callback_context import CallbackContext

async def auto_save_to_memory(callback_context: CallbackContext) -> None:
    await callback_context.add_session_to_memory()

agent = LlmAgent(
    model="gemini-2.5-flash",
    name="QA_Agent",
    instruction="Answer the user's questions.",
    tools=[load_memory],                       # or PreloadMemoryTool()
    after_agent_callback=auto_save_to_memory,
)
```

`PreloadMemoryTool` retrieves memory automatically at the start of every turn;
`load_memory` lets the model decide when to look things up. Both call this
service's `search_memory()` under the hood.

## Atlas Vector Search index

`setup_indexes()` creates this index **programmatically** via pymongo's
`create_search_index` + `SearchIndexModel` (idempotent — it ignores
"already exists"). It declares the embedding field plus `app_name`/`user_id` as
**filter** fields, which is required for the tenant pre-filter in `search_memory`.

If you'd rather create it by hand (Atlas UI → *Atlas Search* → *Create Index* →
*JSON Editor*, index type **Vector Search**), use this definition:

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine" },
    { "type": "filter", "path": "app_name" },
    { "type": "filter", "path": "user_id" }
  ]
}
```

Name it `vector_index` (or match `MONGODB_VECTOR_INDEX_NAME`). Keep
`numDimensions` in sync with your embedding model + `output_dimension`. Voyage
float embeddings are unit-normalized, so keep `similarity: "cosine"` and do not
normalize again.

### How retrieval works

`search_memory` runs an approximate-nearest-neighbour `$vectorSearch`:

```text
numCandidates = default_search_limit * num_candidates_multiplier   # default: 5 * 20 = 100
```

A larger `numCandidates` improves recall at the cost of latency. MongoDB
recommends at least **20×** the `limit` as a starting point (and higher when a
selective pre-filter like the per-tenant `app_name`/`user_id` filter is applied).
Results are ordered by similarity; the score is returned in each entry's
`custom_metadata["score"]`.

## Configuration reference

Constructor parameters (all keyword-configurable, sensible defaults):

| Parameter | Default | Description |
| --- | --- | --- |
| `connection_string` | – | MongoDB URI (or pass your own `client`). |
| `db_name` | `"adk_memory"` | Database name. |
| `embedding_fn` | **required** | `(text, input_type) -> vector`, sync or async. |
| `collection_name` | `"agent_memories"` | Collection for memory docs. |
| `vector_index_name` | `"vector_index"` | Atlas Vector Search index name. |
| `embedding_field` | `"embedding"` | Field storing the vector. |
| `transcript_field` | `"transcript"` | Field storing the (distilled) text. |
| `embedding_dimensions` | `1024` | Must match model `output_dimension` + index. |
| `similarity` | `"cosine"` | `cosine` / `euclidean` / `dotProduct`. |
| `default_search_limit` | `5` | Results per search (ADK never passes a limit). |
| `num_candidates_multiplier` | `20` | `numCandidates = limit * this` (MongoDB recommends ≥20×). |
| `ttl_seconds` | `None` | Enables a TTL purge index when set. |
| `distill_fn` | `None` | Optional transcript-compression hook (sync/async). |
| `memory_author` | `"memory"` | `author` recorded on stored memories. |
| `client` | `None` | Inject an `AsyncMongoClient` (the CSFLE extension point). |

Environment variables consumed by the examples are documented in `.env.example`.

> **Ecosystem interop:** the defaults already mirror `langchain-mongodb`'s
> `MongoDBAtlasVectorSearch` (`embedding` key, `vector_index` name, cosine). If
> you want a LangChain retriever to read the same collection out of the box, also
> set `transcript_field="text"` to match its default `text_key`.

## Enterprise features at a glance

* **Async-correct** — `AsyncMongoClient`, every DB call awaited. Async embedders (coroutine functions *and* callable objects with `async __call__`) are awaited on the loop; sync embedders are off-loaded with `asyncio.to_thread`. No event-loop blocking.
* **Retrieval-tuned embeddings** — Voyage with `input_type="document"` on writes and `input_type="query"` on searches.
* **Tenant isolation** — every search is pre-filtered to one `app_name` + `user_id`; those are indexed filter fields.
* **Index management** — programmatic compound lookup index, vector search index, and an optional **TTL** compliance index.
* **Memory distillation hook** — supply `distill_fn` to compress transcripts (fact extraction) before embedding/storage.
* **Lifecycle** — `await service.close()` or use `async with MongoAtlasMemoryService(...) as service:`.
* **Extra writes** — implements the optional `add_events_to_memory` (incremental, deduped by event id) and `add_memory` (direct `MemoryEntry` inserts) beyond the required `add_session_to_memory` / `search_memory`.

### Extension point: CSFLE (Client-Side Field-Level Encryption)

CSFLE is configured on the Mongo **client**, not per-operation. Rather than ship a
fragile partial implementation, this wrapper lets you inject a fully-configured
client for transparent **automatic** encryption:

```python
from pymongo import AsyncMongoClient
from pymongo.encryption_options import AutoEncryptionOpts

client = AsyncMongoClient(uri, auto_encryption_opts=AutoEncryptionOpts(...))
memory_service = MongoAtlasMemoryService(client=client, db_name="adk_memory", embedding_fn=embedder)
```

(An injected client is owned by you — the service won't close it. Automatic
encryption needs the `crypt_shared` library or `mongocryptd`.)

For **explicit** encryption that needs only `pip install 'pymongo[encryption]'`
(libmongocrypt — no `crypt_shared`/`mongocryptd` to deploy), see
**[`examples/05_csfle.py`](examples/05_csfle.py)**: it encrypts a PII side field
(an SSN in `custom_metadata`) while embedding only the non-sensitive text, then
**finds the memory with Atlas Vector Search over that safe text** — the encrypted
SSN rides back on the result as ciphertext and is decrypted client-side. It's the
proof that Atlas Vector Search and field-level encryption compose cleanly.

> **Never hardcode encryption keys in production.** The example uses a hardcoded
> `local` master key for convenience only. Real deployments must source the key
> from a KMS — **AWS KMS, Azure Key Vault, GCP KMS, or KMIP**.

## Caveats / "go time" checklist

* **Where `$vectorSearch` runs:** the **MongoDB Atlas Local** Docker image
  supports Vector Search locally for development (it ships `mongot`). In the
  cloud, you need an **M10+ or Flex** cluster — a free `M0` tier will error on
  index creation.
* **`VOYAGE_API_KEY` is required** — there is no offline embedder fallback anywhere.
  The examples print friendly guidance and exit cleanly if it is missing.
* Vector Search is **eventually consistent** — there's a short lag between a write and when it's searchable. The examples poll `search_memory` to handle this; the index also takes time to build the first time.
* Keep `embedding_dimensions` aligned across your **model** (`output_dimension`), the **stored vectors**, and the **index** — a mismatch makes searches fail.

## License

MIT — see [`LICENSE`](LICENSE).
