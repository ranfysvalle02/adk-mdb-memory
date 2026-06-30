# Production Agent Memory for Google ADK, Backed by MongoDB Atlas Vector Search

*How a thin, drop-in `BaseMemoryService` turns Google ADK's memory **contract** into a
memory **system** you actually own — with semantic recall, hard multi-tenant
isolation, field-level encryption, compliance TTLs, and memory distillation that
the in-the-box backends don't give you.*

---

## TL;DR

* Google's **Agent Development Kit (ADK)** ships a clean memory *abstraction*
  (`BaseMemoryService`, wired to the `load_memory` / `PreloadMemoryTool` tools),
  but the backends that come with it are either a **volatile, keyword-matching
  dev stub** (`InMemoryMemoryService`) or a **managed, cloud-locked black box**
  (Vertex AI Memory Bank / RAG).
* This repo — [`adk-mongodb-memory`](./README.md) — implements that same
  abstraction on top of **MongoDB Atlas Vector Search**, so memory lives in a
  database **you** run, right next to your operational data.
* It's a **drop-in**: subclass `BaseMemoryService`, match the keyword-only
  `search_memory` signature, pass it to the `Runner`. Nothing else in your agent
  changes.
* The value-add is everything a generic managed service won't let you do:
  **input-type-aware embeddings**, **index-backed per-tenant isolation**,
  **CSFLE for PII side fields**, **native TTL retention**, a
  **transcript-distillation hook**, **async-correct I/O**, and **ecosystem
  interop** with the rest of MongoDB.
* All five runnable examples pass end-to-end against a local **MongoDB Atlas
  Local** container — no cloud account required.

---

## ADK gives you a memory *contract*, not a memory *system*

ADK's memory layer is small and tasteful. The entire surface an agent cares
about is one abstract base class:

```python
class BaseMemoryService(ABC):
    @abstractmethod
    async def add_session_to_memory(self, session: Session) -> None: ...

    @abstractmethod
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse: ...

    # Optional — raise NotImplementedError by default:
    async def add_events_to_memory(self, *, app_name, user_id, events, ...): ...
    async def add_memory(self, *, app_name, user_id, memories, ...): ...
```

When you give a `Runner` a `memory_service`, the built-in `load_memory` tool (the
model decides *when* to recall) and `PreloadMemoryTool` (always-on recall at the
top of a turn) both call `search_memory()` under the hood. That's the contract.
It says nothing about *where* memory lives or *how* recall works.

So what do you get in the box?

| Backend | Storage | Recall | Reality |
| --- | --- | --- | --- |
| `InMemoryMemoryService` | a Python `dict` in process | **keyword matching** (`any(query_word in words_in_event)`) | volatile; lost on restart; explicitly *"for testing and development only"* |
| `VertexAiMemoryBankService` | Google Cloud (Vertex) | managed semantic memory | real, but **GCP-locked**; you don't own the store, the embeddings, the residency, or the retention policy |
| `VertexAiRagMemoryService` | Vertex RAG corpora | managed RAG | same trade-off: convenient, opaque, coupled to one cloud |

The dev stub isn't even semantic — it splits text into words and looks for
overlap. The managed services *are* semantic, but they're a second system to
operate, on someone else's terms, separate from the database your application
already trusts.

**The gap:** teams that already standardize on MongoDB Atlas for operational
*and* semantic data have nowhere to put agent memory without bolting on another
managed dependency.

---

## The thesis: a thin wrapper you own beats a managed black box

`MongoAtlasMemoryService` is a faithful `BaseMemoryService` implementation backed
by Atlas Vector Search. It's deliberately a **light wrapper** — it leans on
native PyMongo async features instead of third-party glue — but that thinness is
exactly what lets it expose enterprise capabilities a lowest-common-denominator
upstream backend never could.

The rest of this post is the value-add, capability by capability: *what ADK gives
you out of the box, what the wrapper adds, and why it matters.* Every claim is
backed by a runnable example in [`examples/`](./examples/).

---

## 1. It's a genuine drop-in (exact contract compliance)

The whole point of an abstraction is substitutability. The service subclasses
`BaseMemoryService` and matches the framework's **keyword-only** `search_memory`
signature exactly, so the built-in tools wire up unchanged:

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory

from adk_mongodb_memory import MongoAtlasMemoryService, VoyageEmbedder

memory_service = MongoAtlasMemoryService(
    connection_string="mongodb+srv://...",
    embedding_fn=VoyageEmbedder(model="voyage-3.5", output_dimension=1024),
    embedding_dimensions=1024,
)
await memory_service.setup_indexes(wait_for_vector_index=True)  # idempotent

agent = LlmAgent(model="gemini-2.5-flash", name="Recall", tools=[load_memory])
runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=InMemorySessionService(),
    memory_service=memory_service,   # <-- the only line that changes
)
```

Swapping `InMemoryMemoryService()` for `MongoAtlasMemoryService(...)` is the
entire migration. `load_memory`, `PreloadMemoryTool`, and the
`callback_context.add_session_to_memory()` auto-save pattern all keep working.

**Out of the box:** the contract. **Added:** a backend you can actually ship,
without touching agent code.

---

## 2. One database for operational data *and* agent memory

This is the headline. With the managed paths, agent memory is a **separate
system** — a second SLA, a second bill, a second data-residency story, a second
access-control surface. With this wrapper, memories are just documents in *your*
Atlas cluster:

```json
{
  "app_name": "support_bot",
  "user_id": "casey",
  "session_id": "capture",
  "transcript": "user: I'm allergic to peanuts and prefer window seats ...",
  "embedding": [0.0123, -0.0456, ...],   // 1024 dims
  "event_ids": ["e1", "e2"],
  "author": "memory",
  "created_at": "2026-06-30T18:43:54Z",
  "updated_at": "2026-06-30T18:43:54Z"
}
```

Because it's your collection, every MongoDB capability applies to agent memory
for free: aggregation pipelines, Atlas Search, role-based access control,
backups, change streams, Charts, the BI connector, and joins against the
operational data your agents are reasoning about. Memory stops being a silo.

**Out of the box:** memory lives wherever the managed service puts it. **Added:**
memory lives where the rest of your data already does.

---

## 3. Persistent, *semantic* recall (not volatile keyword overlap)

`search_memory` runs an approximate-nearest-neighbour `$vectorSearch`, pre-filtered
to the calling tenant:

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": self.vector_index_name,
            "path": self.embedding_field,
            "queryVector": query_vector,
            "numCandidates": num_candidates,   # default_search_limit * 20
            "limit": limit,
            "filter": {"$and": [
                {"app_name": {"$eq": app_name}},
                {"user_id":  {"$eq": user_id}},
            ]},
        }
    },
    {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
    {"$project": {self.embedding_field: 0}},   # don't ship the vector back
]
```

The difference from the dev stub is night and day. The quickstart stores two
memories and asks *"Which coding language do I prefer?"* — a query that shares no
keywords with the stored text:

```text
Top 2 result(s) by similarity:
  1. (score=0.7773) user: My favorite programming language is Rust ...
  2. (score=0.6657) user: I usually start my day with an oat milk latte ...
```

Semantic ranking puts the Rust memory on top; a keyword matcher would have
returned nothing. And unlike the in-memory dict, it survives a restart — it's in
the database.

**Out of the box:** keyword overlap that evaporates on restart. **Added:**
durable, ranked, approximate-nearest-neighbour semantic recall.

---

## 4. Retrieval-tuned, swappable embeddings (`input_type`-aware)

A managed memory service hands you whatever embedding model it chose. Here, the
embedder is a first-class, swappable contract — and it's **`input_type`-aware**,
which is a real recall win that's easy to miss.

The embedder is called as `embedding_fn(text, input_type)`, where `input_type` is
`"document"` for stored memories and `"query"` for searches. Modern retrieval
embedders (Voyage among them) prepend a small role-specific instruction so the
query and the document land in compatible regions of vector space. The service
threads this through automatically:

```python
# on every write path:
embedding = await self._embed(stored_text, DOCUMENT_INPUT)  # "document"
# in search_memory:
query_vector = await self._embed(query, QUERY_INPUT)        # "query"
```

The bundled `VoyageEmbedder` defaults to `voyage-3.5` at 1024 dims and supports
256 / 512 / 1024 / 2048 across the `voyage-4` family and `voyage-code-3` — so you
can dial quality vs. cost without re-defining the index. Bringing your own model
is two positional arguments:

```python
async def my_embedder(text: str, input_type: str) -> list[float]:
    return await my_model.embed(text)   # ignore input_type if your model doesn't care
```

**Out of the box:** the provider's embedding choices. **Added:** your model, your
dimensions, with retrieval-intent baked into every call.

---

## 5. Hard multi-tenant isolation, enforced by the index

In multi-tenant SaaS, "we filter by user in application code" is how data leaks
happen. Here every read and write is scoped to an `(app_name, user_id)` tenant,
and the scope is enforced *inside* the `$vectorSearch` stage as an **index-backed
filter** — `app_name` and `user_id` are declared as `filter` fields in the vector
index definition, so the boundary is part of the query plan, not a post-filter.

Example 03 makes two users store deliberately near-identical content (so vector
similarity *alone* would cross them) and then run the same query:

```text
user_id='alice' -> 1 result(s):
    - user: My account PIN hint is my dog's name, Rex.
user_id='bob' -> 1 result(s):
    - user: My account PIN hint is my cat's name, Mittens.

PASS: every search returned only the calling tenant's memory.
```

Neither tenant can see the other's memory, even though their embeddings are
nearly identical. The filter, not luck, keeps them apart.

**Out of the box:** scope semantics vary by backend and aren't yours to harden.
**Added:** a provable, index-level tenant boundary on every search.

---

## 6. Cryptographic isolation with CSFLE (encrypt PII, still search)

Agent memory accumulates PII — SSNs, PINs, health details. A query filter doesn't
protect that at rest; encryption does. Because CSFLE is configured on the Mongo
**client**, the wrapper exposes a clean extension point: inject your own
`AsyncMongoClient` (the service won't close a client it didn't create), or use
**explicit** encryption, which needs only `pip install 'pymongo[encryption]'`
(libmongocrypt — no `crypt_shared` library or `mongocryptd` process to deploy) and
runs against your Atlas cluster as-is.

The realistic pattern, shown in example 05: **embed the non-sensitive text, encrypt
the sensitive side field you never run `$vectorSearch` on** — and the two compose
cleanly on Atlas.

```python
ssn_ciphertext = await client_encryption.encrypt(
    "123-45-6789",
    algorithm=Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Deterministic,
    key_id=data_key_id,
)
await service.add_memory(
    app_name="secure_support_bot", user_id="erin",
    memories=[MemoryEntry(content=Content(parts=[Part(text=PROFILE_TEXT)]))],
    custom_metadata={"ssn": ssn_ciphertext},   # rides along, never embedded
)
```

Then find it with **Atlas Vector Search over the safe text** — the encrypted SSN
rides back on the result as ciphertext, decryptable only client-side:

```text
Matched by similarity (score~0.81): 'Customer profile: enterprise tier; ...'
PII rode along as BSON Binary subtype 6 (still encrypted): True
Decrypted SSN     : 123-45-6789
Round-trip OK     : True
```

You must **not** encrypt the field you embed (the stored vector and the query
vector have to be computed over the same readable text), so the design encrypts a
*side field* and embeds a safe summary. Vector search still retrieves the memory by
that safe text; the PII just travels alongside, encrypted. Keys come from a real
KMS — AWS KMS, Azure Key Vault, GCP KMS, or KMIP — in production.

**Out of the box:** nothing like this. **Added:** field-level encryption with
keys you control, without breaking semantic search.

---

## 7. Compliance retention with native TTL (zero cron jobs)

GDPR/HIPAA-style "delete this after N days" usually means a scheduled job and the
operational risk that comes with it. MongoDB has retention built in. Construct the
service with `ttl_seconds`, and `setup_indexes()` creates a TTL index on
`updated_at`; the database's background monitor purges expired memories for you:

```python
service = MongoAtlasMemoryService(..., ttl_seconds=60 * 60 * 24 * 30)  # 30 days
await service.setup_indexes(...)
```

```text
TTL index(es) on the collection:
  - idx_ttl_updated_at: expireAfterSeconds=2592000 on [('updated_at', 1)]
```

No cron, no Lambda, no sweeper script — retention is a property of the collection.

**Out of the box:** retention is the managed service's policy, not yours.
**Added:** declarative, database-enforced expiry you configure per deployment.

---

## 8. A memory-distillation hook (compress before you embed)

Raw transcripts are noisy and burn vector-context budget. The wrapper accepts a
`distill_fn` that runs *before* embedding and storage — the natural place to route
a transcript through a fast model (e.g. `gemini-2.5-flash`) and extract atomic
facts. Example 04 uses a trivial "keep user lines only" distiller to show the hook:

```text
Raw transcript (221 chars):
    user: I'm planning a trip to Japan in April.
    assistant: How exciting - that's cherry blossom season!
    user: I want to visit Kyoto and Osaka, and I absolutely love ramen.
    assistant: Noted: Kyoto, Osaka, and plenty of ramen.

Distilled (112 chars, 51% of original) — user lines only, still fully searchable:
    user: I'm planning a trip to Japan in April.
    user: I want to visit Kyoto and Osaka, and I absolutely love ramen.

Query "Where am I traveling and what food do I like?" -> (score=0.7827)
```

Distillation halves what gets vectorized while preserving recall. That's a
domain-specific transform you'd never get to inject into a generic backend.

**Out of the box:** store whatever the framework captured. **Added:** a pre-embed
transform under your control for compression and fact extraction.

---

## 9. Async-correct by construction (no event-loop blocking)

ADK is asyncio-native, so a memory backend that blocks the loop quietly caps your
throughput. This wrapper uses PyMongo's native `AsyncMongoClient` and `await`s
every database call. Embedders are handled correctly whether they're sync or
async:

```python
@staticmethod
async def _maybe_await(fn, *args):
    if MongoAtlasMemoryService._is_async_callable(fn):
        return await fn(*args)                 # awaited on the running loop
    result = await asyncio.to_thread(fn, *args)  # sync work off the loop
    ...
```

Async callables — including *callable objects* whose `__call__` is `async def`,
like `VoyageEmbedder` wrapping `voyageai.AsyncClient` — are awaited directly so the
underlying HTTP client stays bound to one loop. Genuinely synchronous embedders
are off-loaded with `asyncio.to_thread` so a blocking HTTP call never stalls the
event loop. Detecting the callable-object case correctly (inspecting
`type(fn).__call__`, not just `iscoroutinefunction(fn)`) is the kind of sharp edge
a thin, owned wrapper can get exactly right.

**Out of the box:** opaque I/O behavior. **Added:** verified non-blocking I/O that
blends with ADK's runtime.

---

## 10. Operational knobs you actually control

Managed services hide the dials. Here they're yours:

* **Programmatic index management.** `setup_indexes()` creates the compound lookup
  index, the Atlas Vector Search index (via `create_search_index` +
  `SearchIndexModel`), and the optional TTL index — idempotently (it ignores
  "already exists"). It can even wait until the vector index reports queryable.
* **Recall vs. latency.** `numCandidates = default_search_limit *
  num_candidates_multiplier` (default `5 * 20 = 100`), following MongoDB's "≥20×
  the limit" guidance — higher when a selective tenant pre-filter is applied.
* **Similarity metric.** `cosine` (default), `euclidean`, or `dotProduct`.
* **Field names.** `embedding_field`, `transcript_field`, `vector_index_name`,
  `collection_name`, `db_name` — all configurable.

**Out of the box:** the provider's defaults. **Added:** the ANN, indexing, and
schema knobs that let you tune for your workload.

---

## 11. Ecosystem interop (your memory isn't a silo)

The defaults intentionally mirror `langchain-mongodb`'s `MongoDBAtlasVectorSearch`
— the `embedding` field name, the `vector_index` index name, cosine similarity —
so a LangChain retriever can read the *same* collection out of the box (set
`transcript_field="text"` to match its `text_key`). Beyond LangChain, the same
documents are reachable from aggregation pipelines, Atlas Search, Charts, and the
BI connector. Agent memory becomes a first-class citizen of your data platform
rather than an island reachable only through one SDK.

**Out of the box:** access through the framework's API only. **Added:** the entire
MongoDB toolchain, pointed at the same collection.

---

## 12. Beyond the required surface

The base class only *requires* `add_session_to_memory` and `search_memory`; the
other two writers raise `NotImplementedError` by default. This wrapper implements
all four:

* `add_session_to_memory(session)` — join the transcript, optionally distill,
  embed as `"document"`, upsert one doc per `(app_name, user_id, session_id)`.
* `add_events_to_memory(...)` — **incremental**, deduplicated by `event.id`;
  appends a delta and re-embeds the combined transcript.
* `add_memory(...)` — insert pre-built `MemoryEntry` items directly, merging
  `custom_metadata` (the hook example 05 uses to carry CSFLE ciphertext).
* `search_memory(...)` — the tenant-filtered `$vectorSearch` above.

**Out of the box:** two methods, two `NotImplementedError`s. **Added:** the full
writer surface, including incremental and direct-insert paths.

---

## The big picture

| Capability | `InMemoryMemoryService` | Vertex AI Memory Bank / RAG | **`MongoAtlasMemoryService`** |
| --- | --- | --- | --- |
| Persistence | ❌ in-process, volatile | ✅ managed | ✅ your Atlas cluster |
| Recall quality | keyword overlap | semantic (managed) | **semantic ANN (`$vectorSearch`)** |
| You own the data | n/a | ❌ cloud-locked | ✅ co-located with operational data |
| Embedding choice | n/a | provider's | **any model; `input_type`-aware** |
| Tenant isolation | per-backend | per-backend | **index-backed `$vectorSearch` filter** |
| PII encryption (CSFLE) | ❌ | ❌ | ✅ encrypt side fields, embed safe text |
| Retention / TTL | ❌ | provider policy | ✅ native TTL index, zero cron |
| Distillation hook | ❌ | ❌ | ✅ `distill_fn` before embed |
| Async-correct I/O | n/a | opaque | ✅ `AsyncMongoClient`, no loop blocking |
| Ecosystem interop | ❌ | ❌ | ✅ LangChain / aggregation / Charts |
| ADK drop-in | ✅ (dev only) | ✅ | ✅ exact contract |

---

## How it's built: a light wrapper, on purpose

We chose a **local wrapper** over an upstream PR into core ADK — not because the
PR couldn't be written, but because the wrapper is the better engineering
decision:

* **Dependency discipline.** Core frameworks are conservative about their
  footprint; nobody wants `pymongo` forced on every ADK user. A wrapper keeps the
  driver where it belongs — in the projects that need it.
* **Velocity.** Shipping a platform feature shouldn't wait on open-source review
  cycles. A local extension ships today.
* **Native PyMongo async.** PyMongo 4.13+ exposes `AsyncMongoClient` and
  programmatic search-index management. Owning the wrapper lets us use those
  directly instead of legacy thread-pool shims — clean, awaitable DB calls that
  blend with ADK's asyncio core.

The wrapper stays thin (it leans on native driver features rather than bespoke
glue) and exposes **documented extension points** — `distill_fn`, an injectable
`client` for CSFLE — instead of fragile half-implementations.

---

## What's in the repo

| Path | What it is |
| --- | --- |
| `adk_mongodb_memory/service.py` | `MongoAtlasMemoryService` — the `BaseMemoryService` over Atlas Vector Search. |
| `adk_mongodb_memory/embeddings.py` | `VoyageEmbedder` — the `input_type`-aware async embedder + factory. |
| `adk_mongodb_memory/__init__.py` | Public API re-exports. |
| `pyproject.toml` | PEP 621 packaging + the optional `encryption` (CSFLE) extra. |
| `examples/01_quickstart.py` | Connect → index → store → semantic search with scores. |
| `examples/02_agent_load_memory.py` | Full ADK flow: `LlmAgent` + `load_memory`, auto-save via `after_agent_callback`. |
| `examples/03_multi_tenant_isolation.py` | Proves per-tenant isolation. |
| `examples/04_ttl_and_distillation.py` | TTL compliance index + transcript distillation. |
| `examples/05_csfle.py` | Atlas Vector Search + CSFLE together: search the safe text, decrypt the PII that rode along. |

Each example is self-contained and prints a readable narrative. Example 02
demonstrates the enterprise capture-then-recall loop: in turn 1 an agent hears
*"I'm allergic to peanuts and prefer window seats,"* an `after_agent_callback`
auto-saves the session, and in a **brand-new session** a second agent answers a
flight/snack question by recalling both facts from Atlas via `load_memory`.

All five run against the **MongoDB Atlas Local** Docker image (it bundles `mongod`
+ `mongot`, so `$vectorSearch` works on a laptop with no cloud account), and all
five pass.

---

## Getting started

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip

# 2. Install the package (editable) with the CSFLE extra
pip install -e ".[encryption]"

# 3. Keys: copy and fill in VOYAGE_API_KEY (required); a Gemini key for example 02
cp .env.example .env

# 4. A MongoDB with Vector Search (local Atlas — or set MONGODB_URI to cloud)
docker run -d -p 27017:27017 --name adk-mongo mongodb/mongodb-atlas-local

# 5. Run
python examples/01_quickstart.py
```

The connection string resolves simply: `MONGODB_URI` if set, otherwise the local
Atlas default. On first run the service **creates the vector index for you** and
waits until it's queryable.

---

## Closing

Google ADK got the abstraction right: a tiny, swappable memory contract that the
agent runtime already knows how to use. What it doesn't ship is a backend that's
*both* durable-and-semantic *and* yours to operate. The in-memory stub is a toy;
the managed services are convenient but opaque and cloud-locked.

By implementing `BaseMemoryService` on MongoDB Atlas Vector Search, you keep the
clean ADK ergonomics **and** get the things production actually demands: semantic
recall over data you own, an index-level tenant boundary, field-level encryption
for PII, native retention, a distillation hook, async-correct I/O, and the full
MongoDB ecosystem pointed at the same collection. That's the value-add — and it's
about a thousand lines of (heavily documented) wrapper, not a second managed
system.

*Explore the code in [`README.md`](./README.md) and
[`examples/README.md`](./examples/README.md), then swap one line in your
`Runner`.*
