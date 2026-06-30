# Examples

Runnable, self-contained examples for `MongoAtlasMemoryService`. They run against
a **local MongoDB Atlas** container (no cloud account required) or any Atlas
cluster you point them at. The Atlas Local image bundles `mongod` **and**
`mongot`, so `$vectorSearch` works on your laptop for development.

Every example uses **real [Voyage AI](https://docs.voyageai.com/docs/embeddings)
embeddings** and therefore requires `VOYAGE_API_KEY` (there is no offline
fallback). Example 02 additionally needs a Gemini key for its chat `LlmAgent`. If
a required key is missing, the example prints clear guidance and exits cleanly —
no stack trace.

## TL;DR — the value of each example at a glance

What each one *proves about running agent memory on MongoDB Atlas*, and why you'd
care:

| # | Example | Value at a glance |
| --- | --- | --- |
| **01** | `01_quickstart.py` | **Durable semantic recall.** Store memories and get them back ranked by *meaning* via Atlas Vector Search — not keyword matching, and not lost on restart. The "is it working?" smoke test. |
| **02** | `02_agent_load_memory.py` | **Drop-in ADK memory.** A real `LlmAgent` recalls facts in a brand-new session through the built-in `load_memory` tool — wired in with a single `memory_service=` on the `Runner`. |
| **03** | `03_multi_tenant_isolation.py` | **Hard multi-tenant isolation.** An index-backed `$vectorSearch` filter on `(app_name, user_id)` guarantees one tenant can *never* retrieve another's memory — the SaaS safety property. |
| **04** | `04_ttl_and_distillation.py` | **Compliance + cost control.** A native MongoDB TTL index auto-purges old memories (GDPR/HIPAA) with zero cron, and a `distill_fn` compresses transcripts before they're embedded. |
| **05** | `05_csfle.py` | **Security without losing search.** Atlas Vector Search runs over the safe text while PII (an SSN) stays client-side encrypted (CSFLE) — proof the two compose cleanly on Atlas. |

Each example is independent and idempotent — run any of them, in any order. They
all target the same local Atlas container and isolate their data in separate
collections.

## What's here

| File | What it shows | Extra key beyond `VOYAGE_API_KEY`? |
| --- | --- | --- |
| `_shared.py` | Example-only glue: `Config`, MongoDB URI resolution, the Voyage embedder factory, `make_service()`, and presentation helpers. | – |
| `01_quickstart.py` | The "hello world": connect → build indexes → store memories (`input_type="document"`) → semantic `search_memory` (`input_type="query"`) with scores. | No |
| `02_agent_load_memory.py` | Full ADK flow: `LlmAgent` + `Runner`, an `after_agent_callback` that auto-saves the session, then recall in a **new** session via the built-in `load_memory` tool. | **Gemini** |
| `03_multi_tenant_isolation.py` | Two users store near-identical content; each search returns **only** the caller's data (per-tenant `$vectorSearch` filter). | No |
| `04_ttl_and_distillation.py` | Construct with `ttl_seconds=` (creates a TTL index) and a `distill_fn`; store a transcript and inspect the distilled text. | No |
| `05_csfle.py` | **Atlas Vector Search + Client-Side Field-Level Encryption**: encrypt a PII side field (SSN), embed only the non-sensitive text, then find the memory by `$vectorSearch` over that safe text while the PII rides back as ciphertext — and decrypt it. | No (needs the `encryption` extra) |

## Prerequisites

* **Python 3.10+** and **Docker** (for local Atlas) — or a cloud Atlas cluster.
* Install the project (editable) with the CSFLE extra:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[encryption]"     # installs the package + libmongocrypt
```

* Set `VOYAGE_API_KEY` (and, for example 02, a Gemini key). Copy `.env.example`
  to `.env` and fill it in, or export them in your shell.

## 1. Start a local Atlas instance (or use cloud)

```bash
docker run -d -p 27017:27017 --name adk-mongo mongodb/mongodb-atlas-local
```

This single-node Atlas Local image bundles `mongod` + `mongot`, so `$vectorSearch`
works locally. The examples auto-target it at
`mongodb://localhost:27017/?directConnection=true` when `MONGODB_URI` is unset.

> `directConnection=true` matters: the Atlas Local image is a single-node replica
> set, and the flag tells the driver to connect directly to it.

To use **cloud Atlas** instead, set `MONGODB_URI` to your SRV string (the cluster
must support Vector Search — **M10+ or Flex**; free `M0` does not).

Clean up the local container when you're done:

```bash
docker rm -f adk-mongo
```

## 2. URI resolution (no `.env` needed for local)

The connection string is resolved simply: **`MONGODB_URI` if set, otherwise the
local default** `mongodb://localhost:27017/?directConnection=true`. Each example
prints which target it used.

## 3. Run the examples

```bash
python examples/01_quickstart.py
python examples/03_multi_tenant_isolation.py
python examples/04_ttl_and_distillation.py
python examples/05_csfle.py

# Needs GOOGLE_API_KEY / GEMINI_API_KEY in addition to VOYAGE_API_KEY:
python examples/02_agent_load_memory.py
```

### What to expect

* **`01_quickstart.py`** — submits the vector index, waits for it to become
  queryable, stores two memories, then searches *"Which coding language do I
  prefer?"*. The Rust/Project Alpha memory ranks above the latte one, each with a
  real Voyage similarity score:

  ```
  Top 2 result(s) by similarity:
    1. (score=0.7773) user: My favorite programming language is Rust ...
    2. (score=0.6657) user: I usually start my day with an oat milk latte ...
  ```

* **`02_agent_load_memory.py`** — Turn 1 the agent acknowledges *"I'm allergic to
  peanuts and prefer window seats"*; an `after_agent_callback` auto-saves that
  session to MongoDB. Turn 2, in a brand-new session, the agent calls
  `load_memory` and answers a flight/snack question by recalling both facts.

* **`03_multi_tenant_isolation.py`** — `alice` and `bob` both store a "PIN hint",
  then both run the same query. Each gets back only their own memory and the
  script prints `PASS: every search returned only the calling tenant's memory.`

* **`04_ttl_and_distillation.py`** — prints the TTL index it created
  (`expireAfterSeconds=2592000`), shows the raw four-line transcript vs. the
  distilled (user-lines-only) text that was actually stored, and confirms the
  distilled memory is still searchable.

* **`05_csfle.py`** — creates a local Data Encryption Key, encrypts an SSN
  client-side, stores it as a side field in `custom_metadata` while embedding only
  the non-sensitive profile text, then **finds the memory with Atlas Vector Search
  over that safe text**. The encrypted SSN rides back on the result as BSON Binary
  subtype 6 (ciphertext), and is decrypted client-side to plaintext:

  ```
  Matched by similarity (score~0.81): 'Customer profile: enterprise tier; ...'
  PII rode along as BSON Binary subtype 6 (still encrypted): True
  Decrypted SSN     : 123-45-6789
  Round-trip OK     : True
  ```

## CSFLE: explicit vs. automatic encryption

Example 05 proves that **Atlas Vector Search and field-level encryption compose**:
it embeds and `$vectorSearch`-es the safe profile text while the SSN stays
encrypted as a side field. It uses **explicit (manual) encryption** via
`pymongo.encryption`'s `ClientEncryption` — you call `encrypt()` / `decrypt()`
yourself — because that needs only libmongocrypt, which ships with
`pip install 'pymongo[encryption]'`. There's no separate `crypt_shared` library or
`mongocryptd` process to deploy; it runs against your Atlas cluster (or the bundled
Atlas Local image) as-is.

The realistic pattern, shown in the example:

* **Embed and `$vectorSearch` on non-sensitive (distilled) text.** You must *not*
  encrypt the field you run `$vectorSearch` on — the stored vector and the query
  vector have to be computed over the same readable text.
* **Encrypt the sensitive side fields** (e.g. an SSN in `custom_metadata`) that you
  never search by similarity; they ride along with the result as ciphertext.

> **Demo key warning:** example 05 hardcodes a 96-byte `local` KMS master key so
> it runs out of the box. **Never do this in production.** Source the master key
> from a real KMS — **AWS KMS, Azure Key Vault, GCP KMS, or KMIP** — and never
> hardcode or commit it. Only the KMS *provider* config changes; the
> encrypt/decrypt flow is identical.

For transparent, schema-driven **automatic encryption**, build an
`AsyncMongoClient(..., auto_encryption_opts=AutoEncryptionOpts(...))` and inject
it via `MongoAtlasMemoryService(client=...)`. That requires the Automatic
Encryption Shared Library (`crypt_shared`) or `mongocryptd`. See the commented
production sketch at the bottom of `05_csfle.py`.

## Notes

* The examples write to a dedicated database, `adk_memory_examples`, with a
  separate collection per example, so they stay isolated from each other.
  Session-derived memories are upserted per `(app_name, user_id, session_id)`, so
  re-running an example is idempotent.
* `input_type` matters for retrieval quality: the service embeds stored text as
  `"document"` and queries as `"query"` automatically.
* First-time index builds take a few seconds on a cold collection; the scripts
  poll until the vector index is queryable and the writes are searchable.
