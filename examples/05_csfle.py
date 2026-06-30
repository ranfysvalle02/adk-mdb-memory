"""Example 05 - Atlas Vector Search + Client-Side Field-Level Encryption (CSFLE).

Agent memory routinely accumulates PII (SSNs, PINs, health details). CSFLE lets
you encrypt those values *on the client* so they are ciphertext in transit and
at rest - MongoDB Atlas (and anyone with only a DB connection) never sees the
plaintext. This example shows that Atlas Vector Search and field-level encryption
**compose cleanly** - the realistic enterprise pattern on Atlas:

    embed + $vectorSearch on the NON-sensitive text  +  encrypt the sensitive
    side fields you never run $vectorSearch on.

You must NOT encrypt the field you embed: the stored vector and the query vector
have to be computed over the same readable text for similarity to mean anything.
Encrypting a *side field* (here an SSN in ``custom_metadata``) protects the secret
without breaking retrieval - semantic search still finds the memory by its safe
text, and the encrypted PII simply rides along as ciphertext.

We use **explicit (manual) encryption** via ``ClientEncryption`` because it needs
only libmongocrypt, which ships with ``pip install 'pymongo[encryption]'`` - no
separate crypt_shared library or mongocryptd process to stand up. It runs against
your Atlas cluster (or the bundled Atlas Local image) as-is. (The *automatic*
encryption pattern, sketched at the bottom of this file, is transparent but adds
that crypt_shared / mongocryptd deployment dependency.)

Flow:
  1. Configure a ``local`` KMS provider with a hardcoded 96-byte master key.
  2. Create/lookup a Data Encryption Key (DEK) in the key vault.
  3. Encrypt an SSN with that DEK.
  4. Create the Atlas Vector Search index and store a memory whose embedded text
     is non-sensitive, with the encrypted SSN as a side field, via ``add_memory``.
  5. Find the memory with Atlas Vector Search over the safe text - the encrypted
     SSN rides back along on the result as ciphertext (BSON Binary subtype 6).
  6. Decrypt it client-side to recover the plaintext.

Requires ``VOYAGE_API_KEY`` (for the embedding) and ``pymongo[encryption]``. Run::

    pip install -e ".[encryption]"
    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local   # or set MONGODB_URI
    python examples/05_csfle.py
"""

from __future__ import annotations

import asyncio
import base64

from pymongo.errors import OperationFailure

from _shared import (
    Config,
    banner,
    configure_logging,
    make_service,
    print_connection_banner,
    require_voyage_key,
    wait_until_searchable,
)

# ===========================================================================
# !!  DEMO MASTER KEY - DO NOT USE IN PRODUCTION / LIVE  !!
# ---------------------------------------------------------------------------
# CSFLE's "local" KMS provider takes a 96-byte master key that wraps (encrypts)
# your Data Encryption Keys. Hardcoding it here keeps the example self-contained
# and runnable offline, but it is the ONE thing you must never do for real:
# anyone who reads this key (e.g. from source control) can decrypt every secret.
#
# In production/live, source the master key from a real Key Management Service
# and NEVER hardcode or commit it:
#   * AWS KMS         -> kms_providers={"aws": {...}}, master_key={"region","key"}
#   * Azure Key Vault -> kms_providers={"azure": {...}}
#   * GCP KMS         -> kms_providers={"gcp": {...}}
#   * KMIP            -> kms_providers={"kmip": {...}}
# With a KMS provider, the master key stays inside the KMS; libmongocrypt calls
# the KMS to wrap/unwrap your DEKs. Only the *provider* changes below - the
# encrypt/decrypt flow is identical.
# ===========================================================================
_DEMO_LOCAL_MASTER_KEY_B64 = (
    "hjrjmNnAoqTOpD5WFHIWV0pdSFnKC/7VMdfzqUZruEU9dxdmGqYB8ZDr+Bp4mkRA"
    "9iDC8m0h1oLJIJ15oXi/Z9d4SJbuF946rHjnZUSYEgnr5UL6GShJqwuP5x1JWO2q"
)

# The key vault stores your (KMS-wrapped) Data Encryption Keys. The leading "__"
# is a convention marking it internal; it can live in any database.
KEY_VAULT_DB = "encryption"
KEY_VAULT_COLLECTION = "__keyVault"
KEY_VAULT_NAMESPACE = f"{KEY_VAULT_DB}.{KEY_VAULT_COLLECTION}"
# A stable alias so re-runs reuse one DEK instead of minting a new one each time.
DEK_ALT_NAME = "adk-mongodb-memory-demo-pii-key"

DB_NAME = "adk_memory_examples"
COLLECTION = "csfle_secure_memories"
APP_NAME = "secure_support_bot"
USER_ID = "erin"

# A non-sensitive, distilled summary: this is what we embed AND vector-search on.
PROFILE_TEXT = "Customer profile: enterprise tier; prefers email contact; identity verified."
# The sensitive value we encrypt as a side field (kept OUT of the embedding).
SENSITIVE_SSN = "123-45-6789"


async def main() -> None:
    configure_logging()
    cfg = Config()

    banner("Example 05 - Atlas Vector Search + CSFLE on the same memory")
    if not require_voyage_key(cfg):
        return
    print_connection_banner(cfg, db_name=DB_NAME, collection=COLLECTION)
    print(
        "\nThis runs the full Atlas loop: embed + $vectorSearch over the SAFE text, "
        "while the PII\nside field is encrypted client-side (only libmongocrypt is "
        "needed - no crypt_shared/mongocryptd)."
    )

    # Lazy import so the example gives clear guidance if the extra is missing.
    try:
        from bson.binary import STANDARD, Binary
        from bson.codec_options import CodecOptions
        from pymongo.asynchronous.encryption import AsyncClientEncryption
        from pymongo.encryption import Algorithm
    except ImportError:
        print(
            "\nThis example needs the encryption extra (libmongocrypt). Install it:\n"
            "    pip install 'pymongo[encryption]'   # or: pip install -e \".[encryption]\"\n"
        )
        return

    local_master_key = base64.b64decode(_DEMO_LOCAL_MASTER_KEY_B64)
    assert len(local_master_key) == 96, "local KMS master key must be exactly 96 bytes"
    kms_providers = {"local": {"key": local_master_key}}

    async with make_service(cfg, db_name=DB_NAME, collection_name=COLLECTION) as service:
        # Keep re-runs clean (add_memory inserts a fresh doc each time).
        await service.collection.delete_many({"app_name": APP_NAME, "user_id": USER_ID})

        # Best practice: a unique partial index on keyAltNames so a given alt name
        # maps to exactly one DEK. Best-effort - a shared key vault may already
        # carry an equivalent index (possibly under a different name), so we treat
        # an "already exists / options conflict" as success.
        key_vault = service.client[KEY_VAULT_DB][KEY_VAULT_COLLECTION]
        try:
            await key_vault.create_index(
                "keyAltNames",
                unique=True,
                partialFilterExpression={"keyAltNames": {"$exists": True}},
            )
        except OperationFailure as exc:
            print(f"(key vault keyAltNames index already present: {exc.code}; continuing)")

        # ClientEncryption reuses the service's AsyncMongoClient as its key-vault
        # client; STANDARD uuid representation is required for the key vault.
        client_encryption = AsyncClientEncryption(
            kms_providers=kms_providers,
            key_vault_namespace=KEY_VAULT_NAMESPACE,
            key_vault_client=service.client,
            codec_options=CodecOptions(uuid_representation=STANDARD),
        )
        try:
            banner("Step 1: create or look up the Data Encryption Key (DEK)")
            existing = await client_encryption.get_key_by_alt_name(DEK_ALT_NAME)
            if existing is not None:
                data_key_id = existing["_id"]
                print(f"Reusing existing DEK (alt name {DEK_ALT_NAME!r}).")
            else:
                data_key_id = await client_encryption.create_data_key(
                    "local", key_alt_names=[DEK_ALT_NAME]
                )
                print(f"Created a new DEK (alt name {DEK_ALT_NAME!r}).")

            banner("Step 2: encrypt the sensitive field (client-side)")
            # Deterministic encryption yields stable ciphertext, so you *can* run
            # equality queries on the encrypted field - at the cost of revealing
            # which records share a value. For a pure side field you never query,
            # prefer Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Random.
            ssn_ciphertext = await client_encryption.encrypt(
                SENSITIVE_SSN,
                algorithm=Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Deterministic,
                key_id=data_key_id,
            )
            print(f"Plaintext SSN     : {SENSITIVE_SSN}")
            print(f"Encrypted (type)  : Binary subtype {ssn_ciphertext.subtype} (6 = encrypted)")
            print(f"Encrypted (b64)   : {base64.b64encode(bytes(ssn_ciphertext)).decode()[:48]}...")

            banner("Step 3: build the Atlas Vector Search index and store the memory")
            from google.adk.memory.memory_entry import MemoryEntry
            from google.genai.types import Content, Part

            # Create the Atlas Vector Search index (idempotent) so the safe text is
            # searchable. The SSN ciphertext is neither embedded nor indexed.
            await service.setup_indexes(wait_for_vector_index=True, timeout_seconds=180.0)

            # The embedded text is the non-sensitive profile; the SSN ciphertext
            # rides along as a side field in custom_metadata (never embedded).
            entry = MemoryEntry(
                content=Content(role="user", parts=[Part(text=PROFILE_TEXT)]),
                author="user",
            )
            await service.add_memory(
                app_name=APP_NAME,
                user_id=USER_ID,
                memories=[entry],
                custom_metadata={"ssn": ssn_ciphertext, "ssn_algorithm": "deterministic"},
            )
            print(f"Stored memory. Embedded text (safe): {PROFILE_TEXT!r}")

            banner("Step 4: find it with Atlas Vector Search over the SAFE text")
            # A semantic query against the non-sensitive profile - never the SSN.
            query = "enterprise customer contact and identity verification status"
            await wait_until_searchable(
                service, app_name=APP_NAME, user_id=USER_ID, query=query
            )
            response = await service.search_memory(
                app_name=APP_NAME, user_id=USER_ID, query=query
            )
            print(f"Query: {query!r} -> {len(response.memories)} result(s)")
            if not response.memories:
                print("  (nothing searchable yet - on a cold index, wait a moment and re-run)")
                return
            hit = response.memories[0]
            hit_text = " ".join(p.text for p in (hit.content.parts or []) if p.text)
            score = hit.custom_metadata.get("score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"Matched by similarity (score={score_str}): {hit_text!r}")

            # The encrypted SSN came back on the retrieved memory, still ciphertext.
            retrieved_ssn = hit.custom_metadata.get("ssn")
            is_encrypted = isinstance(retrieved_ssn, Binary) and retrieved_ssn.subtype == 6
            print(f"PII rode along as BSON Binary subtype 6 (still encrypted): {is_encrypted}")
            print(f"What Atlas stored & returned: {base64.b64encode(bytes(retrieved_ssn)).decode()[:48]}...")

            banner("Step 5: decrypt the retrieved ciphertext to recover the plaintext")
            recovered = await client_encryption.decrypt(retrieved_ssn)
            print(f"Decrypted SSN     : {recovered}")
            print(f"Round-trip OK     : {recovered == SENSITIVE_SSN}")
        finally:
            await client_encryption.close()

    banner("Done")


# ===========================================================================
# Production alternative: AUTOMATIC encryption (transparent, schema-driven)
# ---------------------------------------------------------------------------
# Instead of encrypting/decrypting each field by hand, you can let the driver do
# it transparently by configuring AutoEncryptionOpts on the client and injecting
# it via MongoAtlasMemoryService(client=...). Reads/writes of the configured
# fields are auto-encrypted/decrypted; your application code stays plaintext.
#
# This requires the Automatic Encryption Shared Library (crypt_shared) on the
# client host, OR a running mongocryptd process - so it is heavier to deploy
# than the explicit pattern above. Sketch:
#
#   from pymongo import AsyncMongoClient
#   from pymongo.encryption_options import AutoEncryptionOpts
#
#   schema_map = {
#       f"{DB_NAME}.{COLLECTION}": {
#           "bsonType": "object",
#           "properties": {
#               "custom_metadata": {
#                   "bsonType": "object",
#                   "properties": {
#                       "ssn": {
#                           "encrypt": {
#                               "bsonType": "string",
#                               # Note the hyphen form used in JSON schema maps:
#                               "algorithm": "AEAD_AES_256_CBC_HMAC_SHA_512-Deterministic",
#                               "keyId": [data_key_id],  # the DEK Binary from above
#                           }
#                       }
#                   }
#               }
#           }
#       }
#   }
#   auto_opts = AutoEncryptionOpts(
#       kms_providers=kms_providers,
#       key_vault_namespace=KEY_VAULT_NAMESPACE,
#       schema_map=schema_map,
#       # one of these is required for automatic encryption:
#       crypt_shared_lib_path="/path/to/mongo_crypt_v1.so",
#       # or rely on a running mongocryptd (mongocryptd_uri="mongodb://localhost:27020").
#   )
#   client = AsyncMongoClient(cfg.mongodb_uri, auto_encryption_opts=auto_opts)
#   service = MongoAtlasMemoryService(client=client, db_name=DB_NAME, embedding_fn=embedder)
#   # ...now custom_metadata.ssn is encrypted on write and decrypted on read,
#   # automatically. The service won't close an injected client for you.
# ===========================================================================


if __name__ == "__main__":
    asyncio.run(main())
