from datetime import datetime
from typing import List, Callable

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions import Session
from google.adk.types import MemoryEntry, SearchMemoryResponse
from google.genai.types import Content, Part

# Import the native Async client introduced in PyMongo 4.9+ / 4.13 GA
from pymongo import AsyncMongoClient


class MongoAtlasMemoryService(BaseMemoryService):
    """
    MongoDB Atlas long-term memory provider 
    leveraging PyMongo's native asynchronous driver.
    """

    def __init__(
        self,
        connection_string: str,
        db_name: str,
        collection_name: str = "agent_memories",
        embedding_fn: Callable[[str], List[float]] = None,
        vector_index_name: str = "vector_index",
        embedding_field: str = "embedding_vector",
    ):
        # Native async client - tied cleanly to a single event loop
        self.client = AsyncMongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        if not embedding_fn:
            raise ValueError("An embedding_fn must be provided for semantic capabilities.")
        self.embedding_fn = embedding_fn
        
        self.vector_index_name = vector_index_name
        self.embedding_field = embedding_field

    async def setup_indexes(self) -> None:
        """
        Optional initialization method to enforce metadata indexing natively.
        Call this during your application bootstrap sequence.
        """
        await self.collection.create_index(
            [
                ("app_name", 1),
                ("user_id", 1),
                ("session_id", 1)
            ],
            unique=True,
            name="idx_tenant_session_lookup"
        )

    async def add_session_to_memory(self, session: Session) -> None:
        """
        Compiles the finished session context and saves it completely non-blocked.
        """
        text_segments = []
        for event in session.events:
            if hasattr(event, "content") and event.content and event.content.parts:
                text = "".join([p.text for p in event.content.parts if p.text])
                if text.strip():
                    author = getattr(event, "author", "Unknown")
                    text_segments.append(f"{author}: {text}")

        if not text_segments:
            return  

        combined_transcript = "\n".join(text_segments)
        vector_embedding = self.embedding_fn(combined_transcript)

        payload = {
            "app_name": session.app_name,
            "user_id": session.user_id,
            "session_id": session.session_id,
            "transcript": combined_transcript,
            self.embedding_field: vector_embedding,
            "updated_at": datetime.utcnow()
        }

        # Direct, awaitable operational database write
        await self.collection.update_one(
            {
                "app_name": session.app_name, 
                "user_id": session.user_id, 
                "session_id": session.session_id
            },
            {"$set": payload},
            upsert=True
        )

    async def search_memory(self, app_name: str, user_id: str, query: str, **kwargs) -> SearchMemoryResponse:
        """
        Executes a true asynchronous aggregate query for Atlas Vector Search.
        """
        query_vector = self.embedding_fn(query)
        limit = kwargs.get("limit", 3)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": self.embedding_field,
                    "queryVector": query_vector,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {
                        "$and": [
                            {"app_name": {"$eq": app_name}},
                            {"user_id": {"$eq": user_id}}
                        ]
                    }
                }
            }
        ]

        cursor = self.collection.aggregate(pipeline)
        docs = await cursor.to_list(length=None)
        
        memories = []
        for doc in docs:
            entry = MemoryEntry(
                content=Content(parts=[Part(text=doc["transcript"])]),
                author="LongTermMemory",
                timestamp=doc.get("updated_at"),
                custom_metadata={"session_id": doc.get("session_id")}
            )
            memories.append(entry)

        return SearchMemoryResponse(memories=memories)
