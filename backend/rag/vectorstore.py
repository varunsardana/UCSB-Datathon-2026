from typing import Optional

import chromadb

from config import settings

_client = None
_collection: Optional[chromadb.Collection] = None

COLLECTION_NAME = "disaster_kb"


def init_vectorstore() -> None:
    global _client, _collection
    print(f"Initializing ChromaDB at: {settings.chroma_persist_dir}")
    _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"ChromaDB ready — collection '{COLLECTION_NAME}' has {_collection.count()} chunks")


def get_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError("Vector store not initialized. Call init_vectorstore() first.")
    return _collection
