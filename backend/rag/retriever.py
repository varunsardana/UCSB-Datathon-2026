from rag.embeddings import embed_query
from rag.vectorstore import get_collection
from config import settings


def retrieve(
    query: str,
    state: str | None = None,
    disaster_type: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """
    Query ChromaDB for the top-k most relevant chunks.

    Metadata filters applied when provided:
    - state: match docs tagged for that state (or no state tag, i.e. national docs)
    - disaster_type: match docs for that disaster type OR model outputs

    Returns a list of dicts: { text, metadata, similarity }
    """
    collection = get_collection()
    query_vec = embed_query(query)
    k = top_k or settings.top_k

    # Build where filter — always include model_output category regardless of filters
    where_filter = None
    if state and disaster_type:
        where_filter = {
            "$or": [
                {"state": {"$eq": state}},
                {"disaster_type": {"$eq": disaster_type}},
                {"category": {"$eq": "model_output"}},
                {"category": {"$eq": "warn_act"}},
                {"category": {"$eq": "cobra"}},
            ]
        }
    elif state:
        where_filter = {
            "$or": [
                {"state": {"$eq": state}},
                {"category": {"$eq": "warn_act"}},
                {"category": {"$eq": "cobra"}},
                {"category": {"$eq": "retraining"}},
            ]
        }
    elif disaster_type:
        where_filter = {
            "$or": [
                {"disaster_type": {"$eq": disaster_type}},
                {"category": {"$eq": "model_output"}},
                {"category": {"$eq": "fema"}},
            ]
        }

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, collection.count()) if collection.count() > 0 else k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": round(1 - results["distances"][0][i], 4),
        })

    return chunks
