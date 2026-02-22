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

    Metadata filter strategy:
    - state + disaster_type: include state docs, disaster docs, model outputs,
      Prophet forecast profiles, WARN Act, COBRA, recovery timelines,
      transferable skills — the full picture for an informed answer.
    - state only: state unemployment docs + national KB (WARN Act, COBRA,
      retraining, recovery timelines, transferable skills).
    - disaster_type only: disaster-specific docs + model outputs + Prophet forecasts
      + FEMA programs + recovery timelines.
    - neither: no metadata filter — pure similarity search across all chunks.

    Note: forecast_context and prediction_context are injected DIRECTLY into the
    system prompt by chat_service.py, so not finding them here is not a problem.
    The retriever's job is to surface supporting KB material (programs, benefits,
    timelines, skills) that enriches the LLM's answer beyond the model outputs.
    """
    collection = get_collection()
    query_vec = embed_query(query)
    k = top_k or settings.top_k

    where_filter = None

    if state and disaster_type:
        where_filter = {
            "$or": [
                {"state":    {"$eq": state}},
                {"disaster_type": {"$eq": disaster_type}},
                {"category": {"$eq": "model_output"}},
                {"category": {"$eq": "forecast"}},
                {"category": {"$eq": "fema"}},
                {"category": {"$eq": "warn_act"}},
                {"category": {"$eq": "cobra"}},
                {"category": {"$eq": "retraining"}},
                {"category": {"$eq": "recovery_timelines"}},
                {"category": {"$eq": "transferable_skills"}},
            ]
        }
    elif state:
        where_filter = {
            "$or": [
                {"state":    {"$eq": state}},
                {"category": {"$eq": "warn_act"}},
                {"category": {"$eq": "cobra"}},
                {"category": {"$eq": "retraining"}},
                {"category": {"$eq": "recovery_timelines"}},
                {"category": {"$eq": "transferable_skills"}},
            ]
        }
    elif disaster_type:
        where_filter = {
            "$or": [
                {"disaster_type": {"$eq": disaster_type}},
                {"category": {"$eq": "model_output"}},
                {"category": {"$eq": "forecast"}},
                {"category": {"$eq": "fema"}},
                {"category": {"$eq": "recovery_timelines"}},
                {"category": {"$eq": "transferable_skills"}},
            ]
        }
    # else: no filter — pure cosine similarity across all chunks

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, collection.count()) if collection.count() > 0 else k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":       results["documents"][0][i],
            "metadata":   results["metadatas"][0][i],
            "similarity": round(1 - results["distances"][0][i], 4),
        })

    return chunks
