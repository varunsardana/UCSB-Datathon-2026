SYSTEM_PROMPT_TEMPLATE = """\
You are DisasterShift's AI Workforce Advisor. Your job is to help workers and policymakers \
understand and respond to disaster-driven workforce disruption.

STRICT RULES:
1. ONLY use information from the PREDICTION CONTEXT and RETRIEVED KNOWLEDGE below.
2. Do NOT invent statistics, dollar amounts, deadlines, or program names.
3. If something is not in the provided context, say "I don't have specific data on that."
4. Always cite your source category in parentheses — e.g., (FEMA programs), (Prediction model), (CA unemployment).
5. Structure your response with clear headers.
6. Be direct and specific — use numbers, timelines, and deadlines from the context.
7. Always end with a "Immediate Next Steps" section with 2-3 actions the person can do TODAY.

---

PREDICTION CONTEXT (from our forecasting model):
{prediction_context}

---

RETRIEVED KNOWLEDGE:
{retrieved_docs}

---

USER CONTEXT:
- Location / State: {state}
- Disaster type: {disaster_type}
- Job title / Industry: {job_title}

---

Now answer the user's question using only the information above.\
"""


def format_retrieved_docs(docs: list[dict]) -> str:
    """Format retrieved chunks with source labels for the prompt."""
    if not docs:
        return "No relevant documents retrieved."
    parts = []
    for i, doc in enumerate(docs, 1):
        category = doc["metadata"].get("category", "unknown")
        source = doc["metadata"].get("source", "")
        label = f"[Source {i} — {category}]"
        if source:
            label += f" ({source})"
        parts.append(f"{label}\n{doc['text']}")
    return "\n\n".join(parts)


def build_system_prompt(
    prediction_context: str,
    retrieved_docs: list[dict],
    state: str | None,
    disaster_type: str | None,
    job_title: str | None,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        prediction_context=prediction_context or "No specific prediction available for this scenario.",
        retrieved_docs=format_retrieved_docs(retrieved_docs),
        state=state or "Not specified",
        disaster_type=disaster_type or "Not specified",
        job_title=job_title or "Not specified",
    )
