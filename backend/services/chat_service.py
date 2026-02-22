"""
Chat service — the core RAG orchestrator.

Flow per request:
  1a. Prophet forecast context   — direct file lookup, guaranteed in prompt
  1b. XGBoost prediction context — exact fips_code → state fallback → None
  2.  Retrieve relevant KB chunks from ChromaDB (knowledge docs + model narratives)
  3.  Build the augmented system prompt (two named model sections + KB)
  4.  Stream response via the configured LLM provider:
        LLM_PROVIDER=local  → Ollama (any local model)
        LLM_PROVIDER=claude → Anthropic Claude API
"""

from typing import AsyncIterator

from config import settings
from rag.prompts import build_system_prompt
from rag.retriever import retrieve
from services.model_service import (
    format_forecast_context,
    format_prediction_context,
    get_prediction,
    get_prediction_by_state,
)


async def chat_stream(
    message: str,
    state: str | None = None,
    disaster_type: str | None = None,
    job_title: str | None = None,
    fips_code: str | None = None,
    audience_type: str | None = None,
) -> AsyncIterator[str]:
    """
    Full RAG pipeline as an async generator of text tokens.

    Both Aliza's Prophet forecast and Nikita's XGBoost prediction are injected
    as guaranteed structured context blocks — not left to chance via ChromaDB
    similarity search. The retriever then adds matching KB chunks on top.
    """

    # ── Step 0: Detect audience upfront so we can show it in the status ─────
    from rag.prompts import detect_audience, AUDIENCE_TYPES
    resolved_audience = (
        audience_type if audience_type in AUDIENCE_TYPES
        else detect_audience(job_title, message)
    )
    audience_label = resolved_audience.replace("_", " ").title()
    yield f"__status__Mode: {audience_label} guidance"

    # ── Step 1a: Prophet forecast context ────────────────────────────────────
    location_label = f"{state} {disaster_type}".strip() if (state or disaster_type) else "your area"
    yield f"__status__Fetching disaster frequency forecast for {location_label}..."
    forecast_context = format_forecast_context(state, disaster_type)

    # ── Step 1b: XGBoost prediction context ──────────────────────────────────
    yield "__status__Analyzing sector-level employment impact..."
    prediction = None
    if disaster_type:
        if fips_code:
            prediction = get_prediction(disaster_type, fips_code)
        if prediction is None and state:
            prediction = get_prediction_by_state(disaster_type, state)
    prediction_context = format_prediction_context(prediction)

    # ── Step 2: SQL structured query (rankings, comparisons, portfolio) ─────
    sql_context = ""
    try:
        from rag.query_router import route_and_query
        sql_result = route_and_query(message, state=state, disaster_type=disaster_type)
        if sql_result:
            yield "__status__Running structured data analysis..."
            sql_context = sql_result
    except Exception as e:
        print(f"SQL routing skipped: {e}")

    # ── Step 3: Retrieve KB chunks from ChromaDB ─────────────────────────────
    yield "__status__Searching knowledge base for programs and resources..."
    chunks = retrieve(
        query=message,
        state=state,
        disaster_type=disaster_type,
        top_k=settings.top_k,
    )

    # ── Step 4: Build augmented system prompt ─────────────────────────────────
    system_prompt = build_system_prompt(
        forecast_context=forecast_context,
        prediction_context=prediction_context,
        retrieved_docs=chunks,
        state=state,
        disaster_type=disaster_type,
        job_title=job_title,
        question=message,
        audience_type=resolved_audience,
        sql_context=sql_context,
    )

    # ── Step 5: Stream via chosen provider ────────────────────────────────────
    yield "__status__Generating response..."
    provider = settings.llm_provider.lower()

    if provider == "local":
        async for token in _stream_ollama(system_prompt, message):
            yield token
    elif provider == "claude":
        async for token in _stream_claude(system_prompt, message):
            yield token
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'local' or 'claude'.")


# ── Ollama provider ────────────────────────────────────────────────────────────

async def _stream_ollama(system_prompt: str, user_message: str) -> AsyncIterator[str]:
    """Stream tokens from a local Ollama model."""
    import ollama

    stream = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


# ── Claude provider ────────────────────────────────────────────────────────────

_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Either add it to .env or switch LLM_PROVIDER=local."
            )
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


async def _stream_claude(system_prompt: str, user_message: str) -> AsyncIterator[str]:
    """Stream tokens from Claude API."""
    client = _get_anthropic_client()

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text
