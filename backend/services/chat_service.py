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
) -> AsyncIterator[str]:
    """
    Full RAG pipeline as an async generator of text tokens.

    Both Aliza's Prophet forecast and Nikita's XGBoost prediction are injected
    as guaranteed structured context blocks — not left to chance via ChromaDB
    similarity search. The retriever then adds matching KB chunks on top.
    """

    # ── Step 1a: Prophet forecast context (Aliza's model) ────────────────────
    # Reads prophet_state_forecasts.json directly for state + disaster_type.
    # Does NOT need a fips_code. Returns "" if no forecast exists for this combo.
    forecast_context = format_forecast_context(state, disaster_type)

    # ── Step 1b: XGBoost prediction context (Nikita's model) ─────────────────
    # Priority 1: exact fips_code match in model_predictions.json
    # Priority 2: best state-level match (most sectors, same disaster type)
    # Priority 3: None → prompt will show "No prediction available"
    prediction = None
    if disaster_type:
        if fips_code:
            prediction = get_prediction(disaster_type, fips_code)
        if prediction is None and state:
            prediction = get_prediction_by_state(disaster_type, state)
    prediction_context = format_prediction_context(prediction)

    # ── Step 2: Retrieve KB chunks from ChromaDB ─────────────────────────────
    # Returns top-k chunks from: knowledge docs (FEMA, unemployment, WARN Act,
    # COBRA, retraining, recovery timelines, transferable skills) + forecast
    # profiles + model output narratives — filtered by state/disaster metadata.
    chunks = retrieve(
        query=message,
        state=state,
        disaster_type=disaster_type,
        top_k=settings.top_k,
    )

    # ── Step 3: Build augmented system prompt ─────────────────────────────────
    system_prompt = build_system_prompt(
        forecast_context=forecast_context,
        prediction_context=prediction_context,
        retrieved_docs=chunks,
        state=state,
        disaster_type=disaster_type,
        job_title=job_title,
    )

    # ── Step 4: Stream via chosen provider ────────────────────────────────────
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
