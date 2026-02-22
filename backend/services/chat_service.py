"""
Chat service — the core RAG orchestrator.

Flow per request:
  1. Get prediction context  (pre-computed JSON → live model fallback)
  2. Retrieve relevant KB chunks from ChromaDB
  3. Build the augmented system prompt
  4. Stream response via the configured LLM provider:
       LLM_PROVIDER=local  → Ollama (gemma3:1b or any local model)
       LLM_PROVIDER=claude → Anthropic Claude API
"""

from typing import AsyncIterator

from config import settings
from rag.prompts import build_system_prompt
from rag.retriever import retrieve
from services.model_service import format_prediction_context, get_prediction


async def chat_stream(
    message: str,
    state: str | None = None,
    disaster_type: str | None = None,
    job_title: str | None = None,
    fips_code: str | None = None,
) -> AsyncIterator[str]:
    """
    Full RAG pipeline as an async generator of text tokens.
    Provider is selected via LLM_PROVIDER in .env (default: "local").
    """
    # ── Step 1: Prediction context ─────────────────────────────────────────
    prediction = None
    if disaster_type and fips_code:
        prediction = get_prediction(disaster_type, fips_code)
    prediction_context = format_prediction_context(prediction)

    # ── Step 2: Retrieve relevant KB chunks ────────────────────────────────
    chunks = retrieve(
        query=message,
        state=state,
        disaster_type=disaster_type,
        top_k=settings.top_k,
    )

    # ── Step 3: Build system prompt ────────────────────────────────────────
    system_prompt = build_system_prompt(
        prediction_context=prediction_context,
        retrieved_docs=chunks,
        state=state,
        disaster_type=disaster_type,
        job_title=job_title,
    )

    # ── Step 4: Stream via chosen provider ─────────────────────────────────
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
