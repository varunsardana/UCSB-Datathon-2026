from sentence_transformers import SentenceTransformer

from config import settings

_model: SentenceTransformer | None = None


def load_embedding_model() -> None:
    global _model
    print(f"Loading embedding model: {settings.embedding_model}...")
    _model = SentenceTransformer(settings.embedding_model)
    print(f"Embedding model ready (dim={_model.get_sentence_embedding_dimension()})")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_embedding_model() first.")
    return _model.encode(texts, show_progress_bar=False).tolist()


def embed_query(text: str) -> list[float]:
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_embedding_model() first.")
    return _model.encode(text, show_progress_bar=False).tolist()
