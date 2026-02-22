from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM provider: "local" (Ollama) or "claude" (Anthropic API)
    llm_provider: str = "local"

    # Anthropic (only needed when llm_provider = "claude")
    anthropic_api_key: str = ""

    # Ollama (only needed when llm_provider = "local")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"

    # RAG / ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 6

    # Server
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_file": ".env"}


settings = Settings()
