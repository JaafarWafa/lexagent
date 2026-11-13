from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from environment or .env. Any OpenAI-compatible endpoint works."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    # Multilingual embeddings for hybrid retrieval (fastembed, ONNX, CPU). Empty = BM25 only.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rrf_k: int = 60
    # Answer budget per LLM call. Reasoning models (gpt-oss...) spend tokens thinking before
    # answering; Groq's default cap (2048) can leave an empty answer on long contracts.
    # Groq counts this budget against the per-minute limit, so it cannot be too large either.
    llm_max_tokens: int = 4096  # prompt + this must stay under the free tier's 8k tokens/min
    # Optional Langfuse tracing (https://langfuse.com): leave keys empty to disable.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
