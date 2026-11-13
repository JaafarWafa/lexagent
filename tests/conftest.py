"""Tests never call a real LLM or download an embedding model."""
import os

os.environ["EMBEDDING_MODEL"] = ""  # env beats .env in pydantic-settings: BM25 only in tests
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9/v1"  # unreachable on purpose
os.environ["LLM_API_KEY"] = "test"
