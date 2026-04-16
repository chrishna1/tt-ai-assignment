"""
Centralized configuration for the LangGraph agent.

All settings have defaults that can be overridden via environment variables.
"""
from dataclasses import dataclass, field
import os


@dataclass
class Configuration:
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5")))
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
