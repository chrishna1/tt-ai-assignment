"""
Centralized configuration for the LangGraph agent.

All settings have defaults that can be overridden via environment variables
or at runtime via LangGraph's RunnableConfig configurable dict.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

T = TypeVar("T", bound="Configuration")


@dataclass(kw_only=True)
class Configuration:
    llm_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
        metadata={"description": "LLM model for synthesis and query rewriting."},
    )
    embedding_model: Annotated[
        str, {"__template_metadata__": {"kind": "embeddings"}}
    ] = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        metadata={"description": "OpenAI embedding model name."},
    )
    retriever_provider: Annotated[
        Literal["chroma"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="chroma",
        metadata={
            "description": "Vector store backend. Currently only 'chroma' is supported."
        },
    )
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5")),
        metadata={"description": "Number of chunks to retrieve per query."},
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        metadata={"description": "Directory for ChromaDB persistent storage."},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Unpack LangGraph's RunnableConfig into a Configuration instance."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
