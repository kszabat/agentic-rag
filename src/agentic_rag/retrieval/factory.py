from __future__ import annotations

from functools import lru_cache

from llama_index.llms.google_genai import GoogleGenAI

from agentic_rag.config import get_settings


@lru_cache
def get_llm() -> GoogleGenAI:
    settings = get_settings().llm
    if settings.provider != "gemini":
        raise NotImplementedError(
            f"Provider {settings.provider} is not supported yet. Please use 'gemini' as the provider."
        )
    return GoogleGenAI(model=settings.model_name, api_key=settings.api_key)
