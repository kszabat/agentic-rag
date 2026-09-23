from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from google.genai.errors import ServerError
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agentic_rag.config import get_settings

logger = logging.getLogger(__name__)

_RETRY_KWARGS: dict[str, Any] = {
    "retry": retry_if_exception_type(ServerError),
    "stop": stop_after_attempt(2),
    "wait": wait_exponential_jitter(initial=2, max=20),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True,
}


@lru_cache
def get_llm() -> GoogleGenAI:
    settings = get_settings().llm
    if settings.provider != "gemini":
        raise NotImplementedError(
            f"Provider {settings.provider} is not supported yet. Please use 'gemini' as the provider."
        )
    return GoogleGenAI(model=settings.model_name, api_key=settings.api_key)


@retry(**_RETRY_KWARGS)
async def achat_with_retry(messages: list[ChatMessage]) -> ChatResponse:
    return await get_llm().achat(messages)


@retry(**_RETRY_KWARGS)
async def astructured_predict_with_retry(
    output_cls: type[BaseModel], prompt: PromptTemplate, **kwargs: Any
) -> BaseModel:
    return await get_llm().astructured_predict(output_cls, prompt, **kwargs)
