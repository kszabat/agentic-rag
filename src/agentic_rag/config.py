from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RerankerMode(StrEnum):
    LOCAL = "local"
    API = "api"


class LLMSettings(BaseModel):
    provider: str = "gemini"
    model_name: str
    api_key: str


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None


class TextEmbeddingSettings(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda"
    vector_size: int = 1024


class ColPaliSettings(BaseModel):
    model_name: str = "vidore/colqwen2.5-v0.2"
    device: str = "cuda"
    pdf_render_dpi: int = 150
    vector_size: int = 128


class RerankerSettings(BaseModel):
    mode: RerankerMode = RerankerMode.LOCAL
    model_name: str = "jinaai/jina-reranker-m0"
    device: str = "cuda"
    top_n: int = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    hf_token: str | None = None
    llm: LLMSettings = Field(
        default_factory=lambda: LLMSettings(
            model_name="gemini-3-flash-preview", api_key=""
        )
    )
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    text_embedding: TextEmbeddingSettings = Field(default_factory=TextEmbeddingSettings)
    colpali: ColPaliSettings = Field(default_factory=ColPaliSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)

    data_dir: Path = PROJECT_ROOT / "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
