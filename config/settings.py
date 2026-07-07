"""Application settings loaded from YAML with environment variable overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
_DEFAULT_YAML = _CONFIG_DIR / "settings.yaml"


def _load_yaml_defaults() -> dict[str, Any]:
    """Load default values from settings.yaml."""
    if not _DEFAULT_YAML.exists():
        return {}
    try:
        with _DEFAULT_YAML.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to load settings from {_DEFAULT_YAML}") from exc


class Settings(BaseSettings):
    """Merged configuration: YAML defaults overridden by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_pages_per_query: int = 35
    max_crawl_depth: int = 3
    max_iterations: int = 5
    confidence_threshold: float = 0.70
    top_k: int = 10
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 40
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieve_multiplier: int = 3
    requests_per_second_per_domain: float = 1.0
    min_content_length: int = 200
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free",
        alias="OPENROUTER_MODEL",
    )
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="GROQ_MODEL",
    )
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    chroma_persist_dir: str = Field(
        default="./data/vectorstore",
        alias="CHROMA_PERSIST_DIR",
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    user_agent: str = Field(
        default="AgenticCrawlerUniProject/1.0 (Educational IR coursework project; contact: f.zardodkhonova@studenti.unina.it)",
        alias="USER_AGENT",
    )
    log_level: str = "INFO"
    log_file: str = "logs/agentic_crawler.log"
    http_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0

    def __init__(self, **data: Any) -> None:
        yaml_defaults = _load_yaml_defaults()
        merged = {**yaml_defaults, **data}
        super().__init__(**merged)

    @property
    def chroma_path(self) -> Path:
        """Resolved absolute path to the ChromaDB persistence directory."""
        path = Path(self.chroma_persist_dir)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path

    @property
    def log_path(self) -> Path:
        """Resolved absolute path to the log file."""
        path = Path(self.log_file)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path


def get_settings() -> Settings:
    """Return a cached-ready Settings instance."""
    return Settings()