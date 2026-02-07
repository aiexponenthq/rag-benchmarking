from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Central application settings loaded from environment and optional .env file.

    Attributes
    ----------
    app_env: str
        Application environment (e.g., dev, prod).
    log_level: str
        Logging level string (e.g., INFO, DEBUG).
    openai_api_key: Optional[str]
        API key for OpenAI (optional).
    gemini_api_key: Optional[str]
        API key for Google Gemini (optional).
    qdrant_url: Optional[str]
        Qdrant Cloud URL.
    qdrant_api_key: Optional[str]
        Qdrant Cloud API key.
    qdrant_collection: str
        Default Qdrant collection name.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    api_key: str | None = Field(default=None, alias="API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    llm_provider: str | None = Field(default=None, alias="LLM_PROVIDER")
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")
    gemini_model: str | None = Field(default=None, alias="GEMINI_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")

    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="agentic_rag_poc", alias="QDRANT_COLLECTION")
    # Self-check configuration
    self_check_min_groundedness: float = Field(default=0.7, alias="SELF_CHECK_MIN_GROUNDEDNESS")
    self_check_retry: bool = Field(default=True, alias="SELF_CHECK_RETRY")

    # Prompts
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Answer based only on the provided context. Cite sources."
        ),
        alias="SYSTEM_PROMPT",
    )
    user_prompt_template: str = Field(
        default="Context:\n{context_blocks}\n\nQuestion: {query}\nAnswer:",
        alias="USER_PROMPT_TEMPLATE",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached instance of application settings.

    Returns
    -------
    AppSettings
        Loaded settings object.
    """

    return AppSettings()  # type: ignore[call-arg]
