from __future__ import annotations

import json
from dataclasses import dataclass, field

import requests

from app.config.settings import get_settings


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """Container for LLM text + token accounting."""

    text: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    # Keep .content as alias so existing callers that do resp.content keep working
    @property
    def content(self) -> str:
        return self.text


class LLMClient:
    """Simple LLM client supporting OpenAI and Gemini for text generation.

    ``generate()`` now returns an :class:`LLMResponse` so callers can access
    both the generated text and token counts.  For backward compatibility the
    convenience method ``generate_text()`` returns the plain ``str``.

    Configuration is read from environment via ``AppSettings``.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.provider = (settings.llm_provider or "").lower()
        self.openai_model = settings.openai_model or "gpt-4o-mini"
        self.gemini_model = settings.gemini_model or "gemini-1.5-flash"
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.openai_api_key = settings.openai_api_key
        self.gemini_api_key = settings.gemini_api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return generated text (str).

        This preserves backward compatibility for all existing callers such as
        ``RAGEngine._call_llm`` which expect a plain string.  Token usage is
        stored on ``self._last_token_usage`` so the API layer can read it after
        the call without changing the return type.
        """
        resp = self._generate_response(system_prompt, user_prompt)
        self._last_token_usage: dict[str, int] = resp.token_usage.as_dict()
        return resp.text

    def generate_with_usage(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return :class:`LLMResponse` with text *and* token_usage populated."""
        return self._generate_response(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _generate_response(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if self.provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            return self._generate_openai(system_prompt, user_prompt)
        if self.provider == "gemini":
            if not self.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            return self._generate_gemini(system_prompt, user_prompt)
        # Fallback: echo user prompt (no token data)
        return LLMResponse(text=user_prompt)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]

            # OpenAI usage object: {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
            usage_raw: dict[str, int] = data.get("usage") or {}
            token_usage = TokenUsage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            )
            return LLMResponse(text=text, token_usage=token_usage)
        except Exception as e:
            from app.exceptions import LLMError

            raise LLMError(f"OpenAI API error: {str(e)}") from e

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        # Gemini v1beta generateContent endpoint
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
            f"?key={self.gemini_api_key}"
        )
        headers = {"Content-Type": "application/json"}
        contents = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ]
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # Safety checks for empty response
            if "candidates" not in data or not data["candidates"]:
                if "promptFeedback" in data:
                    raise ValueError(f"Blocked by safety settings: {data['promptFeedback']}")
                raise ValueError("No candidates returned")

            text = data["candidates"][0]["content"]["parts"][0]["text"]

            # Gemini usageMetadata: {"promptTokenCount": N, "candidatesTokenCount": N, "totalTokenCount": N}
            usage_raw: dict[str, int] = data.get("usageMetadata") or {}
            token_usage = TokenUsage(
                prompt_tokens=usage_raw.get("promptTokenCount", 0),
                completion_tokens=usage_raw.get("candidatesTokenCount", 0),
                total_tokens=usage_raw.get("totalTokenCount", 0),
            )
            return LLMResponse(text=text, token_usage=token_usage)
        except Exception as e:
            from app.exceptions import LLMError

            raise LLMError(f"Gemini API error: {str(e)}") from e
