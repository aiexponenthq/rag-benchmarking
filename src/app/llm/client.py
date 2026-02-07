from __future__ import annotations

import json

import requests

from app.config.settings import get_settings


class LLMClient:
    """Simple LLM client supporting OpenAI and Gemini for text generation.

    Configuration is read from environment via `AppSettings`.
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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            return self._generate_openai(system_prompt, user_prompt)
        if self.provider == "gemini":
            if not self.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            return self._generate_gemini(system_prompt, user_prompt)
        # Fallback: echo user prompt for now
        return user_prompt

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
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
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            from app.exceptions import LLMError

            raise LLMError(f"OpenAI API error: {str(e)}") from e

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
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
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            from app.exceptions import LLMError

            raise LLMError(f"Gemini API error: {str(e)}") from e
