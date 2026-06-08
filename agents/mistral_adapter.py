"""
VeritasAI Mistral AI Integration Client.
Uses OpenAI-compatible endpoint for reliability across SDK versions.
Mistral's API is OpenAI-compatible at https://api.mistral.ai/v1.
"""

import os
import asyncio
import time
from typing import Optional

from openai import AsyncOpenAI
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus


class MistralAdapter(LLMAdapter):
    """Adapter for Mistral AI API using OpenAI-compatible interface."""

    def __init__(self, model_id: str = "mistral-small-latest") -> None:
        """
        Initialize Mistral adapter using OpenAI-compatible client.

        Args:
            model_id: The Mistral model identifier (default: mistral-small-latest)
        """
        self.model_id = model_id
        api_key = os.getenv("MISTRAL_API_KEY", "")
        self._api_key = api_key
        self._client = None
        if api_key:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
            )

    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """
        Make an async call to Mistral API via OpenAI-compatible endpoint.

        Args:
            prompt: The text prompt to send to the model
            image_b64: Not supported — ignored

        Returns:
            LLMResult with response text or error details
        """
        start = time.time()
        elapsed = lambda: int((time.time() - start) * 1000)

        if not self._client or not self._api_key:
            return LLMResult(
                self.name, LLMStatus.SKIPPED, None,
                "missing_key", "Mistral API key not configured", elapsed()
            )

        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=2048,
                    ),
                    timeout=40.0,
                )

                text = response.choices[0].message.content if response.choices else None
                if not text or len(text.strip().split()) < 20:
                    return LLMResult(
                        self.name, LLMStatus.FAILED, None,
                        "empty_response", "Response below minimum word threshold", elapsed()
                    )

                tokens = getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0
                return LLMResult(self.name, LLMStatus.SUCCESS, text.strip(), None, None, elapsed(), tokens)

            except asyncio.TimeoutError:
                return LLMResult(
                    self.name, LLMStatus.FAILED, None,
                    "timeout", "Request timed out after 40s", elapsed()
                )
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["401", "403", "auth", "unauthorized", "invalid api"]):
                    return LLMResult(
                        self.name, LLMStatus.FAILED, None,
                        "auth", f"Authentication failed: {str(e)[:100]}", elapsed()
                    )
                if any(x in err_str for x in ["429", "rate limit", "quota"]):
                    if attempt == 0:
                        await asyncio.sleep(2)
                        continue
                    return LLMResult(
                        self.name, LLMStatus.FAILED, None,
                        "rate_limit", "Rate limit exceeded", elapsed()
                    )
                if any(x in err_str for x in ["500", "502", "503", "504"]):
                    if attempt == 0:
                        await asyncio.sleep(2)
                        continue
                    return LLMResult(
                        self.name, LLMStatus.FAILED, None,
                        "server_error", f"Server error: {str(e)[:100]}", elapsed()
                    )
                if any(x in err_str for x in ["connection", "ssl", "network"]):
                    return LLMResult(
                        self.name, LLMStatus.FAILED, None,
                        "network", f"Network error: {str(e)[:100]}", elapsed()
                    )
                return LLMResult(
                    self.name, LLMStatus.FAILED, None,
                    type(e).__name__, str(e)[:150], elapsed()
                )

        return LLMResult(
            self.name, LLMStatus.FAILED, None,
            "max_retries", "Max retries exhausted", elapsed()
        )

    def supports_images(self) -> bool:
        """Mistral does not support image inputs in this adapter."""
        return False

    def get_model_id(self) -> str:
        """Return the Mistral model identifier."""
        return self.model_id

    @property
    def name(self) -> str:
        """Human-readable display name."""
        return "Mistral"