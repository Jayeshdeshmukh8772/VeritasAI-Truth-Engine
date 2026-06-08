"""
VeritasAI Cerebras Cloud Integration Client.
Uses Cerebras chat completions API for ultra-fast LLM inference.
"""

import os
import asyncio
import time
from typing import Optional

from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus


class CerebrasAdapter(LLMAdapter):
    """Adapter for Cerebras Cloud SDK using OpenAI-compatible chat completions."""

    def __init__(self, model_id: str = "gpt-oss-120b") -> None:
        """
        Initialize Cerebras adapter.

        Args:
            model_id: Cerebras model identifier (default: llama-3.3-70b)
        """
        self.model_id = model_id
        api_key = os.getenv("CEREBRAS_API_KEY", "")
        self._api_key = api_key
        self._client = None
        if api_key:
            try:
                from cerebras.cloud.sdk import AsyncCerebras
                self._client = AsyncCerebras(api_key=api_key)
            except ImportError:
                self._client = None

    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """
        Make an async call to Cerebras API using chat completions.

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
                "missing_key", "Cerebras API key not configured", elapsed()
            )

        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        temperature=0.2,
                    ),
                    timeout=20.0,
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
                    "timeout", "Request timed out after 20s", elapsed()
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
                if any(x in err_str for x in ["connection", "ssl", "network", "dns"]):
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
        """Cerebras does not support image inputs."""
        return False

    def get_model_id(self) -> str:
        """Return the Cerebras model identifier."""
        return self.model_id

    @property
    def name(self) -> str:
        """Human-readable display name."""
        return "Cerebras"