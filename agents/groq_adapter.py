"""
VeritasAI Groq Cloud Integration Client.
Optimized for high-speed sub-second token delivery streams.
"""

import os
import asyncio
import time
from typing import Optional
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus

try:
    from groq import AsyncGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    AsyncGroq = None


class GroqAdapter(LLMAdapter):
    """Adapter for Groq's high-speed LLM API."""

    def __init__(self, model_id: str = "llama-3.3-70b-versatile"):
        """Initialize Groq adapter."""
        self.model_id = model_id
        api_key = os.getenv("GROQ_API_KEY", "")
        self.client = AsyncGroq(api_key=api_key) if (api_key and _GROQ_AVAILABLE) else None

    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """Make an async call to Groq API."""
        start = time.time()
        elapsed = lambda: int((time.time() - start) * 1000)

        if not _GROQ_AVAILABLE:
            return LLMResult(self.name, LLMStatus.SKIPPED, None, "missing_dep", "groq package not installed", elapsed())
        if not self.client or not os.getenv("GROQ_API_KEY"):
            return LLMResult(self.name, LLMStatus.SKIPPED, None, "missing_key", "Groq API key not configured", elapsed())

        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2
                    ),
                    timeout=30.0
                )
                text = response.choices[0].message.content
                if not text or len(text.strip().split()) < 20:
                    return LLMResult(self.name, LLMStatus.FAILED, None, "empty_response", "Response below threshold", elapsed())
                
                tokens = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
                return LLMResult(self.name, LLMStatus.SUCCESS, text.strip(), None, None, elapsed(), tokens)

            except asyncio.TimeoutError:
                return LLMResult(self.name, LLMStatus.FAILED, None, "timeout", "Request timed out", elapsed())
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ['401', '403', 'auth']):
                    return LLMResult(self.name, LLMStatus.FAILED, None, "auth", f"Auth failed: {str(e)[:100]}", elapsed())
                if any(x in err_str for x in ['429', 'rate limit']) and attempt == 0:
                    await asyncio.sleep(1)
                    continue
                if any(x in err_str for x in ['500', '502', '503']) and attempt == 0:
                    await asyncio.sleep(2)
                    continue
                return LLMResult(self.name, LLMStatus.FAILED, None, type(e).__name__, str(e)[:100], elapsed())

        return LLMResult(self.name, LLMStatus.FAILED, None, "max_retries", "Max retries exhausted", elapsed())

    def supports_images(self) -> bool:
        """Groq does not support images."""
        return False

    def get_model_id(self) -> str:
        """Get the model ID."""
        return self.model_id

    @property
    def name(self) -> str:
        """Get display name."""
        return "Groq"