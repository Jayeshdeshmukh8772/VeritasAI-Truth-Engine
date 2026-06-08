"""
VeritasAI Cohere Integration Client.
Advanced language models for production-grade applications.
Compatible with Cohere SDK v7.x (uses message parameter, not messages).
"""

import os
import asyncio
import time
from typing import Optional
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus


class CohereAdapter(LLMAdapter):
    """Adapter for Cohere API (SDK v7.x)."""

    def __init__(self, model_id: str = "command-r-plus-08-2024"):
        """
        Initialize Cohere adapter.
        
        Args:
            model_id: The Cohere model identifier to use (default: command-r-plus)
        """
        self.model_id = model_id
        api_key = os.getenv("COHERE_API_KEY", "")
        self.client = None
        if api_key:
            try:
                from cohere import AsyncClient
                self.client = AsyncClient(api_key=api_key)
            except ImportError:
                pass

    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """
        Make an async call to Cohere API.
        
        Cohere SDK v7.x uses `message` (str) not `messages` (list).
        
        Args:
            prompt: The text prompt to send to the model
            image_b64: Optional base64-encoded image data (not supported)
            
        Returns:
            LLMResult object containing the response or error details
        """
        start = time.time()
        elapsed = lambda: int((time.time() - start) * 1000)

        if not self.client or not os.getenv("COHERE_API_KEY"):
            return LLMResult(self.name, LLMStatus.SKIPPED, None, "missing_key", "Cohere API key not configured", elapsed())

        try:
            response = await asyncio.wait_for(
                self.client.chat(
                    message=prompt,
                    model=self.model_id,
                    temperature=0.7
                ),
                timeout=60.0
            )
            
            text = response.text if hasattr(response, 'text') else None
            if not text or len(text.strip().split()) < 20:
                return LLMResult(self.name, LLMStatus.FAILED, None, "empty_response", "Response below threshold", elapsed())
            
            # Extract token count — v7.x uses meta.tokens
            tokens = 0
            if hasattr(response, 'meta') and response.meta:
                if hasattr(response.meta, 'tokens') and response.meta.tokens:
                    tokens = getattr(response.meta.tokens, 'output_tokens', 0) or 0
            
            return LLMResult(self.name, LLMStatus.SUCCESS, text.strip(), None, None, elapsed(), tokens)

        except asyncio.TimeoutError:
            return LLMResult(self.name, LLMStatus.FAILED, None, "timeout", "Request timed out", elapsed())
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ['401', '403', 'auth', 'unauthorized']):
                return LLMResult(self.name, LLMStatus.FAILED, None, "auth", f"Auth failed: {str(e)[:100]}", elapsed())
            if any(x in err_str for x in ['429', 'rate limit']):
                return LLMResult(self.name, LLMStatus.FAILED, None, "rate_limit", f"Rate limit exceeded: {str(e)[:100]}", elapsed())
            if any(x in err_str for x in ['500', '502', '503']):
                return LLMResult(self.name, LLMStatus.FAILED, None, "server_error", f"Server error: {str(e)[:100]}", elapsed())
            return LLMResult(self.name, LLMStatus.FAILED, None, type(e).__name__, str(e)[:100], elapsed())

    def supports_images(self) -> bool:
        """
        Determine if this adapter supports image inputs.
        
        Returns:
            False, as Cohere does not support image analysis in this context
        """
        return False

    def get_model_id(self) -> str:
        """
        Get the model ID.
        
        Returns:
            The Cohere model identifier string
        """
        return self.model_id

    @property
    def name(self) -> str:
        """
        Get display name.
        
        Returns:
            Human-readable adapter name
        """
        return "Cohere"
