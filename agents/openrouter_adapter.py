"""
VeritasAI OpenRouter Integration Client.
Access to premium open-source models through OpenRouter's unified API.
"""

import os
import asyncio
import time
from typing import Optional
from openai import AsyncOpenAI
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus


class OpenRouterAdapter(LLMAdapter):
    """Adapter for OpenRouter API (OpenAI-compatible interface)."""

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize OpenRouter adapter with fallback model selection.
        
        Args:
            model_id: The OpenRouter model identifier to use (default: deepseek-r1:free, falls back to llama-4-maverick:free)
        """
        self.primary_model = "deepseek/deepseek-r1:free"
        self.fallback_model = "meta-llama/llama-4-maverick:free"
        self.model_id = model_id if model_id else self.primary_model
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        ) if api_key else None

    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """
        Make an async call to OpenRouter API with fallback logic.
        
        Args:
            prompt: The text prompt to send to the model
            image_b64: Optional base64-encoded image data (not supported)
            
        Returns:
            LLMResult object containing the response or error details
        """
        start = time.time()
        elapsed = lambda: int((time.time() - start) * 1000)

        if not self.client or not os.getenv("OPENROUTER_API_KEY"):
            return LLMResult(self.name, LLMStatus.SKIPPED, None, "missing_key", "OpenRouter API key not configured", elapsed())

        models_to_try = [self.model_id, self.fallback_model] if self.model_id != self.fallback_model else [self.primary_model, self.fallback_model]

        for model in models_to_try:
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    ),
                    timeout=45.0
                )
                
                text = response.choices[0].message.content if response.choices else None
                if not text or len(text.strip().split()) < 20:
                    return LLMResult(self.name, LLMStatus.FAILED, None, "empty_response", "Response below threshold", elapsed())
                
                tokens = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
                return LLMResult(self.name, LLMStatus.SUCCESS, text.strip(), None, None, elapsed(), tokens)

            except asyncio.TimeoutError:
                return LLMResult(self.name, LLMStatus.FAILED, None, "timeout", "Request timed out", elapsed())
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ['401', '403', 'auth', 'unauthorized']):
                    return LLMResult(self.name, LLMStatus.FAILED, None, "auth", f"Auth failed: {str(e)[:100]}", elapsed())
                if any(x in err_str for x in ['429', 'rate limit']):
                    continue
                if any(x in err_str for x in ['500', '502', '503']):
                    if model == self.primary_model:
                        continue
                    return LLMResult(self.name, LLMStatus.FAILED, None, "server_error", f"Server error: {str(e)[:100]}", elapsed())
                if any(x in err_str for x in ['model not found', '404']) and model == self.primary_model:
                    continue
                return LLMResult(self.name, LLMStatus.FAILED, None, type(e).__name__, str(e)[:100], elapsed())

        return LLMResult(self.name, LLMStatus.FAILED, None, "max_retries", "All models exhausted", elapsed())

    def supports_images(self) -> bool:
        """
        Determine if this adapter supports image inputs.
        
        Returns:
            False, as OpenRouter does not support image analysis in this context
        """
        return False

    def get_model_id(self) -> str:
        """
        Get the model ID.
        
        Returns:
            The currently active OpenRouter model identifier string
        """
        return self.model_id

    @property
    def name(self) -> str:
        """
        Get display name.
        
        Returns:
            Human-readable adapter name
        """
        return "OpenRouter"