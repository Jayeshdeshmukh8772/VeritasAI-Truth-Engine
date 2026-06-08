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
            model_id: The OpenRouter model identifier to use
        """
        self.primary_model = "meta-llama/llama-3.3-70b-instruct:free"
        self.fallback_model = "google/gemma-4-31b-it:free"
        self.model_id = model_id if model_id else self.primary_model
        # Generate a unique display name from the model_id to avoid collisions
        # when multiple OpenRouterAdapter instances are used
        self._display_name = self._derive_display_name(self.model_id)
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

        models_to_try = [self.model_id]
        # Add fallback if the configured model isn't already the fallback
        if self.model_id != self.fallback_model:
            models_to_try.append(self.fallback_model)

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
                    # Try fallback model if response is too short
                    if model != models_to_try[-1]:
                        continue
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
                    if model != models_to_try[-1]:
                        continue
                    return LLMResult(self.name, LLMStatus.FAILED, None, "server_error", f"Server error: {str(e)[:100]}", elapsed())
                if any(x in err_str for x in ['model not found', '404', 'unavailable']) and model != models_to_try[-1]:
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

    @staticmethod
    def _derive_display_name(model_id: str) -> str:
        """
        Derive a unique human-readable name from the model_id.
        e.g. 'meta-llama/llama-3.3-70b-instruct:free' -> 'OpenRouter-Llama-3.3-70b-Instruct'
             'google/gemma-4-31b-it:free' -> 'OpenRouter-Gemma-4-31b-It'
        """
        if not model_id:
            return "OpenRouter"
        # Strip version/pricing suffix (':free', ':latest', etc.)
        clean = model_id.split(":")[0]
        # Take just the model name after '/'
        parts = clean.split("/")
        short = parts[-1] if len(parts) > 1 else parts[0]
        # Title case with hyphens
        readable = "-".join(word.capitalize() for word in short.replace("_", "-").split("-"))
        return f"OpenRouter-{readable}"

    @property
    def name(self) -> str:
        """
        Get unique display name for this OpenRouter instance.
        
        Returns:
            Human-readable adapter name unique to the model_id
        """
        return self._display_name