"""
VeritasAI Google Gemini Integration Client.
Supports image-based analysis with advanced vision capabilities.
"""

import os
import asyncio
import time
import base64
from typing import Optional
import google.generativeai as genai
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus


class GeminiAdapter(LLMAdapter):
    """Adapter for Google Gemini API with image support."""

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        """
        Initialize Gemini adapter.
        
        Args:
            model_id: The Gemini model identifier to use (default: gemini-2.5-flash)
        """
        self.model_id = model_id
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)

    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """
        Make an async call to Gemini API.
        
        Args:
            prompt: The text prompt to send to the model
            image_b64: Optional base64-encoded image data
            
        Returns:
            LLMResult object containing the response or error details
        """
        start = time.time()
        elapsed = lambda: int((time.time() - start) * 1000)

        if not os.getenv("GEMINI_API_KEY"):
            return LLMResult(self.name, LLMStatus.SKIPPED, None, "missing_key", "Gemini API key not configured", elapsed())

        for attempt in range(2):
            try:
                model = genai.GenerativeModel(self.model_id)
                content_parts = []
                
                if image_b64:
                    try:
                        image_data = base64.b64decode(image_b64)
                        content_parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64
                            }
                        })
                    except Exception as e:
                        return LLMResult(self.name, LLMStatus.FAILED, None, "invalid_image", f"Image decode error: {str(e)[:100]}", elapsed())
                
                content_parts.append(prompt)
                
                response = await asyncio.wait_for(
                    asyncio.to_thread(model.generate_content, content_parts),
                    timeout=30.0
                )
                
                text = response.text
                if not text or len(text.strip().split()) < 20:
                    return LLMResult(self.name, LLMStatus.FAILED, None, "empty_response", "Response below threshold", elapsed())
                
                return LLMResult(self.name, LLMStatus.SUCCESS, text.strip(), None, None, elapsed())

            except asyncio.TimeoutError:
                return LLMResult(self.name, LLMStatus.FAILED, None, "timeout", "Request timed out", elapsed())
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ['401', '403', 'auth', 'permission']):
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
        """
        Determine if this adapter supports image inputs.
        
        Returns:
            True, as Gemini supports image-based analysis
        """
        return True

    def get_model_id(self) -> str:
        """
        Get the model ID.
        
        Returns:
            The Gemini model identifier string
        """
        return self.model_id

    @property
    def name(self) -> str:
        """
        Get display name.
        
        Returns:
            Human-readable adapter name
        """
        return "Gemini"