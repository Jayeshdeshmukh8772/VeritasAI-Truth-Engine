"""
VeritasAI Unified LangChain Adapter.
Consolidates all LLM API providers (Groq, Gemini, Mistral, OpenRouter, Cerebras, Nvidia NIM, Cohere)
into a single class using LangChain's ChatOpenAI abstraction.
"""

import os
import time
import asyncio
from typing import Optional
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus

try:
    from langchain_community.chat_models.openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None



class LangChainAdapter(LLMAdapter):
    """Unified adapter utilizing LangChain's OpenAI-compatible chat model."""

    def __init__(self, name: str, base_url: str, model_id: str, api_key_env: str, supports_img: bool = False):
        self._name = name
        self.base_url = base_url
        self.model_id = model_id
        self.api_key_env = api_key_env
        self._supports_images = supports_img
        
        # Initialize client lazily
        self.llm = None

    def _init_client(self) -> bool:
        if not _LANGCHAIN_AVAILABLE:
            return False
            
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            return False
            
        # Try importing streamlit dynamically to avoid circular issues
        try:
            import streamlit as st
            # Get dynamically adjusted UI sampling parameters
            temp = st.session_state.get("sampling_temperature", 0.2)
            top_p = st.session_state.get("sampling_top_p", 0.9)
            freq_pen = st.session_state.get("sampling_frequency_penalty", 0.0)
            pres_pen = st.session_state.get("sampling_presence_penalty", 0.0)
        except Exception:
            temp = 0.2
            top_p = 0.9
            freq_pen = 0.0
            pres_pen = 0.0
        
        model_kwargs = {"top_p": top_p}
        if "gemini" not in self._name.lower():
            model_kwargs["frequency_penalty"] = freq_pen
            model_kwargs["presence_penalty"] = pres_pen

        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=self.base_url,
            model_name=self.model_id,
            temperature=temp,
            model_kwargs=model_kwargs,
            max_retries=1
        )
        return True

    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """Call the underlying LLM using LangChain."""
        start = time.time()
        elapsed = lambda: int((time.time() - start) * 1000)

        if not _LANGCHAIN_AVAILABLE:
            return LLMResult(self.name, LLMStatus.SKIPPED, None, "missing_dep", "langchain package not installed", elapsed())
            
        # Re-initialize to load dynamic sampling parameters from UI on each query
        if not self._init_client():
            return LLMResult(self.name, LLMStatus.SKIPPED, None, "missing_key", f"{self.api_key_env} key not configured", elapsed())

        for attempt in range(2):
            try:
                # Build message payload
                messages = [{"role": "user", "content": prompt}]
                
                # Execute async call via LangChain
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=30.0
                )
                
                text = response.content
                if not text or len(text.strip().split()) < 15:
                    return LLMResult(self.name, LLMStatus.FAILED, None, "empty_response", "Response below word count limit", elapsed())
                
                tokens = 0
                if response.response_metadata and "token_usage" in response.response_metadata:
                    usage = response.response_metadata["token_usage"]
                    tokens = usage.get("total_tokens", 0)
                    
                return LLMResult(self.name, LLMStatus.SUCCESS, text.strip(), None, None, elapsed(), tokens)

            except asyncio.TimeoutError:
                return LLMResult(self.name, LLMStatus.FAILED, None, "timeout", "Request timed out", elapsed())
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ['401', '403', 'auth']):
                    return LLMResult(self.name, LLMStatus.FAILED, None, "auth", f"Auth failed: {str(e)[:80]}", elapsed())
                if any(x in err_str for x in ['429', 'rate limit']) and attempt == 0:
                    await asyncio.sleep(1)
                    continue
                if any(x in err_str for x in ['500', '502', '503']) and attempt == 0:
                    await asyncio.sleep(2)
                    continue
                return LLMResult(self.name, LLMStatus.FAILED, None, type(e).__name__, str(e)[:100], elapsed())

        return LLMResult(self.name, LLMStatus.FAILED, None, "max_retries", "Max retries exhausted", elapsed())

    def supports_images(self) -> bool:
        return self._supports_images

    def get_model_id(self) -> str:
        return self.model_id

    @property
    def name(self) -> str:
        return self._name
