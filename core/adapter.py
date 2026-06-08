"""
VeritasAI LLM Adapter Interface.
Abstract base class that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
from core.result import LLMResult


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    All adapters must implement this interface.
    """

    @abstractmethod
    async def call(self, prompt: str, image_b64: Optional[str] = None) -> LLMResult:
        """
        Make a single API call to the LLM provider.
        Never raises exceptions; always returns LLMResult with error info.
        
        Args:
            prompt: The text prompt to send to the model
            image_b64: Optional base64-encoded image data
            
        Returns:
            LLMResult object containing the response or error details
        """
        pass

    @abstractmethod
    def supports_images(self) -> bool:
        """
        Determine if this adapter can handle image inputs.
        
        Returns:
            True if the adapter supports image processing, False otherwise
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """
        Get the exact model identifier string used for API calls.
        
        Returns:
            Model ID string (e.g., 'llama-3.3-70b-versatile')
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get a human-readable name for display in the UI.
        
        Returns:
            Display name (e.g., 'Groq (Llama-3.3-70B)')
        """
        pass
