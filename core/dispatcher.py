"""
VeritasAI Parallel Workload Orchestrator.
Fires concurrent request matrices to available provider networks.
"""

import asyncio
from typing import List, Optional
from core.adapter import LLMAdapter
from core.result import LLMResult, LLMStatus
from core.rate_tracker import RateTracker


class AsyncDispatcher:
    """Dispatch queries to multiple LLM adapters in parallel."""

    def __init__(self, adapters: List[LLMAdapter], rate_tracker: RateTracker, config: dict):
        """Initialize dispatcher."""
        self.adapters = adapters
        self.rate_tracker = rate_tracker
        self.config = config

    async def dispatch_all(self, prompt: str, image_b64: Optional[str] = None) -> List[LLMResult]:
        """Fire parallel requests to all enabled adapters."""
        tasks = []
        model_cfg_map = {cfg['name']: cfg for cfg in self.config.get('models', [])}

        for adapter in self.adapters:
            # Map adapter name to config slug
            slug = self._get_slug(adapter.name)
            cfg = model_cfg_map.get(slug, {"enabled": True, "daily_limit": 100})
            
            if not cfg.get("enabled", True):
                tasks.append(self._build_skipped(adapter.name, "disabled"))
                continue

            if not self.rate_tracker.check_quota(slug, cfg.get("daily_limit", 100)):
                tasks.append(self._build_skipped(adapter.name, "quota_exceeded"))
                continue

            tasks.append(self._execute_and_track(adapter, prompt, image_b64, slug))

        return await asyncio.gather(*tasks)

    def _get_slug(self, name: str) -> str:
        """Get config slug from adapter name."""
        name_lower = name.lower()
        if "groq" in name_lower:
            return "groq"
        elif "gemini" in name_lower:
            return "gemini"
        elif "cerebras" in name_lower:
            return "cerebras"
        elif "mistral" in name_lower:
            return "mistral"
        elif "nvidia" in name_lower:
            return "nvidia_nim"
        elif "deepseek" in name_lower:
            return "openrouter_deepseek"
        elif "llama" in name_lower:
            return "openrouter_llama4"
        return "cohere"

    async def _execute_and_track(self, adapter: LLMAdapter, prompt: str, image_b64: Optional[str], slug: str) -> LLMResult:
        """Execute adapter call and track usage."""
        result = await adapter.call(prompt, image_b64)
        if result.status == LLMStatus.SUCCESS:
            self.rate_tracker.increment(slug)
        return result

    async def _build_skipped(self, name: str, error_type: str) -> LLMResult:
        """Build a skipped result."""
        return LLMResult(name, LLMStatus.SKIPPED, None, error_type, "Model skipped", 0)