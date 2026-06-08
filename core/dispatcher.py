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

    async def dispatch_all(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        callback=None,
        fast_mode: bool = False
    ) -> List[LLMResult]:
        """Fire parallel requests to all enabled adapters with speculative execution."""
        tasks = {}
        model_cfg_map = {cfg['name']: cfg for cfg in self.config.get('models', [])}
        results_dict = {}

        adapters_to_use = self.adapters[:3] if fast_mode else self.adapters

        for adapter in adapters_to_use:
            slug = self._get_slug(adapter.name)
            cfg = model_cfg_map.get(slug, {"enabled": True, "daily_limit": 100})
            
            if not cfg.get("enabled", True):
                results_dict[adapter.name] = await self._build_skipped(adapter.name, "disabled")
                if callback:
                    callback(adapter.name, results_dict[adapter.name])
                continue

            if not self.rate_tracker.check_quota(slug, cfg.get("daily_limit", 100)):
                results_dict[adapter.name] = await self._build_skipped(adapter.name, "quota_exceeded")
                if callback:
                    callback(adapter.name, results_dict[adapter.name])
                continue

            # Create an async task
            coro = self._execute_and_track(adapter, prompt, image_b64, slug)
            task = asyncio.create_task(coro)
            tasks[task] = adapter.name

        # For any models in full adapters list but NOT in adapters_to_use, mark as skipped
        for adapter in self.adapters:
            if adapter.name not in results_dict and adapter not in adapters_to_use:
                results_dict[adapter.name] = await self._build_skipped(adapter.name, "disabled")
                if callback:
                    callback(adapter.name, results_dict[adapter.name])

        if not tasks:
            return [results_dict[adapter.name] for adapter in self.adapters if adapter.name in results_dict]

        import time
        start_time = time.time()
        
        # Adaptive Percentile Cutoff configuration
        # If N >= 4 and 4 models complete within 1.5 seconds, trigger a 1-second warning countdown
        completed_count = 0
        speculative_deadline = None
        
        pending = set(tasks.keys())
        
        while pending:
            now = time.time()
            timeout = None
            if speculative_deadline is not None:
                timeout = max(0.0, speculative_deadline - now)
                if timeout == 0.0:
                    break
            
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout
            )
            
            if not done and speculative_deadline is not None and time.time() >= speculative_deadline:
                break
                
            for task in done:
                name = tasks[task]
                try:
                    result = task.result()
                    results_dict[name] = result
                    if result.status == LLMStatus.SUCCESS:
                        completed_count += 1
                except Exception as e:
                    results_dict[name] = LLMResult(
                        model=name,
                        status=LLMStatus.FAILED,
                        response=None,
                        error_type=type(e).__name__,
                        error_msg=str(e),
                        latency_ms=int((time.time() - start_time) * 1000)
                    )
                
                if callback:
                    callback(name, results_dict[name])

            # Trigger check: if 4 or more successful completions within 1.5s, trigger 1.0s warning deadline
            elapsed = time.time() - start_time
            if completed_count >= 4 and elapsed <= 1.5 and speculative_deadline is None:
                speculative_deadline = time.time() + 1.0

        # Cancel remaining pending tasks as stragglers
        for task in pending:
            task.cancel()
            name = tasks[task]
            results_dict[name] = LLMResult(
                model=name,
                status=LLMStatus.SKIPPED,
                response=None,
                error_type="latency",
                error_msg="Skipped due to latency (straggler termination)",
                latency_ms=int((time.time() - start_time) * 1000)
            )
            if callback:
                callback(name, results_dict[name])

        # Return results in the order of self.adapters
        ordered_results = []
        for adapter in self.adapters:
            if adapter.name in results_dict:
                ordered_results.append(results_dict[adapter.name])
            else:
                ordered_results.append(await self._build_skipped(adapter.name, "unknown"))
        return ordered_results

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