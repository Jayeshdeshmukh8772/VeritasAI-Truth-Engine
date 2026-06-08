"""Quick full adapter diagnostic — tests all 8 adapters."""
import asyncio, os, sys, time
from dotenv import load_dotenv
load_dotenv()

prompt = "What is the capital of France? Provide a detailed explanation."

async def main():
    from agents.groq_adapter import GroqAdapter
    from agents.gemini_adapter import GeminiAdapter
    from agents.cerebras_adapter import CerebrasAdapter
    from agents.mistral_adapter import MistralAdapter
    from agents.nvidia_nim_adapter import NvidiaNimAdapter
    from agents.openrouter_adapter import OpenRouterAdapter
    from agents.cohere_adapter import CohereAdapter

    adapters = [
        GroqAdapter(),
        GeminiAdapter(),
        CerebrasAdapter(),
        MistralAdapter(),
        NvidiaNimAdapter(),
        OpenRouterAdapter(model_id="meta-llama/llama-3.3-70b-instruct:free"),
        OpenRouterAdapter(model_id="google/gemma-4-31b-it:free"),
        CohereAdapter(),
    ]

    print(f"{'Adapter':<30} {'Status':<10} {'Latency':<10} {'Error'}")
    print("-" * 90)

    for a in adapters:
        try:
            r = await a.call(prompt)
            err = r.error_type or r.error_msg or ""
            if err:
                err = f"{r.error_type}: {(r.error_msg or '')[:80]}"
            print(f"{a.name:<30} {r.status.value:<10} {r.latency_ms or 0:<10} {err}")
        except Exception as e:
            print(f"{a.name:<30} {'CRASH':<10} {'N/A':<10} {type(e).__name__}: {str(e)[:60]}")

    print()

asyncio.run(main())
