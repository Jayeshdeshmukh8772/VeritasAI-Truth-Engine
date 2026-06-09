"""
VeritasAI Web Search Utility.
Queries DuckDuckGo via the 'duckduckgo-search' library to fetch live web snippets.
Provides real-time grounding context for the LLM adapters.
"""

import logging

logger = logging.getLogger("VeritasAI")


def search_web(query: str, max_results: int = 3) -> str:
    """
    Perform a live search query and format results into a structured text snippet.

    Args:
        query: The search query string
        max_results: Max number of snippets to return (default: 3)

    Returns:
        Structured context string with titles, sources, and snippets, or empty string on error.
    """
    try:
        # duckduckgo-search >= 6.x uses: from duckduckgo_search import DDGS
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return ""

            context_parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                href = r.get("href", "#")
                body = r.get("body", "")
                context_parts.append(
                    f"Result [{i}]:\n"
                    f"Title: {title}\n"
                    f"Source: {href}\n"
                    f"Content: {body}\n"
                )

            return "\n---\n".join(context_parts)

    except ImportError:
        logger.warning("duckduckgo-search not installed. Web search disabled.", extra={"context": "web_search"})
        return ""
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}", extra={"context": "web_search"})
        return ""
