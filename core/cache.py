"""
VeritasAI Response Caching Engine.
Session-level SHA-256 keyed cache to avoid redundant API calls for identical queries.
Integrates with st.session_state for Streamlit persistence.
"""

import hashlib
from typing import Optional
from core.result import FinalOutput


class ResponseCache:
    """
    In-memory query response cache backed by a dict in session state.
    Key: SHA-256(enhanced_query). Value: FinalOutput dataclass.
    """

    def __init__(self) -> None:
        """Initialize the internal cache dictionary."""
        self._store: dict[str, FinalOutput] = {}

    def compute_signature(self, query: str) -> str:
        """
        Compute SHA-256 hash of a query string for use as cache key.

        Args:
            query: The enhanced query string to hash

        Returns:
            64-character hex SHA-256 digest
        """
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    def lookup(self, query: str) -> Optional[FinalOutput]:
        """
        Retrieve a cached FinalOutput for the given query, if it exists.

        Args:
            query: The enhanced query string (will be hashed internally)

        Returns:
            FinalOutput if cache hit, None if cache miss
        """
        key = self.compute_signature(query)
        return self._store.get(key)

    def set(self, query: str, result: FinalOutput) -> None:
        """
        Store a FinalOutput in the cache under the query's hash key.

        Args:
            query: The enhanced query string (will be hashed internally)
            result: The FinalOutput to cache
        """
        key = self.compute_signature(query)
        self._store[key] = result

    def clear(self) -> None:
        """Clear all entries from the session cache."""
        self._store.clear()

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._store)

    def contains(self, query: str) -> bool:
        """
        Check if a query result is cached without retrieving it.

        Args:
            query: The enhanced query string

        Returns:
            True if cached, False otherwise
        """
        key = self.compute_signature(query)
        return key in self._store