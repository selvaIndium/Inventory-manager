"""Diskcache wrapper used by knowledge graph query tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from diskcache import Cache


class CacheLayer:
    """Simple cache interface around diskcache with TTL helpers."""

    def __init__(self, cache_dir: str = ".cache/inventory-agent") -> None:
        """Initialize the cache store.

        Args:
            cache_dir: Directory path for local diskcache storage.
        """
        self._cache = Cache(str(Path(cache_dir)))

    def get(self, key: str) -> Tuple[bool, Optional[Any], Optional[int]]:
        """Get a value and its remaining TTL.

        Returns:
            Tuple of (hit, value, ttl_remaining_seconds).
        """
        if key not in self._cache:
            return False, None, None
        value, expire_time = self._cache.get(key, default=(None, None), expire_time=True)
        if expire_time is None:
            return True, value, None
        ttl_remaining = max(0, int(expire_time - __import__("time").time()))
        return True, value, ttl_remaining

    def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Set a value with TTL and return success status."""
        return bool(self._cache.set(key, value, expire=ttl_seconds))

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


CACHE = CacheLayer()
