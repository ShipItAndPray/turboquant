"""
TurboQuant Memcached Adapter
=============================
Transparent vector compression for Memcached.

Requirements: pip install pymemcache

Usage:
    from pymemcache.client.base import Client
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.memcached import MemcachedTurboCache

    mc = Client('localhost:11211')
    encoder = TurboQuantEncoder(dim=768)
    cache = MemcachedTurboCache(encoder, mc)

    cache.put("doc:1", vector, ttl=3600)
    vec = cache.get("doc:1")
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class MemcachedTurboCache(BaseTurboAdapter):
    """
    Memcached adapter with TurboQuant compression.

    Features:
    - get_multi/set_multi for batch operations
    - TTL support
    - Key prefix namespace isolation
    - CAS (check-and-set) support for atomic updates

    Note: Memcached has a 1MB value limit. TurboQuant compressed vectors
    are typically <1KB, so this is never an issue.
    """

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 prefix: str = "tq:", ttl: Optional[int] = None):
        """
        Args:
            client: pymemcache.client.base.Client or compatible
            prefix: Key prefix for namespace isolation
            ttl: Default TTL in seconds (0 = no expiry)
        """
        super().__init__(encoder)
        self.mc = client
        self.prefix = prefix
        self.default_ttl = ttl or 0

    def _fkey(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _raw_get(self, key: str) -> Optional[bytes]:
        return self.mc.get(self._fkey(key))

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        exp = ttl if ttl is not None else self.default_ttl
        self.mc.set(self._fkey(key), value, expire=exp)

    def _raw_delete(self, key: str) -> bool:
        return bool(self.mc.delete(self._fkey(key)))

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        # Memcached doesn't support key listing natively.
        # Use stats cachedump if available, or maintain a separate key set.
        raise NotImplementedError(
            "Memcached does not support key enumeration. "
            "Pass explicit key list to search() or use get_batch()."
        )

    # --- Optimized batch operations ---

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch store using set_multi."""
        exp = ttl if ttl is not None else self.default_ttl
        to_set = {}
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            to_set[self._fkey(key)] = data
            total_orig += len(vector) * 4
            total_comp += len(data)

        self.mc.set_many(to_set, expire=exp)
        self._stats["puts"] += len(items)
        self._stats["bytes_original"] += total_orig
        self._stats["bytes_compressed"] += total_comp

        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Batch retrieve using get_multi."""
        fkeys = {self._fkey(k): k for k in keys}
        results = self.mc.get_many(list(fkeys.keys()))

        output = {}
        for fk, orig_key in fkeys.items():
            self._stats["gets"] += 1
            data = results.get(fk)
            if data is not None:
                self._stats["hits"] += 1
                compressed = CompressedVector.from_bytes(data)
                output[orig_key] = self.encoder.decode(compressed)
            else:
                self._stats["misses"] += 1
                output[orig_key] = None
        return output

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Search over explicit key list (required for Memcached)."""
        if keys is None:
            raise ValueError("Memcached requires explicit key list for search()")
        return super().search(query, k=k, keys=keys)

    def cas_put(self, key: str, vector: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Atomic check-and-set: only writes if key hasn't changed since last get."""
        fk = self._fkey(key)
        result = self.mc.gets(fk)
        compressed = self.encoder.encode(np.asarray(vector, dtype=np.float32).ravel())
        data = compressed.to_bytes()
        exp = ttl if ttl is not None else self.default_ttl
        if result is None:
            self.mc.set(fk, data, expire=exp)
            return True
        _, cas_token = result
        return bool(self.mc.cas(fk, data, cas_token, expire=exp))
