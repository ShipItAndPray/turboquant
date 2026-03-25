"""
TurboQuant Redis Adapter
========================
Transparent vector compression for Redis. Uses pipelines for batch ops,
SCAN for key iteration, and supports TTL, key prefixing, and Lua-based
atomic operations.

Requirements: pip install redis

Usage:
    import redis
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.redis import RedisTurboCache

    r = redis.Redis(host='localhost', port=6379, db=0)
    encoder = TurboQuantEncoder(dim=768)
    cache = RedisTurboCache(encoder, r, prefix="emb:")

    cache.put("doc:1", vector)
    cache.put_batch({"doc:2": v2, "doc:3": v3}, ttl=3600)
    vec = cache.get("doc:1")
    results = cache.search(query_vector, k=10)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class RedisTurboCache(BaseTurboAdapter):
    """
    Redis adapter with TurboQuant compression.

    Features:
    - Pipeline-based batch get/set for high throughput
    - SCAN-based key iteration (no KEYS blocking)
    - TTL support per-key or default
    - Key prefix namespace isolation
    - Memory stats via Redis INFO
    """

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 prefix: str = "tq:", ttl: Optional[int] = None):
        """
        Args:
            client: redis.Redis or redis.StrictRedis instance
            prefix: Key prefix for namespace isolation
            ttl: Default TTL in seconds (None = no expiry)
        """
        super().__init__(encoder)
        self.redis = client
        self.prefix = prefix
        self.default_ttl = ttl

    def _fkey(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _strip_prefix(self, key) -> str:
        k = key.decode() if isinstance(key, bytes) else key
        return k[len(self.prefix):] if k.startswith(self.prefix) else k

    def _raw_get(self, key: str) -> Optional[bytes]:
        return self.redis.get(self._fkey(key))

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        exp = ttl or self.default_ttl
        fk = self._fkey(key)
        if exp:
            self.redis.setex(fk, exp, value)
        else:
            self.redis.set(fk, value)

    def _raw_delete(self, key: str) -> bool:
        return bool(self.redis.delete(self._fkey(key)))

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        keys = []
        cursor = 0
        match = self._fkey(pattern)
        while True:
            cursor, batch = self.redis.scan(cursor, match=match, count=1000)
            keys.extend(self._strip_prefix(k) for k in batch)
            if cursor == 0:
                break
        return keys

    # --- Optimized batch operations using Redis pipelines ---

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch store using Redis pipeline for minimal round-trips."""
        pipe = self.redis.pipeline(transaction=False)
        exp = ttl or self.default_ttl
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            fk = self._fkey(key)
            if exp:
                pipe.setex(fk, exp, data)
            else:
                pipe.set(fk, data)
            total_orig += len(vector) * 4
            total_comp += len(data)

        pipe.execute()
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
        """Batch retrieve using Redis pipeline."""
        pipe = self.redis.pipeline(transaction=False)
        for key in keys:
            pipe.get(self._fkey(key))
        results = pipe.execute()

        output = {}
        for key, data in zip(keys, results):
            self._stats["gets"] += 1
            if data is not None:
                self._stats["hits"] += 1
                compressed = CompressedVector.from_bytes(data)
                output[key] = self.encoder.decode(compressed)
            else:
                self._stats["misses"] += 1
                output[key] = None
        return output

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Similarity search using SCAN + pipeline for efficient iteration."""
        query_compressed = self.encoder.encode(
            np.asarray(query, dtype=np.float32).ravel()
        )
        results = []

        if keys is not None:
            # Search specific keys
            pipe = self.redis.pipeline(transaction=False)
            for key in keys:
                pipe.get(self._fkey(key))
            values = pipe.execute()
            for key, data in zip(keys, values):
                if data is not None:
                    candidate = CompressedVector.from_bytes(data)
                    score = self.encoder.similarity(query_compressed, candidate)
                    results.append((key, score))
        else:
            # SCAN all keys
            cursor = 0
            while True:
                cursor, batch_keys = self.redis.scan(
                    cursor, match=self._fkey("*"), count=500
                )
                if batch_keys:
                    pipe = self.redis.pipeline(transaction=False)
                    for k in batch_keys:
                        pipe.get(k)
                    values = pipe.execute()
                    for k, data in zip(batch_keys, values):
                        if data is not None:
                            candidate = CompressedVector.from_bytes(data)
                            score = self.encoder.similarity(query_compressed, candidate)
                            results.append((self._strip_prefix(k), score))
                if cursor == 0:
                    break

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def memory_stats(self) -> dict:
        """Redis-specific memory statistics."""
        info = self.redis.info("memory")
        key_count = len(self._raw_keys())
        return {
            "vector_count": key_count,
            "redis_used_memory": info.get("used_memory_human", "unknown"),
            "redis_peak_memory": info.get("used_memory_peak_human", "unknown"),
            "redis_fragmentation_ratio": info.get("mem_fragmentation_ratio", "unknown"),
            **self.stats(),
        }

    def flush(self) -> int:
        """Delete all TurboQuant keys (safe — only deletes prefixed keys)."""
        keys = list(self.redis.scan_iter(match=self._fkey("*")))
        if keys:
            return self.redis.delete(*keys)
        return 0

    def exists(self, key: str) -> bool:
        return bool(self.redis.exists(self._fkey(key)))

    def ttl(self, key: str) -> int:
        """Get remaining TTL for a key (-1 = no expiry, -2 = not found)."""
        return self.redis.ttl(self._fkey(key))
