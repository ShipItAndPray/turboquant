"""
Base adapter class — all TurboQuant adapters inherit from this.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core import TurboQuantEncoder, CompressedVector


class BaseTurboAdapter(ABC):
    """
    Base class for all TurboQuant cache/database/storage adapters.

    Subclasses implement _raw_get, _raw_set, _raw_delete, _raw_keys.
    This base class handles encode/decode, batching, search, and stats.
    """

    def __init__(self, encoder: TurboQuantEncoder, **kwargs):
        self.encoder = encoder
        self._stats = {
            "puts": 0, "gets": 0, "hits": 0, "misses": 0,
            "bytes_original": 0, "bytes_compressed": 0,
            "encode_time_ms": 0.0, "decode_time_ms": 0.0,
        }

    # --- Subclasses implement these ---

    @abstractmethod
    def _raw_get(self, key: str) -> Optional[bytes]:
        """Get raw bytes from backend."""

    @abstractmethod
    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set raw bytes in backend."""

    @abstractmethod
    def _raw_delete(self, key: str) -> bool:
        """Delete key from backend."""

    @abstractmethod
    def _raw_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""

    # --- Public API (same for all adapters) ---

    def put(self, key: str, vector: np.ndarray, ttl: Optional[int] = None) -> dict:
        """Store a compressed vector."""
        vector = np.asarray(vector, dtype=np.float32).ravel()

        t0 = time.time()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()
        self._stats["encode_time_ms"] += (time.time() - t0) * 1000

        self._raw_set(key, data, ttl=ttl)

        original_bytes = len(vector) * 4
        self._stats["puts"] += 1
        self._stats["bytes_original"] += original_bytes
        self._stats["bytes_compressed"] += len(data)

        return {
            "key": key,
            "original_bytes": original_bytes,
            "compressed_bytes": len(data),
            "ratio": f"{original_bytes / len(data):.1f}x",
        }

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve and decompress a vector."""
        self._stats["gets"] += 1
        data = self._raw_get(key)
        if data is None:
            self._stats["misses"] += 1
            return None
        self._stats["hits"] += 1

        t0 = time.time()
        compressed = CompressedVector.from_bytes(data)
        result = self.encoder.decode(compressed)
        self._stats["decode_time_ms"] += (time.time() - t0) * 1000
        return result

    def get_compressed(self, key: str) -> Optional[CompressedVector]:
        """Get compressed vector without decoding (for similarity search)."""
        data = self._raw_get(key)
        if data is None:
            return None
        return CompressedVector.from_bytes(data)

    def delete(self, key: str) -> bool:
        return self._raw_delete(key)

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch store vectors. Subclasses can override for pipeline/bulk optimizations."""
        total_orig = 0
        total_comp = 0
        for key, vec in items.items():
            info = self.put(key, vec, ttl=ttl)
            total_orig += info["original_bytes"]
            total_comp += info["compressed_bytes"]
        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Batch retrieve vectors. Subclasses can override for pipeline optimizations."""
        return {key: self.get(key) for key in keys}

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Brute-force similarity search over stored compressed vectors.

        Args:
            query: Query vector
            k: Number of results
            keys: Specific keys to search (default: all keys)
        """
        if keys is None:
            keys = self._raw_keys()

        query_compressed = self.encoder.encode(np.asarray(query, dtype=np.float32).ravel())
        results = []

        for key in keys:
            data = self._raw_get(key)
            if data is not None:
                candidate = CompressedVector.from_bytes(data)
                score = self.encoder.similarity(query_compressed, candidate)
                results.append((key, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def stats(self) -> dict:
        s = dict(self._stats)
        s["hit_rate"] = f"{s['hits'] / max(s['gets'], 1):.1%}"
        s["avg_encode_ms"] = f"{s['encode_time_ms'] / max(s['puts'], 1):.2f}"
        s["avg_decode_ms"] = f"{s['decode_time_ms'] / max(s['hits'], 1):.2f}"
        if s["bytes_original"] > 0:
            s["overall_ratio"] = f"{s['bytes_original'] / max(s['bytes_compressed'], 1):.1f}x"
        return s
