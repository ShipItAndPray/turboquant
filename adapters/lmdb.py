"""
TurboQuant LMDB Adapter
=========================
Ultra-fast embedded key-value store with TurboQuant compression.

Requirements: pip install lmdb

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.lmdb import LMDBTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = LMDBTurboCache(encoder, path="./vectors.lmdb")

    cache.put("doc:1", vector)
    vec = cache.get("doc:1")
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class LMDBTurboCache(BaseTurboAdapter):
    """
    LMDB adapter with TurboQuant compression.

    LMDB is a memory-mapped B+ tree — zero-copy reads, ACID transactions,
    and the fastest embedded key-value store. Combined with TurboQuant,
    it's ideal for local vector caches.
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 path: str = "./turboquant.lmdb",
                 map_size: int = 10 * 1024 * 1024 * 1024):  # 10GB default
        super().__init__(encoder)
        import lmdb
        self.env = lmdb.open(path, map_size=map_size)

    def _raw_get(self, key: str) -> Optional[bytes]:
        with self.env.begin() as txn:
            data = txn.get(key.encode())
            return bytes(data) if data else None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), value)

    def _raw_delete(self, key: str) -> bool:
        with self.env.begin(write=True) as txn:
            return txn.delete(key.encode())

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                keys.append(key.decode())
        return keys

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch write in single LMDB transaction."""
        total_orig = 0
        total_comp = 0

        with self.env.begin(write=True) as txn:
            for key, vector in items.items():
                vector = np.asarray(vector, dtype=np.float32).ravel()
                compressed = self.encoder.encode(vector)
                data = compressed.to_bytes()
                txn.put(key.encode(), data)
                total_orig += len(vector) * 4
                total_comp += len(data)

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
        """Batch read in single LMDB transaction."""
        output = {}
        with self.env.begin() as txn:
            for key in keys:
                self._stats["gets"] += 1
                data = txn.get(key.encode())
                if data:
                    self._stats["hits"] += 1
                    compressed = CompressedVector.from_bytes(bytes(data))
                    output[key] = self.encoder.decode(compressed)
                else:
                    self._stats["misses"] += 1
                    output[key] = None
        return output

    def env_stats(self) -> dict:
        stat = self.env.stat()
        info = self.env.info()
        return {
            "entries": stat["entries"],
            "page_size": stat["psize"],
            "tree_depth": stat["depth"],
            "map_size": info["map_size"],
            "last_page": info["last_pgno"],
        }

    def close(self):
        self.env.close()
