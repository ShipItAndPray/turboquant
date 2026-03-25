"""
TurboQuant RocksDB Adapter
============================
Embedded LSM-tree store with TurboQuant compression.

Requirements: pip install python-rocksdb

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.rocksdb import RocksDBTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = RocksDBTurboCache(encoder, path="./vectors.rocksdb")

    cache.put("doc:1", vector)
    vec = cache.get("doc:1")
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class RocksDBTurboCache(BaseTurboAdapter):
    """
    RocksDB adapter with TurboQuant compression.

    RocksDB excels at write-heavy workloads. Combined with TurboQuant,
    it reduces write amplification (smaller values = fewer compactions).
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 path: str = "./turboquant.rocksdb"):
        super().__init__(encoder)
        import rocksdb
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 300000
        opts.write_buffer_size = 64 * 1024 * 1024
        opts.max_write_buffer_number = 3
        opts.target_file_size_base = 64 * 1024 * 1024
        self.db = rocksdb.DB(path, opts)

    def _raw_get(self, key: str) -> Optional[bytes]:
        data = self.db.get(key.encode())
        return bytes(data) if data else None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        self.db.put(key.encode(), value)

    def _raw_delete(self, key: str) -> bool:
        self.db.delete(key.encode())
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        keys = []
        it = self.db.iterkeys()
        it.seek_to_first()
        for key in it:
            keys.append(key.decode())
        return keys

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch write using RocksDB WriteBatch."""
        import rocksdb
        batch = rocksdb.WriteBatch()
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            batch.put(key.encode(), data)
            total_orig += len(vector) * 4
            total_comp += len(data)

        self.db.write(batch)
        self._stats["puts"] += len(items)
        self._stats["bytes_original"] += total_orig
        self._stats["bytes_compressed"] += total_comp

        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def close(self):
        del self.db
