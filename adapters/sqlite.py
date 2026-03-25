"""
TurboQuant SQLite Adapter
==========================
Zero-dependency compressed vector storage using Python's built-in sqlite3.

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.sqlite import SQLiteTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = SQLiteTurboCache(encoder, db_path="vectors.db")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    vec = cache.get("doc:1")
    results = cache.search(query_vector, k=10)
"""

import json
import sqlite3
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class SQLiteTurboCache(BaseTurboAdapter):
    """
    SQLite adapter with TurboQuant compression.

    Zero external dependencies — uses Python's built-in sqlite3 module.
    Great for local development, testing, and embedded applications.

    Features:
    - BLOB storage for compressed vectors
    - WAL mode for concurrent reads
    - JSON1 extension for metadata queries
    - Batch insert via executemany
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 db_path: str = "turboquant_vectors.db",
                 table: str = "tq_vectors"):
        super().__init__(encoder)
        self.table = table
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_table()

    def _init_table(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id TEXT PRIMARY KEY,
                vector_data BLOB NOT NULL,
                original_dim INTEGER,
                compression_ratio REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _raw_get(self, key: str) -> Optional[bytes]:
        cur = self.conn.execute(
            f"SELECT vector_data FROM {self.table} WHERE id = ?", (key,)
        )
        row = cur.fetchone()
        return bytes(row[0]) if row else None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self.table} (id, vector_data, original_dim) VALUES (?, ?, ?)",
            (key, value, self.encoder.dim)
        )
        self.conn.commit()

    def _raw_delete(self, key: str) -> bool:
        cur = self.conn.execute(f"DELETE FROM {self.table} WHERE id = ?", (key,))
        self.conn.commit()
        return cur.rowcount > 0

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        if pattern == "*":
            cur = self.conn.execute(f"SELECT id FROM {self.table}")
        else:
            like = pattern.replace("*", "%")
            cur = self.conn.execute(f"SELECT id FROM {self.table} WHERE id LIKE ?", (like,))
        return [row[0] for row in cur.fetchall()]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        self.conn.execute(f"""
            INSERT OR REPLACE INTO {self.table}
            (id, vector_data, original_dim, compression_ratio, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (key, data, self.encoder.dim, compressed.compression_ratio(),
              json.dumps(metadata) if metadata else None))
        self.conn.commit()

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

    def bulk_put(self, items: Dict[str, np.ndarray],
                 metadata: Optional[Dict[str, dict]] = None) -> dict:
        rows = []
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)
            meta = json.dumps((metadata or {}).get(key)) if metadata else None
            rows.append((key, data, self.encoder.dim, compressed.compression_ratio(), meta))

        self.conn.executemany(f"""
            INSERT OR REPLACE INTO {self.table}
            (id, vector_data, original_dim, compression_ratio, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()

        self._stats["puts"] += len(items)
        self._stats["bytes_original"] += total_orig
        self._stats["bytes_compressed"] += total_comp

        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32).ravel()
        query_c = self.encoder.encode(query)

        if keys:
            placeholders = ",".join(["?"] * len(keys))
            cur = self.conn.execute(
                f"SELECT id, vector_data FROM {self.table} WHERE id IN ({placeholders})", keys
            )
        else:
            cur = self.conn.execute(f"SELECT id, vector_data FROM {self.table}")

        results = []
        for row in cur.fetchall():
            candidate = CompressedVector.from_bytes(bytes(row[1]))
            score = self.encoder.similarity(query_c, candidate)
            results.append((row[0], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def table_stats(self) -> dict:
        cur = self.conn.execute(
            f"SELECT COUNT(*), AVG(compression_ratio), SUM(LENGTH(vector_data)) FROM {self.table}"
        )
        count, avg_ratio, total_bytes = cur.fetchone()
        return {
            "vector_count": count,
            "avg_compression_ratio": f"{avg_ratio:.1f}x" if avg_ratio else "N/A",
            "total_compressed_bytes": total_bytes or 0,
        }

    def close(self):
        self.conn.close()
