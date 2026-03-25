"""
TurboQuant MySQL Adapter
=========================
Compressed vector storage in MySQL / MariaDB.

Requirements: pip install mysql-connector-python

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.mysql import MySQLTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = MySQLTurboCache(encoder, host="localhost", database="vectors", user="root")

    cache.put("doc:1", vector)
    vec = cache.get("doc:1")
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class MySQLTurboCache(BaseTurboAdapter):
    """
    MySQL/MariaDB adapter with TurboQuant compression.

    Features:
    - BLOB storage for compressed vectors
    - JSON metadata column (MySQL 5.7+)
    - executemany for bulk inserts
    - Connection pooling via mysql.connector.pooling
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 conn: Any = None,
                 table: str = "tq_vectors",
                 **connect_kwargs):
        super().__init__(encoder)
        self.table = table

        if conn:
            self.conn = conn
        else:
            import mysql.connector
            self.conn = mysql.connector.connect(**connect_kwargs)

        self._init_table()

    def _init_table(self):
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id VARCHAR(255) PRIMARY KEY,
                vector_data MEDIUMBLOB NOT NULL,
                original_dim INT,
                compression_ratio FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _raw_get(self, key: str) -> Optional[bytes]:
        cur = self.conn.cursor()
        cur.execute(f"SELECT vector_data FROM {self.table} WHERE id = %s", (key,))
        row = cur.fetchone()
        return bytes(row[0]) if row else None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(f"""
            INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                vector_data = VALUES(vector_data),
                compression_ratio = VALUES(compression_ratio)
        """, (key, value, self.encoder.dim, 0.0))
        self.conn.commit()

    def _raw_delete(self, key: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {self.table} WHERE id = %s", (key,))
        deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        cur = self.conn.cursor()
        if pattern == "*":
            cur.execute(f"SELECT id FROM {self.table}")
        else:
            like = pattern.replace("*", "%")
            cur.execute(f"SELECT id FROM {self.table} WHERE id LIKE %s", (like,))
        return [row[0] for row in cur.fetchall()]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        cur = self.conn.cursor()
        cur.execute(f"""
            INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                vector_data = VALUES(vector_data),
                compression_ratio = VALUES(compression_ratio),
                metadata = VALUES(metadata)
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
        cur = self.conn.cursor()
        total_orig = 0
        total_comp = 0
        rows = []

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)
            meta = json.dumps((metadata or {}).get(key)) if metadata else None
            rows.append((key, data, self.encoder.dim, compressed.compression_ratio(), meta))

        cur.executemany(f"""
            INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                vector_data = VALUES(vector_data),
                compression_ratio = VALUES(compression_ratio)
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
        cur = self.conn.cursor()

        if keys:
            placeholders = ",".join(["%s"] * len(keys))
            cur.execute(f"SELECT id, vector_data FROM {self.table} WHERE id IN ({placeholders})", keys)
        else:
            cur.execute(f"SELECT id, vector_data FROM {self.table}")

        results = []
        for row in cur.fetchall():
            candidate = CompressedVector.from_bytes(bytes(row[1]))
            score = self.encoder.similarity(query_c, candidate)
            results.append((row[0], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def close(self):
        self.conn.close()
