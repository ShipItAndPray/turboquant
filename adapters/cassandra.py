"""
TurboQuant Cassandra Adapter
===============================
Compressed vector storage in Apache Cassandra / ScyllaDB.

Requirements: pip install cassandra-driver

Usage:
    from cassandra.cluster import Cluster
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.cassandra import CassandraTurboCache

    cluster = Cluster(["localhost"])
    encoder = TurboQuantEncoder(dim=768)
    cache = CassandraTurboCache(encoder, cluster, keyspace="myapp")

    cache.put("doc:1", vector)
    vec = cache.get("doc:1")
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class CassandraTurboCache(BaseTurboAdapter):
    """
    Cassandra/ScyllaDB adapter with TurboQuant compression.

    Features:
    - BLOB storage for compressed vectors
    - TTL per-insert (Cassandra native TTL)
    - Prepared statements for performance
    - Batch statements for bulk operations
    - Token-aware routing friendly (small payloads)
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 cluster: Any = None,
                 session: Any = None,
                 keyspace: str = "turboquant",
                 table: str = "vectors",
                 replication_factor: int = 1):
        super().__init__(encoder)
        self.keyspace = keyspace
        self.table = table

        if session:
            self.session = session
        else:
            self.session = cluster.connect()

        self._init_schema(replication_factor)
        self._prepare_statements()

    def _init_schema(self, rf):
        self.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': {rf}}}
        """)
        self.session.set_keyspace(self.keyspace)
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id text PRIMARY KEY,
                vector_data blob,
                original_dim int,
                compression_ratio float,
                metadata text,
                created_at timestamp
            )
        """)

    def _prepare_statements(self):
        self._insert_stmt = self.session.prepare(f"""
            INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, toTimestamp(now()))
        """)
        self._insert_ttl_stmt = self.session.prepare(f"""
            INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, toTimestamp(now()))
            USING TTL ?
        """)
        self._select_stmt = self.session.prepare(
            f"SELECT vector_data FROM {self.table} WHERE id = ?"
        )
        self._delete_stmt = self.session.prepare(
            f"DELETE FROM {self.table} WHERE id = ?"
        )

    def _raw_get(self, key: str) -> Optional[bytes]:
        row = self.session.execute(self._select_stmt, (key,)).one()
        return bytes(row.vector_data) if row else None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        if ttl:
            self.session.execute(self._insert_ttl_stmt,
                                 (key, value, self.encoder.dim, 0.0, None, ttl))
        else:
            self.session.execute(self._insert_stmt,
                                 (key, value, self.encoder.dim, 0.0, None))

    def _raw_delete(self, key: str) -> bool:
        self.session.execute(self._delete_stmt, (key,))
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        rows = self.session.execute(f"SELECT id FROM {self.table}")
        return [row.id for row in rows]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()
        meta = json.dumps(metadata) if metadata else None

        if ttl:
            self.session.execute(self._insert_ttl_stmt,
                                 (key, data, self.encoder.dim,
                                  compressed.compression_ratio(), meta, ttl))
        else:
            self.session.execute(self._insert_stmt,
                                 (key, data, self.encoder.dim,
                                  compressed.compression_ratio(), meta))

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

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch insert using Cassandra BATCH statement."""
        from cassandra.query import BatchStatement, BatchType

        batch = BatchStatement(batch_type=BatchType.UNLOGGED)
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            if ttl:
                batch.add(self._insert_ttl_stmt,
                          (key, data, self.encoder.dim,
                           compressed.compression_ratio(), None, ttl))
            else:
                batch.add(self._insert_stmt,
                          (key, data, self.encoder.dim,
                           compressed.compression_ratio(), None))

        self.session.execute(batch)
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
            placeholders = ", ".join(["?"] * len(keys))
            stmt = self.session.prepare(
                f"SELECT id, vector_data FROM {self.table} WHERE id IN ({placeholders})"
            )
            rows = self.session.execute(stmt, keys)
        else:
            rows = self.session.execute(f"SELECT id, vector_data FROM {self.table}")

        results = []
        for row in rows:
            candidate = CompressedVector.from_bytes(bytes(row.vector_data))
            score = self.encoder.similarity(query_c, candidate)
            results.append((row.id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
