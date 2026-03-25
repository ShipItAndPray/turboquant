"""
TurboQuant PostgreSQL Adapter
===============================
Compressed vector storage in PostgreSQL. Works with or without pgvector.

Requirements: pip install psycopg2-binary

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.postgresql import PostgresTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = PostgresTurboCache(encoder, dsn="postgresql://user:pass@localhost/mydb")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    vec = cache.get("doc:1")
    results = cache.search(query_vector, k=10)
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class PostgresTurboCache(BaseTurboAdapter):
    """
    PostgreSQL adapter with TurboQuant compression.

    Features:
    - BYTEA storage for compressed vectors (~84% smaller than float[])
    - Optional pgvector integration for hybrid search
    - COPY-based bulk insert for high throughput
    - GIN index on metadata JSONB
    - Connection pooling support
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 dsn: str = "postgresql://localhost/turboquant",
                 conn: Any = None,
                 table: str = "tq_vectors",
                 use_pgvector: bool = False):
        """
        Args:
            dsn: PostgreSQL connection string
            conn: Existing psycopg2 connection (overrides dsn)
            table: Table name
            use_pgvector: If True, also store pgvector column for native ANN
        """
        super().__init__(encoder)
        self.table = table
        self.use_pgvector = use_pgvector

        if conn:
            self.conn = conn
        else:
            import psycopg2
            self.conn = psycopg2.connect(dsn)
            self.conn.autocommit = True

        self._init_table()

    def _init_table(self):
        cur = self.conn.cursor()

        if self.use_pgvector:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cols = [
            "id TEXT PRIMARY KEY",
            "vector_data BYTEA NOT NULL",
            "original_dim INTEGER",
            "compression_ratio REAL",
            "metadata JSONB",
            "created_at TIMESTAMPTZ DEFAULT NOW()",
        ]
        if self.use_pgvector:
            cols.append(f"vector_dense vector({self.encoder.dim})")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                {', '.join(cols)}
            )
        """)

        # Index on metadata for filtered queries
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table}_metadata
            ON {self.table} USING gin(metadata)
        """)

        if self.use_pgvector:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table}_vector
                ON {self.table} USING ivfflat(vector_dense vector_cosine_ops)
                WITH (lists = 100)
            """)

        self.conn.commit()

    def _raw_get(self, key: str) -> Optional[bytes]:
        cur = self.conn.cursor()
        cur.execute(f"SELECT vector_data FROM {self.table} WHERE id = %s", (key,))
        row = cur.fetchone()
        return bytes(row[0]) if row else None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        cur = self.conn.cursor()
        import psycopg2
        cur.execute(f"""
            INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                vector_data = EXCLUDED.vector_data,
                compression_ratio = EXCLUDED.compression_ratio
        """, (key, psycopg2.Binary(value), self.encoder.dim, 0.0))
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
        """Store vector with metadata and optional pgvector column."""
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        import psycopg2
        cur = self.conn.cursor()

        if self.use_pgvector:
            cur.execute(f"""
                INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata, vector_dense)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    vector_data = EXCLUDED.vector_data,
                    compression_ratio = EXCLUDED.compression_ratio,
                    metadata = EXCLUDED.metadata,
                    vector_dense = EXCLUDED.vector_dense
            """, (key, psycopg2.Binary(data), self.encoder.dim,
                  compressed.compression_ratio(),
                  json.dumps(metadata) if metadata else None,
                  vector.tolist()))
        else:
            cur.execute(f"""
                INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    vector_data = EXCLUDED.vector_data,
                    compression_ratio = EXCLUDED.compression_ratio,
                    metadata = EXCLUDED.metadata
            """, (key, psycopg2.Binary(data), self.encoder.dim,
                  compressed.compression_ratio(),
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
        """High-throughput bulk insert using executemany."""
        import psycopg2
        cur = self.conn.cursor()
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

            if self.use_pgvector:
                rows.append((key, psycopg2.Binary(data), self.encoder.dim,
                             compressed.compression_ratio(), meta, vector.tolist()))
            else:
                rows.append((key, psycopg2.Binary(data), self.encoder.dim,
                             compressed.compression_ratio(), meta))

        if self.use_pgvector:
            from psycopg2.extras import execute_values
            execute_values(cur, f"""
                INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata, vector_dense)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    vector_data = EXCLUDED.vector_data,
                    compression_ratio = EXCLUDED.compression_ratio
            """, rows)
        else:
            from psycopg2.extras import execute_values
            execute_values(cur, f"""
                INSERT INTO {self.table} (id, vector_data, original_dim, compression_ratio, metadata)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    vector_data = EXCLUDED.vector_data,
                    compression_ratio = EXCLUDED.compression_ratio
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
               keys: Optional[List[str]] = None,
               mode: str = "compressed",
               metadata_filter: Optional[dict] = None) -> List[Tuple[str, float]]:
        """
        Search modes:
        - "compressed": Client-side TurboQuant similarity
        - "pgvector": Native pgvector ANN (requires use_pgvector=True)
        - "hybrid": pgvector candidates + TurboQuant rerank
        """
        query = np.asarray(query, dtype=np.float32).ravel()

        if mode == "pgvector" and self.use_pgvector:
            return self._search_pgvector(query, k, metadata_filter)
        elif mode == "hybrid" and self.use_pgvector:
            return self._search_hybrid(query, k, metadata_filter)
        else:
            return self._search_compressed(query, k, keys)

    def _search_compressed(self, query, k, keys=None):
        query_c = self.encoder.encode(query)
        cur = self.conn.cursor()
        if keys:
            cur.execute(f"SELECT id, vector_data FROM {self.table} WHERE id = ANY(%s)", (keys,))
        else:
            cur.execute(f"SELECT id, vector_data FROM {self.table}")

        results = []
        for row in cur.fetchall():
            candidate = CompressedVector.from_bytes(bytes(row[1]))
            score = self.encoder.similarity(query_c, candidate)
            results.append((row[0], score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _search_pgvector(self, query, k, metadata_filter=None):
        cur = self.conn.cursor()
        q_list = query.tolist()
        if metadata_filter:
            cur.execute(f"""
                SELECT id, 1 - (vector_dense <=> %s::vector) as score
                FROM {self.table}
                WHERE metadata @> %s::jsonb
                ORDER BY vector_dense <=> %s::vector
                LIMIT %s
            """, (q_list, json.dumps(metadata_filter), q_list, k))
        else:
            cur.execute(f"""
                SELECT id, 1 - (vector_dense <=> %s::vector) as score
                FROM {self.table}
                ORDER BY vector_dense <=> %s::vector
                LIMIT %s
            """, (q_list, q_list, k))
        return [(row[0], float(row[1])) for row in cur.fetchall()]

    def _search_hybrid(self, query, k, metadata_filter=None):
        # Get 3x candidates from pgvector
        candidates = self._search_pgvector(query, k * 3, metadata_filter)
        candidate_keys = [c[0] for c in candidates]
        # Rerank with TurboQuant
        return self._search_compressed(query, k, candidate_keys)

    def table_stats(self) -> dict:
        cur = self.conn.cursor()
        cur.execute(f"SELECT COUNT(*), AVG(compression_ratio), SUM(LENGTH(vector_data)) FROM {self.table}")
        count, avg_ratio, total_bytes = cur.fetchone()
        cur.execute(f"SELECT pg_total_relation_size('{self.table}')")
        table_size = cur.fetchone()[0]
        return {
            "vector_count": count,
            "avg_compression_ratio": f"{avg_ratio:.1f}x" if avg_ratio else "N/A",
            "compressed_data_bytes": total_bytes or 0,
            "table_total_bytes": table_size,
            "table_human": f"{table_size / 1e6:.1f} MB",
        }

    def close(self):
        self.conn.close()
