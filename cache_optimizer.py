"""
TurboQuant Cache Optimizer — Compress Redis, Elasticsearch, Ehcache, and Any K-V Cache
========================================================================================
Drop-in adapters that transparently compress/decompress vectors using TurboQuant.

Supported backends:
  - Redis: Store compressed vectors as binary strings
  - Elasticsearch: Compressed vector search with score correction
  - Generic: Wraps any dict-like cache (Ehcache-style via Py4J, memcached, etc.)
  - Database: SQLite/Postgres/MySQL compressed BLOB storage

Usage:
    from turboquant.core import TurboQuantEncoder, TurboQuantConfig
    from turboquant.cache_optimizer import RedisTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = RedisTurboCache(encoder, redis_client)

    cache.put("doc:1", embedding_vector)
    vec = cache.get("doc:1")
    results = cache.search(query_vector, k=10)
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Protocol
from dataclasses import dataclass
import numpy as np

from core import TurboQuantEncoder, TurboQuantConfig, CompressedVector


# ============================================================================
# Base Cache Protocol
# ============================================================================

class CacheBackend(Protocol):
    """Protocol for any cache backend."""
    def get(self, key: str) -> Optional[bytes]: ...
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def keys(self, pattern: str = "*") -> List[str]: ...


# ============================================================================
# Redis Adapter
# ============================================================================

class RedisTurboCache:
    """
    Redis adapter with TurboQuant compression.

    Stores vectors as compressed binary strings. Achieves ~4-6x memory reduction
    on vector data while maintaining near-perfect recall for similarity search.

    Features:
    - Transparent encode/decode on get/put
    - Batch operations for pipeline efficiency
    - Built-in similarity search (brute-force over compressed vectors)
    - TTL support
    - Key prefixing for namespace isolation
    """

    def __init__(self, encoder: TurboQuantEncoder, redis_client: Any,
                 prefix: str = "tq:", ttl: Optional[int] = None):
        self.encoder = encoder
        self.redis = redis_client
        self.prefix = prefix
        self.default_ttl = ttl

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def put(self, key: str, vector: np.ndarray, ttl: Optional[int] = None) -> dict:
        """Store a compressed vector in Redis."""
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        rkey = self._key(key)
        exp = ttl or self.default_ttl
        if exp:
            self.redis.setex(rkey, exp, data)
        else:
            self.redis.set(rkey, data)

        return {
            "key": key,
            "original_bytes": len(vector) * 4,
            "compressed_bytes": len(data),
            "ratio": f"{(len(vector) * 4) / len(data):.1f}x",
        }

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve and decompress a vector from Redis."""
        data = self.redis.get(self._key(key))
        if data is None:
            return None
        compressed = CompressedVector.from_bytes(data)
        return self.encoder.decode(compressed)

    def get_compressed(self, key: str) -> Optional[CompressedVector]:
        """Get the compressed vector (for similarity without decompression)."""
        data = self.redis.get(self._key(key))
        if data is None:
            return None
        return CompressedVector.from_bytes(data)

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch store compressed vectors using Redis pipeline."""
        pipe = self.redis.pipeline()
        exp = ttl or self.default_ttl
        total_original = 0
        total_compressed = 0

        for key, vector in items.items():
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            rkey = self._key(key)
            if exp:
                pipe.setex(rkey, exp, data)
            else:
                pipe.set(rkey, data)
            total_original += len(vector) * 4
            total_compressed += len(data)

        pipe.execute()
        return {
            "count": len(items),
            "original_bytes": total_original,
            "compressed_bytes": total_compressed,
            "ratio": f"{total_original / max(total_compressed, 1):.1f}x",
        }

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Batch retrieve and decompress vectors."""
        pipe = self.redis.pipeline()
        for key in keys:
            pipe.get(self._key(key))
        results = pipe.execute()

        output = {}
        for key, data in zip(keys, results):
            if data is not None:
                compressed = CompressedVector.from_bytes(data)
                output[key] = self.encoder.decode(compressed)
            else:
                output[key] = None
        return output

    def search(self, query: np.ndarray, k: int = 10,
               key_pattern: str = "*") -> List[Tuple[str, float]]:
        """
        Brute-force similarity search over compressed vectors.

        For large-scale search, use Elasticsearch adapter instead.
        This is suitable for < 100K vectors.
        """
        query_compressed = self.encoder.encode(query)
        results = []

        # Scan all keys matching pattern
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=self._key(key_pattern))
            pipe = self.redis.pipeline()
            for key in keys:
                pipe.get(key)
            values = pipe.execute()

            for key, data in zip(keys, values):
                if data is not None:
                    candidate = CompressedVector.from_bytes(data)
                    score = self.encoder.similarity(query_compressed, candidate)
                    clean_key = key.decode() if isinstance(key, bytes) else key
                    clean_key = clean_key[len(self.prefix):]
                    results.append((clean_key, score))

            if cursor == 0:
                break

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def memory_stats(self) -> dict:
        """Get memory usage statistics for compressed vectors."""
        info = self.redis.info("memory")
        keys = list(self.redis.scan_iter(match=self._key("*")))
        total_compressed = 0
        for key in keys:
            data = self.redis.get(key)
            if data:
                total_compressed += len(data)

        return {
            "vector_count": len(keys),
            "total_compressed_bytes": total_compressed,
            "avg_bytes_per_vector": total_compressed // max(len(keys), 1),
            "redis_used_memory": info.get("used_memory_human", "unknown"),
        }

    def delete(self, key: str) -> bool:
        return bool(self.redis.delete(self._key(key)))

    def flush(self) -> int:
        """Delete all TurboQuant keys."""
        keys = list(self.redis.scan_iter(match=self._key("*")))
        if keys:
            return self.redis.delete(*keys)
        return 0


# ============================================================================
# Elasticsearch Adapter
# ============================================================================

class ElasticsearchTurboCache:
    """
    Elasticsearch adapter with TurboQuant compression.

    Stores compressed vectors alongside dense_vector fields for hybrid search.
    Uses compressed similarity for re-ranking and memory-efficient storage.

    Strategies:
    1. Store compressed vectors as binary fields + approximate kNN
    2. Use TurboQuant similarity as a script_score for re-ranking
    3. Bulk index with compression for memory reduction
    """

    def __init__(self, encoder: TurboQuantEncoder, es_client: Any,
                 index_name: str = "turboquant_vectors"):
        self.encoder = encoder
        self.es = es_client
        self.index_name = index_name

    def create_index(self, dims: Optional[int] = None) -> dict:
        """Create an optimized index for compressed vector storage."""
        dims = dims or self.encoder.dim
        mapping = {
            "mappings": {
                "properties": {
                    "vector_compressed": {
                        "type": "binary",  # Stores TurboQuant compressed bytes
                    },
                    "vector_dense": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True,
                    },
                    "compression_ratio": {
                        "type": "float",
                    },
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "30s",
            }
        }

        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        return self.es.indices.create(index=self.index_name, body=mapping)

    def index_vector(self, doc_id: str, vector: np.ndarray,
                     metadata: Optional[dict] = None,
                     store_dense: bool = False) -> dict:
        """
        Index a vector with TurboQuant compression.

        Args:
            store_dense: If True, also stores the full dense_vector for ES kNN.
                        Set False for maximum compression (use compressed search).
        """
        import base64

        compressed = self.encoder.encode(vector)
        compressed_b64 = base64.b64encode(compressed.to_bytes()).decode()

        doc = {
            "vector_compressed": compressed_b64,
            "compression_ratio": compressed.compression_ratio(),
            "metadata": metadata or {},
        }

        if store_dense:
            doc["vector_dense"] = vector.tolist()

        return self.es.index(index=self.index_name, id=doc_id, body=doc)

    def bulk_index(self, vectors: Dict[str, np.ndarray],
                   metadata: Optional[Dict[str, dict]] = None,
                   store_dense: bool = False,
                   chunk_size: int = 500) -> dict:
        """Bulk index vectors with compression."""
        import base64

        actions = []
        total_original = 0
        total_compressed = 0

        for doc_id, vector in vectors.items():
            compressed = self.encoder.encode(vector)
            compressed_bytes = compressed.to_bytes()
            compressed_b64 = base64.b64encode(compressed_bytes).decode()

            total_original += len(vector) * 4
            total_compressed += len(compressed_bytes)

            doc = {
                "vector_compressed": compressed_b64,
                "compression_ratio": compressed.compression_ratio(),
                "metadata": (metadata or {}).get(doc_id, {}),
            }
            if store_dense:
                doc["vector_dense"] = vector.tolist()

            actions.append({"index": {"_index": self.index_name, "_id": doc_id}})
            actions.append(doc)

            if len(actions) >= chunk_size * 2:
                self.es.bulk(body=actions, refresh=False)
                actions = []

        if actions:
            self.es.bulk(body=actions, refresh=False)

        self.es.indices.refresh(index=self.index_name)

        return {
            "indexed": len(vectors),
            "original_bytes": total_original,
            "compressed_bytes": total_compressed,
            "ratio": f"{total_original / max(total_compressed, 1):.1f}x",
        }

    def search(self, query: np.ndarray, k: int = 10,
               use_dense_knn: bool = False, rerank: bool = True) -> List[dict]:
        """
        Search for similar vectors.

        Modes:
        1. use_dense_knn=True: Use ES native kNN, optionally rerank with TurboQuant
        2. use_dense_knn=False: Fetch all compressed vectors, compute similarity client-side
        """
        import base64

        if use_dense_knn:
            # Use ES native approximate kNN
            body = {
                "knn": {
                    "field": "vector_dense",
                    "query_vector": query.tolist(),
                    "k": k * 3 if rerank else k,
                    "num_candidates": max(k * 10, 100),
                },
                "_source": ["vector_compressed", "metadata", "compression_ratio"],
            }
            response = self.es.search(index=self.index_name, body=body)
            hits = response["hits"]["hits"]

            if rerank and hits:
                query_compressed = self.encoder.encode(query)
                reranked = []
                for hit in hits:
                    compressed_b64 = hit["_source"]["vector_compressed"]
                    compressed = CompressedVector.from_bytes(
                        base64.b64decode(compressed_b64)
                    )
                    score = self.encoder.similarity(query_compressed, compressed)
                    reranked.append({
                        "id": hit["_id"],
                        "score": score,
                        "es_score": hit["_score"],
                        "metadata": hit["_source"].get("metadata", {}),
                    })
                reranked.sort(key=lambda x: x["score"], reverse=True)
                return reranked[:k]
            else:
                return [{
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {}),
                } for hit in hits[:k]]

        else:
            # Client-side compressed search (no dense_vector needed)
            query_compressed = self.encoder.encode(query)
            body = {
                "query": {"match_all": {}},
                "_source": ["vector_compressed", "metadata"],
                "size": 10000,  # Fetch all (for small indices)
            }
            response = self.es.search(index=self.index_name, body=body)

            results = []
            for hit in response["hits"]["hits"]:
                compressed_b64 = hit["_source"]["vector_compressed"]
                compressed = CompressedVector.from_bytes(
                    base64.b64decode(compressed_b64)
                )
                score = self.encoder.similarity(query_compressed, compressed)
                results.append({
                    "id": hit["_id"],
                    "score": score,
                    "metadata": hit["_source"].get("metadata", {}),
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]

    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """Retrieve and decompress a vector by document ID."""
        import base64
        try:
            doc = self.es.get(index=self.index_name, id=doc_id)
            compressed_b64 = doc["_source"]["vector_compressed"]
            compressed = CompressedVector.from_bytes(base64.b64decode(compressed_b64))
            return self.encoder.decode(compressed)
        except Exception:
            return None

    def stats(self) -> dict:
        """Get index statistics."""
        stats = self.es.indices.stats(index=self.index_name)
        idx_stats = stats["indices"][self.index_name]["total"]
        return {
            "doc_count": idx_stats["docs"]["count"],
            "store_size": idx_stats["store"]["size_in_bytes"],
            "store_size_human": f"{idx_stats['store']['size_in_bytes'] / 1e6:.1f}MB",
        }


# ============================================================================
# Generic Cache Adapter (Ehcache-style, memcached, dict, etc.)
# ============================================================================

class GenericTurboCache:
    """
    Generic cache adapter with TurboQuant compression.

    Works with any backend that supports get/set/delete with bytes values.
    Compatible with: Python dict, memcached, Ehcache (via Py4J), shelve, etc.

    For Ehcache (Java) integration, use Py4J gateway or subprocess bridge.
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 backend: Optional[Any] = None):
        """
        Args:
            backend: Any object with get(key)->bytes, set(key, bytes), delete(key).
                    If None, uses an in-memory dict.
        """
        self.encoder = encoder
        self.backend = backend or InMemoryBackend()
        self._stats = {"puts": 0, "gets": 0, "hits": 0, "bytes_saved": 0}

    def put(self, key: str, vector: np.ndarray) -> dict:
        """Store a compressed vector."""
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()
        self.backend.set(key, data)

        original_bytes = len(vector) * 4
        self._stats["puts"] += 1
        self._stats["bytes_saved"] += original_bytes - len(data)

        return {
            "key": key,
            "original_bytes": original_bytes,
            "compressed_bytes": len(data),
            "ratio": f"{original_bytes / len(data):.1f}x",
        }

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve and decompress a vector."""
        self._stats["gets"] += 1
        data = self.backend.get(key)
        if data is None:
            return None
        self._stats["hits"] += 1
        compressed = CompressedVector.from_bytes(data)
        return self.encoder.decode(compressed)

    def delete(self, key: str) -> None:
        self.backend.delete(key)

    def put_batch(self, items: Dict[str, np.ndarray]) -> dict:
        """Batch store vectors."""
        total_original = 0
        total_compressed = 0
        for key, vec in items.items():
            info = self.put(key, vec)
            total_original += info["original_bytes"]
            total_compressed += info["compressed_bytes"]
        return {
            "count": len(items),
            "original_bytes": total_original,
            "compressed_bytes": total_compressed,
            "ratio": f"{total_original / max(total_compressed, 1):.1f}x",
        }

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Batch retrieve vectors."""
        return {key: self.get(key) for key in keys}

    def search(self, query: np.ndarray, keys: List[str], k: int = 10) -> List[Tuple[str, float]]:
        """Search over specified keys for similar vectors."""
        query_compressed = self.encoder.encode(query)
        results = []

        for key in keys:
            data = self.backend.get(key)
            if data is not None:
                candidate = CompressedVector.from_bytes(data)
                score = self.encoder.similarity(query_compressed, candidate)
                results.append((key, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def stats(self) -> dict:
        return {
            **self._stats,
            "hit_rate": f"{self._stats['hits'] / max(self._stats['gets'], 1):.1%}",
            "total_bytes_saved": self._stats["bytes_saved"],
        }


class InMemoryBackend:
    """Simple in-memory dict backend for GenericTurboCache."""

    def __init__(self):
        self._store: Dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self._store.get(key)

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def keys(self, pattern: str = "*") -> List[str]:
        return list(self._store.keys())


# ============================================================================
# Database Adapter (SQLite / PostgreSQL / MySQL)
# ============================================================================

class DatabaseTurboCache:
    """
    Database adapter storing TurboQuant-compressed vectors as BLOBs.

    Supports SQLite (built-in), PostgreSQL (psycopg2), MySQL (mysql-connector).
    Creates a table with: id TEXT PK, vector_data BLOB, metadata JSON, created_at.
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 db_url: str = "sqlite:///turboquant_vectors.db",
                 table_name: str = "vectors"):
        self.encoder = encoder
        self.table_name = table_name
        self.db_url = db_url
        self._conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database connection and create table if needed."""
        if self.db_url.startswith("sqlite"):
            import sqlite3
            db_path = self.db_url.replace("sqlite:///", "")
            self._conn = sqlite3.connect(db_path)
            self._db_type = "sqlite"
        elif self.db_url.startswith("postgresql"):
            import psycopg2
            self._conn = psycopg2.connect(self.db_url)
            self._db_type = "postgresql"
        elif self.db_url.startswith("mysql"):
            import mysql.connector
            # Parse mysql://user:pass@host/db
            self._conn = mysql.connector.connect(
                host=self.db_url.split("@")[1].split("/")[0],
                database=self.db_url.split("/")[-1],
            )
            self._db_type = "mysql"
        else:
            raise ValueError(f"Unsupported database URL: {self.db_url}")

        cursor = self._conn.cursor()
        blob_type = "BLOB" if self._db_type in ("sqlite", "mysql") else "BYTEA"
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                vector_data {blob_type} NOT NULL,
                original_dim INTEGER,
                compression_ratio REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def put(self, key: str, vector: np.ndarray, metadata: Optional[dict] = None) -> dict:
        """Store a compressed vector."""
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()
        meta_json = json.dumps(metadata) if metadata else None

        cursor = self._conn.cursor()
        if self._db_type == "sqlite":
            cursor.execute(
                f"INSERT OR REPLACE INTO {self.table_name} "
                f"(id, vector_data, original_dim, compression_ratio, metadata) "
                f"VALUES (?, ?, ?, ?, ?)",
                (key, data, self.encoder.dim, compressed.compression_ratio(), meta_json)
            )
        else:
            cursor.execute(
                f"INSERT INTO {self.table_name} "
                f"(id, vector_data, original_dim, compression_ratio, metadata) "
                f"VALUES (%s, %s, %s, %s, %s) "
                f"ON CONFLICT (id) DO UPDATE SET vector_data = EXCLUDED.vector_data",
                (key, data, self.encoder.dim, compressed.compression_ratio(), meta_json)
            )
        self._conn.commit()

        return {
            "key": key,
            "original_bytes": len(vector) * 4,
            "compressed_bytes": len(data),
            "ratio": f"{(len(vector) * 4) / len(data):.1f}x",
        }

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve and decompress a vector."""
        cursor = self._conn.cursor()
        placeholder = "?" if self._db_type == "sqlite" else "%s"
        cursor.execute(
            f"SELECT vector_data FROM {self.table_name} WHERE id = {placeholder}",
            (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        compressed = CompressedVector.from_bytes(bytes(row[0]))
        return self.encoder.decode(compressed)

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search all stored vectors for similarity."""
        query_compressed = self.encoder.encode(query)
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT id, vector_data FROM {self.table_name}")

        results = []
        for row in cursor.fetchall():
            doc_id = row[0]
            compressed = CompressedVector.from_bytes(bytes(row[1]))
            score = self.encoder.similarity(query_compressed, compressed)
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def stats(self) -> dict:
        """Get storage statistics."""
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT COUNT(*), AVG(compression_ratio) FROM {self.table_name}")
        count, avg_ratio = cursor.fetchone()

        cursor.execute(f"SELECT SUM(LENGTH(vector_data)) FROM {self.table_name}")
        total_bytes = cursor.fetchone()[0] or 0

        return {
            "vector_count": count,
            "avg_compression_ratio": f"{avg_ratio:.1f}x" if avg_ratio else "N/A",
            "total_compressed_bytes": total_bytes,
        }

    def close(self):
        if self._conn:
            self._conn.close()
