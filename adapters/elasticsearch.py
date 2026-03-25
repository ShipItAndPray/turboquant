"""
TurboQuant Elasticsearch Adapter
==================================
Compressed vector storage + hybrid kNN search with TurboQuant reranking.

Requirements: pip install elasticsearch

Usage:
    from elasticsearch import Elasticsearch
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.elasticsearch import ElasticsearchTurboCache

    es = Elasticsearch("http://localhost:9200")
    encoder = TurboQuantEncoder(dim=768)
    cache = ElasticsearchTurboCache(encoder, es, index="my_vectors")

    cache.create_index()
    cache.put("doc:1", vector, metadata={"title": "Hello"})
    cache.bulk_put({"doc:2": v2, "doc:3": v3})
    results = cache.search(query_vector, k=10)
"""

import base64
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class ElasticsearchTurboCache(BaseTurboAdapter):
    """
    Elasticsearch adapter with TurboQuant compression.

    Storage modes:
    1. Compressed-only: Store binary field only (maximum compression)
    2. Hybrid: Store both compressed + dense_vector (ES kNN + TQ rerank)

    Search modes:
    1. Client-side: Fetch compressed vectors, compute similarity locally
    2. ES kNN + rerank: Use ES approximate kNN, rerank with TurboQuant
    3. ES kNN only: Use dense_vector field directly
    """

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 index: str = "turboquant_vectors",
                 store_dense: bool = False):
        """
        Args:
            client: elasticsearch.Elasticsearch instance
            index: Index name
            store_dense: If True, also store full dense_vector for ES native kNN
        """
        super().__init__(encoder)
        self.es = client
        self.index = index
        self.store_dense = store_dense

    def create_index(self, shards: int = 1, replicas: int = 0) -> dict:
        """Create an optimized index for compressed vector storage."""
        mapping = {
            "mappings": {
                "properties": {
                    "vector_compressed": {"type": "binary"},
                    "metadata": {"type": "object", "enabled": True},
                    "compression_ratio": {"type": "float"},
                }
            },
            "settings": {
                "number_of_shards": shards,
                "number_of_replicas": replicas,
                "refresh_interval": "30s",
            }
        }

        if self.store_dense:
            mapping["mappings"]["properties"]["vector_dense"] = {
                "type": "dense_vector",
                "dims": self.encoder.dim,
                "index": True,
                "similarity": "cosine",
            }

        if self.es.indices.exists(index=self.index):
            self.es.indices.delete(index=self.index)

        return self.es.indices.create(index=self.index, body=mapping)

    def _raw_get(self, key: str) -> Optional[bytes]:
        try:
            doc = self.es.get(index=self.index, id=key, _source=["vector_compressed"])
            b64 = doc["_source"]["vector_compressed"]
            return base64.b64decode(b64)
        except Exception:
            return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        doc = {
            "vector_compressed": base64.b64encode(value).decode(),
        }
        self.es.index(index=self.index, id=key, body=doc)

    def _raw_delete(self, key: str) -> bool:
        try:
            self.es.delete(index=self.index, id=key)
            return True
        except Exception:
            return False

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        keys = []
        body = {"query": {"match_all": {}}, "_source": False, "size": 10000}
        resp = self.es.search(index=self.index, body=body)
        for hit in resp["hits"]["hits"]:
            keys.append(hit["_id"])
        return keys

    # --- Enhanced put with metadata + dense vector ---

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        """Store vector with optional metadata and dense_vector field."""
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        doc = {
            "vector_compressed": base64.b64encode(data).decode(),
            "compression_ratio": compressed.compression_ratio(),
            "metadata": metadata or {},
        }

        if self.store_dense:
            doc["vector_dense"] = vector.tolist()

        self.es.index(index=self.index, id=key, body=doc)

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
                 metadata: Optional[Dict[str, dict]] = None,
                 chunk_size: int = 500) -> dict:
        """Bulk index vectors using ES _bulk API."""
        actions = []
        total_orig = 0
        total_comp = 0

        for doc_id, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            doc = {
                "vector_compressed": base64.b64encode(data).decode(),
                "compression_ratio": compressed.compression_ratio(),
                "metadata": (metadata or {}).get(doc_id, {}),
            }
            if self.store_dense:
                doc["vector_dense"] = vector.tolist()

            actions.append(json.dumps({"index": {"_index": self.index, "_id": doc_id}}))
            actions.append(json.dumps(doc))

            if len(actions) >= chunk_size * 2:
                self.es.bulk(body="\n".join(actions) + "\n")
                actions = []

        if actions:
            self.es.bulk(body="\n".join(actions) + "\n")

        self.es.indices.refresh(index=self.index)
        self._stats["puts"] += len(items)
        self._stats["bytes_original"] += total_orig
        self._stats["bytes_compressed"] += total_comp

        return {
            "indexed": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None,
               mode: str = "compressed") -> List[dict]:
        """
        Search for similar vectors.

        Modes:
        - "compressed": Fetch all, compute TurboQuant similarity client-side
        - "knn": Use ES native kNN (requires store_dense=True)
        - "knn_rerank": ES kNN candidates, rerank with TurboQuant similarity
        """
        query = np.asarray(query, dtype=np.float32).ravel()

        if mode == "knn" and self.store_dense:
            return self._search_knn(query, k)
        elif mode == "knn_rerank" and self.store_dense:
            return self._search_knn_rerank(query, k)
        else:
            return self._search_compressed(query, k, keys)

    def _search_compressed(self, query: np.ndarray, k: int,
                           keys: Optional[List[str]] = None) -> List[dict]:
        """Client-side search over compressed vectors."""
        query_c = self.encoder.encode(query)

        body = {
            "query": {"match_all": {}},
            "_source": ["vector_compressed", "metadata"],
            "size": 10000,
        }
        resp = self.es.search(index=self.index, body=body)

        results = []
        for hit in resp["hits"]["hits"]:
            if keys and hit["_id"] not in keys:
                continue
            data = base64.b64decode(hit["_source"]["vector_compressed"])
            candidate = CompressedVector.from_bytes(data)
            score = self.encoder.similarity(query_c, candidate)
            results.append({
                "id": hit["_id"],
                "score": score,
                "metadata": hit["_source"].get("metadata", {}),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _search_knn(self, query: np.ndarray, k: int) -> List[dict]:
        """ES native approximate kNN search."""
        body = {
            "knn": {
                "field": "vector_dense",
                "query_vector": query.tolist(),
                "k": k,
                "num_candidates": max(k * 10, 100),
            },
            "_source": ["metadata"],
        }
        resp = self.es.search(index=self.index, body=body)
        return [{
            "id": hit["_id"],
            "score": hit["_score"],
            "metadata": hit["_source"].get("metadata", {}),
        } for hit in resp["hits"]["hits"]]

    def _search_knn_rerank(self, query: np.ndarray, k: int) -> List[dict]:
        """ES kNN for candidates, TurboQuant rerank for precision."""
        body = {
            "knn": {
                "field": "vector_dense",
                "query_vector": query.tolist(),
                "k": k * 3,
                "num_candidates": max(k * 10, 100),
            },
            "_source": ["vector_compressed", "metadata"],
        }
        resp = self.es.search(index=self.index, body=body)

        query_c = self.encoder.encode(query)
        results = []
        for hit in resp["hits"]["hits"]:
            data = base64.b64decode(hit["_source"]["vector_compressed"])
            candidate = CompressedVector.from_bytes(data)
            score = self.encoder.similarity(query_c, candidate)
            results.append({
                "id": hit["_id"],
                "score": score,
                "es_score": hit["_score"],
                "metadata": hit["_source"].get("metadata", {}),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def index_stats(self) -> dict:
        stats = self.es.indices.stats(index=self.index)
        idx = stats["indices"][self.index]["total"]
        return {
            "doc_count": idx["docs"]["count"],
            "store_bytes": idx["store"]["size_in_bytes"],
            "store_human": f"{idx['store']['size_in_bytes'] / 1e6:.1f} MB",
            **self.stats(),
        }
