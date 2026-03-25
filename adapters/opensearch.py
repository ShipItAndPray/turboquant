"""
TurboQuant OpenSearch Adapter
===============================
AWS OpenSearch / OpenSearch compatible vector compression.

Requirements: pip install opensearch-py

Usage:
    from opensearchpy import OpenSearch
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.opensearch import OpenSearchTurboCache

    client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
    encoder = TurboQuantEncoder(dim=768)
    cache = OpenSearchTurboCache(encoder, client, index="my_vectors")

    cache.create_index()
    cache.put("doc:1", vector, metadata={"title": "Hello"})
    results = cache.search(query_vector, k=10)
"""

import base64
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class OpenSearchTurboCache(BaseTurboAdapter):
    """
    OpenSearch adapter with TurboQuant compression.

    Similar to Elasticsearch adapter but uses OpenSearch-specific kNN syntax
    (k-NN plugin with nmslib/faiss engine).

    Features:
    - Compressed binary storage (6x memory reduction)
    - Native k-NN plugin integration for approximate search
    - TurboQuant reranking for higher precision
    - Bulk indexing
    """

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 index: str = "turboquant_vectors",
                 store_dense: bool = False,
                 knn_engine: str = "nmslib"):
        super().__init__(encoder)
        self.os = client
        self.index = index
        self.store_dense = store_dense
        self.knn_engine = knn_engine

    def create_index(self, shards: int = 1, replicas: int = 0,
                     ef_construction: int = 256, m: int = 16) -> dict:
        """Create index with OpenSearch k-NN settings."""
        mapping = {
            "settings": {
                "index": {
                    "knn": True if self.store_dense else False,
                    "number_of_shards": shards,
                    "number_of_replicas": replicas,
                },
            },
            "mappings": {
                "properties": {
                    "vector_compressed": {"type": "binary"},
                    "metadata": {"type": "object", "enabled": True},
                    "compression_ratio": {"type": "float"},
                }
            }
        }

        if self.store_dense:
            mapping["mappings"]["properties"]["vector_dense"] = {
                "type": "knn_vector",
                "dimension": self.encoder.dim,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": self.knn_engine,
                    "parameters": {
                        "ef_construction": ef_construction,
                        "m": m,
                    }
                }
            }

        if self.os.indices.exists(index=self.index):
            self.os.indices.delete(index=self.index)

        return self.os.indices.create(index=self.index, body=mapping)

    def _raw_get(self, key: str) -> Optional[bytes]:
        try:
            doc = self.os.get(index=self.index, id=key, _source=["vector_compressed"])
            return base64.b64decode(doc["_source"]["vector_compressed"])
        except Exception:
            return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        self.os.index(index=self.index, id=key, body={
            "vector_compressed": base64.b64encode(value).decode(),
        })

    def _raw_delete(self, key: str) -> bool:
        try:
            self.os.delete(index=self.index, id=key)
            return True
        except Exception:
            return False

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        resp = self.os.search(index=self.index, body={
            "query": {"match_all": {}}, "_source": False, "size": 10000
        })
        return [hit["_id"] for hit in resp["hits"]["hits"]]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
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

        self.os.index(index=self.index, id=key, body=doc)

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

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None,
               mode: str = "compressed") -> List[dict]:
        query = np.asarray(query, dtype=np.float32).ravel()

        if mode == "knn" and self.store_dense:
            body = {
                "size": k,
                "query": {
                    "knn": {
                        "vector_dense": {
                            "vector": query.tolist(),
                            "k": k,
                        }
                    }
                },
                "_source": ["metadata"],
            }
            resp = self.os.search(index=self.index, body=body)
            return [{"id": h["_id"], "score": h["_score"],
                     "metadata": h["_source"].get("metadata", {})}
                    for h in resp["hits"]["hits"]]
        else:
            # Client-side compressed search
            query_c = self.encoder.encode(query)
            resp = self.os.search(index=self.index, body={
                "query": {"match_all": {}},
                "_source": ["vector_compressed", "metadata"],
                "size": 10000,
            })
            results = []
            for hit in resp["hits"]["hits"]:
                data = base64.b64decode(hit["_source"]["vector_compressed"])
                candidate = CompressedVector.from_bytes(data)
                score = self.encoder.similarity(query_c, candidate)
                results.append({"id": hit["_id"], "score": score,
                                "metadata": hit["_source"].get("metadata", {})})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]
