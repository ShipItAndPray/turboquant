"""
TurboQuant Qdrant Adapter
============================
Requirements: pip install qdrant-client

Usage:
    from qdrant_client import QdrantClient
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.qdrant import QdrantTurboCache

    client = QdrantClient("localhost", port=6333)
    encoder = TurboQuantEncoder(dim=768)
    cache = QdrantTurboCache(encoder, client, collection="my_vectors")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    results = cache.search(query_vector, k=10)
"""

import base64
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class QdrantTurboCache(BaseTurboAdapter):
    """
    Qdrant adapter with TurboQuant compression.

    Features:
    - Native HNSW search + TurboQuant reranking
    - Compressed backup in payload
    - Batch upsert
    - Payload filtering
    """

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 collection: str = "turboquant_vectors",
                 create_collection: bool = True):
        super().__init__(encoder)
        self.client = client
        self.collection = collection

        if create_collection:
            self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.encoder.dim, distance=Distance.COSINE
                ),
            )

    def _raw_get(self, key: str) -> Optional[bytes]:
        results = self.client.retrieve(
            collection_name=self.collection, ids=[key], with_payload=True
        )
        if results and "tq_compressed" in results[0].payload:
            return base64.b64decode(results[0].payload["tq_compressed"])
        return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        raise NotImplementedError("Use put()")

    def _raw_delete(self, key: str) -> bool:
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[key])
        )
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        # Qdrant scroll API
        results, _ = self.client.scroll(
            collection_name=self.collection, limit=10000, with_payload=False
        )
        return [str(p.id) for p in results]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        from qdrant_client.models import PointStruct

        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        payload = metadata.copy() if metadata else {}
        payload["tq_compressed"] = base64.b64encode(data).decode()
        payload["tq_ratio"] = round(compressed.compression_ratio(), 1)

        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=key, vector=vector.tolist(), payload=payload)]
        )

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

    def put_batch(self, items: Dict[str, np.ndarray],
                  metadata: Optional[Dict[str, dict]] = None,
                  ttl: Optional[int] = None) -> dict:
        from qdrant_client.models import PointStruct

        points = []
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            payload = (metadata or {}).get(key, {}).copy()
            payload["tq_compressed"] = base64.b64encode(data).decode()

            points.append(PointStruct(id=key, vector=vector.tolist(), payload=payload))

        self.client.upsert(collection_name=self.collection, points=points)
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
               mode: str = "rerank",
               filter: Optional[dict] = None) -> List[dict]:
        query = np.asarray(query, dtype=np.float32).ravel()

        search_params = {
            "collection_name": self.collection,
            "query_vector": query.tolist(),
            "limit": k * 3 if mode == "rerank" else k,
            "with_payload": True,
        }
        if filter:
            from qdrant_client.models import Filter
            search_params["query_filter"] = Filter(**filter)

        results = self.client.search(**search_params)

        if mode == "rerank":
            query_c = self.encoder.encode(query)
            reranked = []
            for hit in results:
                if "tq_compressed" in hit.payload:
                    data = base64.b64decode(hit.payload["tq_compressed"])
                    candidate = CompressedVector.from_bytes(data)
                    score = self.encoder.similarity(query_c, candidate)
                    meta = {k: v for k, v in hit.payload.items()
                            if k not in ("tq_compressed", "tq_ratio")}
                    reranked.append({"id": str(hit.id), "score": score,
                                     "qdrant_score": hit.score, "metadata": meta})
            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked[:k]
        else:
            return [{"id": str(h.id), "score": h.score,
                     "metadata": {k: v for k, v in h.payload.items()
                                  if k not in ("tq_compressed", "tq_ratio")}}
                    for h in results[:k]]
