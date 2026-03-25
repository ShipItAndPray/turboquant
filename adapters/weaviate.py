"""
TurboQuant Weaviate Adapter
==============================
Requirements: pip install weaviate-client

Usage:
    import weaviate
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.weaviate import WeaviateTurboCache

    client = weaviate.Client("http://localhost:8080")
    encoder = TurboQuantEncoder(dim=768)
    cache = WeaviateTurboCache(encoder, client, class_name="Document")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    results = cache.search(query_vector, k=10)
"""

import base64
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class WeaviateTurboCache(BaseTurboAdapter):
    """Weaviate adapter with TurboQuant compression."""

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 class_name: str = "TurboQuantVector",
                 create_class: bool = True):
        super().__init__(encoder)
        self.client = client
        self.class_name = class_name

        if create_class:
            self._ensure_class()

    def _ensure_class(self):
        try:
            self.client.schema.get(self.class_name)
        except Exception:
            schema = {
                "class": self.class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "vector_id", "dataType": ["text"]},
                    {"name": "tq_compressed", "dataType": ["text"]},
                    {"name": "metadata_json", "dataType": ["text"]},
                ],
            }
            self.client.schema.create_class(schema)

    def _raw_get(self, key: str) -> Optional[bytes]:
        result = (self.client.query
                  .get(self.class_name, ["tq_compressed"])
                  .with_where({"path": ["vector_id"], "operator": "Equal", "valueText": key})
                  .with_limit(1).do())
        data = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        if data and data[0].get("tq_compressed"):
            return base64.b64decode(data[0]["tq_compressed"])
        return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        raise NotImplementedError("Use put()")

    def _raw_delete(self, key: str) -> bool:
        self.client.batch.delete_objects(
            class_name=self.class_name,
            where={"path": ["vector_id"], "operator": "Equal", "valueText": key},
        )
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        result = (self.client.query
                  .get(self.class_name, ["vector_id"])
                  .with_limit(10000).do())
        data = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        return [d["vector_id"] for d in data if "vector_id" in d]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        import json
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        props = {
            "vector_id": key,
            "tq_compressed": base64.b64encode(data).decode(),
            "metadata_json": json.dumps(metadata) if metadata else "{}",
        }

        self.client.data_object.create(
            data_object=props,
            class_name=self.class_name,
            vector=vector.tolist(),
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

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None,
               mode: str = "rerank") -> List[dict]:
        import json
        query = np.asarray(query, dtype=np.float32).ravel()

        result = (self.client.query
                  .get(self.class_name, ["vector_id", "tq_compressed", "metadata_json"])
                  .with_near_vector({"vector": query.tolist()})
                  .with_limit(k * 3 if mode == "rerank" else k)
                  .with_additional(["distance"]).do())

        hits = result.get("data", {}).get("Get", {}).get(self.class_name, [])

        if mode == "rerank":
            query_c = self.encoder.encode(query)
            reranked = []
            for hit in hits:
                if hit.get("tq_compressed"):
                    candidate = CompressedVector.from_bytes(base64.b64decode(hit["tq_compressed"]))
                    score = self.encoder.similarity(query_c, candidate)
                    meta = json.loads(hit.get("metadata_json", "{}"))
                    reranked.append({"id": hit["vector_id"], "score": score, "metadata": meta})
            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked[:k]

        return [{"id": h["vector_id"],
                 "score": 1 - float(h.get("_additional", {}).get("distance", 1)),
                 "metadata": json.loads(h.get("metadata_json", "{}"))}
                for h in hits[:k]]
