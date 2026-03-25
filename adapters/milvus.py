"""
TurboQuant Milvus Adapter
============================
Requirements: pip install pymilvus

Usage:
    from pymilvus import connections
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.milvus import MilvusTurboCache

    connections.connect("default", host="localhost", port="19530")
    encoder = TurboQuantEncoder(dim=768)
    cache = MilvusTurboCache(encoder, collection="my_vectors")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    results = cache.search(query_vector, k=10)
"""

import base64
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class MilvusTurboCache(BaseTurboAdapter):
    """
    Milvus adapter with TurboQuant compression.

    Features:
    - IVF_FLAT/HNSW index for ANN search
    - Compressed backup in VARCHAR field
    - Batch insert
    - Partition support
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 collection: str = "turboquant_vectors",
                 create: bool = True):
        super().__init__(encoder)
        self.collection_name = collection

        if create:
            self._ensure_collection()

        from pymilvus import Collection
        self.collection = Collection(collection)
        self.collection.load()

    def _ensure_collection(self):
        from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility

        if utility.has_collection(self.collection_name):
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.encoder.dim),
            FieldSchema(name="tq_compressed", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields)
        col = Collection(self.collection_name, schema)

        # Create HNSW index
        col.create_index("vector", {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256},
        })

    def _raw_get(self, key: str) -> Optional[bytes]:
        results = self.collection.query(
            expr=f'id == "{key}"',
            output_fields=["tq_compressed"]
        )
        if results and results[0].get("tq_compressed"):
            return base64.b64decode(results[0]["tq_compressed"])
        return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        raise NotImplementedError("Use put()")

    def _raw_delete(self, key: str) -> bool:
        self.collection.delete(expr=f'id == "{key}"')
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        results = self.collection.query(expr="id != ''", output_fields=["id"], limit=10000)
        return [r["id"] for r in results]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        import json
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        self.collection.insert([
            [key],
            [vector.tolist()],
            [base64.b64encode(data).decode()],
            [json.dumps(metadata) if metadata else "{}"],
        ])

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
        import json
        ids, vectors, compressed_list, meta_list = [], [], [], []
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            ids.append(key)
            vectors.append(vector.tolist())
            compressed_list.append(base64.b64encode(data).decode())
            meta_list.append(json.dumps((metadata or {}).get(key, {})))

        self.collection.insert([ids, vectors, compressed_list, meta_list])
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
               mode: str = "rerank") -> List[dict]:
        import json
        query = np.asarray(query, dtype=np.float32).ravel()

        results = self.collection.search(
            data=[query.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": max(k * 10, 64)}},
            limit=k * 3 if mode == "rerank" else k,
            output_fields=["tq_compressed", "metadata_json"],
        )

        if mode == "rerank":
            query_c = self.encoder.encode(query)
            reranked = []
            for hit in results[0]:
                if hit.entity.get("tq_compressed"):
                    data = base64.b64decode(hit.entity.get("tq_compressed"))
                    candidate = CompressedVector.from_bytes(data)
                    score = self.encoder.similarity(query_c, candidate)
                    meta = json.loads(hit.entity.get("metadata_json", "{}"))
                    reranked.append({"id": hit.id, "score": score,
                                     "milvus_score": hit.distance, "metadata": meta})
            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked[:k]

        return [{"id": h.id, "score": h.distance,
                 "metadata": json.loads(h.entity.get("metadata_json", "{}"))}
                for h in results[0][:k]]
