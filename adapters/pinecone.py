"""
TurboQuant Pinecone Adapter
==============================
Compressed metadata storage for Pinecone (reduces metadata costs).

Requirements: pip install pinecone-client

Usage:
    import pinecone
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.pinecone import PineconeTurboCache

    pc = pinecone.Pinecone(api_key="...")
    index = pc.Index("my-index")
    encoder = TurboQuantEncoder(dim=768)
    cache = PineconeTurboCache(encoder, index)

    # Store compressed backup alongside Pinecone vectors
    cache.put("doc:1", vector, metadata={"title": "Hello"})
    results = cache.search(query_vector, k=10)
"""

import base64
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class PineconeTurboCache(BaseTurboAdapter):
    """
    Pinecone adapter with TurboQuant compression.

    Strategy: Store the compressed vector as base64 in Pinecone metadata.
    This allows TurboQuant reranking after Pinecone's approximate search.

    Features:
    - Pinecone native ANN + TurboQuant reranking
    - Compressed backup in metadata (recover vectors without original source)
    - Batch upsert
    - Namespace support
    """

    def __init__(self, encoder: TurboQuantEncoder, index: Any,
                 namespace: str = ""):
        super().__init__(encoder)
        self.index = index
        self.namespace = namespace

    def _raw_get(self, key: str) -> Optional[bytes]:
        result = self.index.fetch(ids=[key], namespace=self.namespace)
        vec_data = result.get("vectors", {}).get(key)
        if vec_data and "tq_compressed" in vec_data.get("metadata", {}):
            return base64.b64decode(vec_data["metadata"]["tq_compressed"])
        return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        # Cannot upsert without the original vector for Pinecone
        raise NotImplementedError("Use put() which handles both vector and compressed data")

    def _raw_delete(self, key: str) -> bool:
        self.index.delete(ids=[key], namespace=self.namespace)
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        # Pinecone doesn't support key listing; use list() if available
        try:
            result = self.index.list(namespace=self.namespace)
            return [v for v in result.get("vectors", [])]
        except Exception:
            return []

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        """Upsert vector to Pinecone with compressed backup in metadata."""
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        meta = metadata.copy() if metadata else {}
        meta["tq_compressed"] = base64.b64encode(data).decode()
        meta["tq_ratio"] = round(compressed.compression_ratio(), 1)

        self.index.upsert(
            vectors=[{"id": key, "values": vector.tolist(), "metadata": meta}],
            namespace=self.namespace,
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
        """Batch upsert to Pinecone."""
        vectors_to_upsert = []
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            meta = (metadata or {}).get(key, {}).copy()
            meta["tq_compressed"] = base64.b64encode(data).decode()
            meta["tq_ratio"] = round(compressed.compression_ratio(), 1)

            vectors_to_upsert.append({
                "id": key, "values": vector.tolist(), "metadata": meta,
            })

            # Pinecone batch limit: 100 vectors
            if len(vectors_to_upsert) >= 100:
                self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                vectors_to_upsert = []

        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)

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
        """
        Search modes:
        - "pinecone": Native Pinecone ANN only
        - "rerank": Pinecone ANN candidates + TurboQuant reranking
        """
        query = np.asarray(query, dtype=np.float32).ravel()

        query_kwargs = {
            "vector": query.tolist(),
            "top_k": k * 3 if mode == "rerank" else k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if filter:
            query_kwargs["filter"] = filter

        results = self.index.query(**query_kwargs)

        if mode == "rerank":
            query_c = self.encoder.encode(query)
            reranked = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                if "tq_compressed" in meta:
                    compressed_data = base64.b64decode(meta["tq_compressed"])
                    candidate = CompressedVector.from_bytes(compressed_data)
                    score = self.encoder.similarity(query_c, candidate)
                    clean_meta = {k: v for k, v in meta.items()
                                  if k not in ("tq_compressed", "tq_ratio")}
                    reranked.append({
                        "id": match["id"],
                        "score": score,
                        "pinecone_score": match["score"],
                        "metadata": clean_meta,
                    })
            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked[:k]
        else:
            return [{
                "id": m["id"],
                "score": m["score"],
                "metadata": {k: v for k, v in m.get("metadata", {}).items()
                             if k not in ("tq_compressed", "tq_ratio")},
            } for m in results.get("matches", [])[:k]]
