"""
TurboQuant ChromaDB Adapter
==============================
Requirements: pip install chromadb

Usage:
    import chromadb
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.chromadb import ChromaTurboCache

    chroma = chromadb.Client()
    encoder = TurboQuantEncoder(dim=768)
    cache = ChromaTurboCache(encoder, chroma, collection="my_vectors")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    results = cache.search(query_vector, k=10)
"""

import base64
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class ChromaTurboCache(BaseTurboAdapter):
    """ChromaDB adapter with TurboQuant compression in metadata."""

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 collection: str = "turboquant_vectors"):
        super().__init__(encoder)
        self.collection = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def _raw_get(self, key: str) -> Optional[bytes]:
        result = self.collection.get(ids=[key], include=["metadatas"])
        if result["ids"] and result["metadatas"][0].get("tq_compressed"):
            return base64.b64decode(result["metadatas"][0]["tq_compressed"])
        return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        raise NotImplementedError("Use put()")

    def _raw_delete(self, key: str) -> bool:
        self.collection.delete(ids=[key])
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        result = self.collection.get(include=[])
        return result["ids"]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        meta = {}
        if metadata:
            # ChromaDB metadata must be flat string/int/float
            for k, v in metadata.items():
                meta[k] = str(v) if not isinstance(v, (int, float, str)) else v
        meta["tq_compressed"] = base64.b64encode(data).decode()

        self.collection.upsert(
            ids=[key],
            embeddings=[vector.tolist()],
            metadatas=[meta],
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
        ids, embeddings, metadatas = [], [], []
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            meta = {}
            if metadata and key in metadata:
                for k, v in metadata[key].items():
                    meta[k] = str(v) if not isinstance(v, (int, float, str)) else v
            meta["tq_compressed"] = base64.b64encode(data).decode()

            ids.append(key)
            embeddings.append(vector.tolist())
            metadatas.append(meta)

        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
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
               where: Optional[dict] = None) -> List[dict]:
        query = np.asarray(query, dtype=np.float32).ravel()

        query_kwargs = {
            "query_embeddings": [query.tolist()],
            "n_results": k * 3 if mode == "rerank" else k,
            "include": ["metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        if mode == "rerank" and results["ids"][0]:
            query_c = self.encoder.encode(query)
            reranked = []
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                if "tq_compressed" in meta:
                    data = base64.b64decode(meta["tq_compressed"])
                    candidate = CompressedVector.from_bytes(data)
                    score = self.encoder.similarity(query_c, candidate)
                    clean = {k: v for k, v in meta.items() if k != "tq_compressed"}
                    reranked.append({"id": doc_id, "score": score, "metadata": clean})
            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked[:k]
        else:
            return [{"id": results["ids"][0][i],
                     "score": 1 - results["distances"][0][i],
                     "metadata": {k: v for k, v in results["metadatas"][0][i].items()
                                  if k != "tq_compressed"}}
                    for i in range(min(k, len(results["ids"][0])))]
