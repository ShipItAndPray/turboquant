"""
TurboQuant FAISS Adapter
==========================
Compressed vector index using FAISS for fast ANN search.

Requirements: pip install faiss-cpu (or faiss-gpu)

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.faiss import FAISSTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = FAISSTurboCache(encoder)

    cache.put_batch({"doc:1": v1, "doc:2": v2, ...})
    results = cache.search(query_vector, k=10)
    cache.save("vectors.index")
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class FAISSTurboCache(BaseTurboAdapter):
    """
    FAISS adapter with TurboQuant compression.

    Maintains a FAISS index for fast ANN search plus a separate
    compressed vector store for TurboQuant reranking and recovery.

    Features:
    - FAISS IVF/HNSW for fast ANN
    - TurboQuant compressed backup (6x smaller than float32 index)
    - Save/load to disk
    - Reranking mode
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 index_type: str = "Flat"):
        """
        Args:
            index_type: FAISS index factory string. Examples:
                "Flat" - exact search (brute force)
                "IVF256,Flat" - inverted file index
                "HNSW32" - hierarchical navigable small world
        """
        super().__init__(encoder)
        import faiss

        self.faiss = faiss
        self.index = faiss.index_factory(encoder.dim, index_type)
        self._id_map: List[str] = []       # index position -> key
        self._compressed: Dict[str, bytes] = {}  # key -> compressed bytes
        self._trained = False

    def _raw_get(self, key: str) -> Optional[bytes]:
        return self._compressed.get(key)

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        self._compressed[key] = value

    def _raw_delete(self, key: str) -> bool:
        if key in self._compressed:
            del self._compressed[key]
            return True
        return False

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        return list(self._compressed.keys())

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()
        self._compressed[key] = data

        # Add to FAISS index
        self.index.add(vector.reshape(1, -1))
        self._id_map.append(key)

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
        keys = list(items.keys())
        vectors = np.array([np.asarray(v, dtype=np.float32).ravel() for v in items.values()])

        total_orig = 0
        total_comp = 0

        for key, vec in zip(keys, vectors):
            compressed = self.encoder.encode(vec)
            data = compressed.to_bytes()
            self._compressed[key] = data
            total_orig += len(vec) * 4
            total_comp += len(data)

        # Train index if needed (IVF indices)
        if not self._trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(vectors)
            self._trained = True

        self.index.add(vectors)
        self._id_map.extend(keys)

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
        """
        Search modes:
        - "faiss": FAISS ANN only
        - "rerank": FAISS candidates + TurboQuant reranking
        - "compressed": TurboQuant similarity only (no FAISS)
        """
        query = np.asarray(query, dtype=np.float32).ravel().reshape(1, -1)

        if mode == "compressed":
            return self._search_compressed(query.ravel(), k, keys)

        # FAISS search
        fetch_k = k * 3 if mode == "rerank" else k
        distances, indices = self.index.search(query, min(fetch_k, self.index.ntotal))

        if mode == "rerank":
            query_c = self.encoder.encode(query.ravel())
            reranked = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                doc_id = self._id_map[idx]
                data = self._compressed.get(doc_id)
                if data:
                    candidate = CompressedVector.from_bytes(data)
                    score = self.encoder.similarity(query_c, candidate)
                    reranked.append({"id": doc_id, "score": score, "faiss_dist": float(dist)})
            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked[:k]

        return [{"id": self._id_map[idx], "score": float(-dist)}
                for dist, idx in zip(distances[0], indices[0])
                if 0 <= idx < len(self._id_map)][:k]

    def _search_compressed(self, query, k, keys=None):
        query_c = self.encoder.encode(query)
        search_keys = keys or list(self._compressed.keys())
        results = []
        for key in search_keys:
            data = self._compressed.get(key)
            if data:
                candidate = CompressedVector.from_bytes(data)
                score = self.encoder.similarity(query_c, candidate)
                results.append({"id": key, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def save(self, path: str):
        """Save FAISS index + compressed vectors to disk."""
        import pickle
        self.faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.tq", "wb") as f:
            pickle.dump({"id_map": self._id_map, "compressed": self._compressed}, f)

    def load(self, path: str):
        """Load FAISS index + compressed vectors from disk."""
        import pickle
        self.index = self.faiss.read_index(f"{path}.faiss")
        with open(f"{path}.tq", "rb") as f:
            data = pickle.load(f)
            self._id_map = data["id_map"]
            self._compressed = data["compressed"]

    def memory_stats(self) -> dict:
        compressed_bytes = sum(len(v) for v in self._compressed.values())
        return {
            "faiss_vectors": self.index.ntotal,
            "compressed_vectors": len(self._compressed),
            "compressed_bytes": compressed_bytes,
            "avg_compressed_bytes": compressed_bytes // max(len(self._compressed), 1),
        }
