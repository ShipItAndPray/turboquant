"""
TurboQuant MongoDB Adapter
============================
Compressed vector storage in MongoDB with Atlas Vector Search support.

Requirements: pip install pymongo

Usage:
    from pymongo import MongoClient
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.mongodb import MongoTurboCache

    client = MongoClient("mongodb://localhost:27017")
    encoder = TurboQuantEncoder(dim=768)
    cache = MongoTurboCache(encoder, client, db="myapp", collection="vectors")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    vec = cache.get("doc:1")
    results = cache.search(query_vector, k=10)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class MongoTurboCache(BaseTurboAdapter):
    """
    MongoDB adapter with TurboQuant compression.

    Features:
    - Binary storage using BSON Binary type
    - Atlas Vector Search integration for ANN
    - Bulk write operations
    - Flexible metadata queries via MongoDB query language
    - TTL index support
    """

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 db: str = "turboquant",
                 collection: str = "vectors",
                 store_dense: bool = False,
                 ttl: Optional[int] = None):
        super().__init__(encoder)
        self.db = client[db]
        self.coll = self.db[collection]
        self.store_dense = store_dense
        self.default_ttl = ttl
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.coll.create_index("_vector_id", unique=True, sparse=True)
        if self.default_ttl:
            self.coll.create_index("created_at", expireAfterSeconds=self.default_ttl)

    def _raw_get(self, key: str) -> Optional[bytes]:
        doc = self.coll.find_one({"_vector_id": key}, {"vector_data": 1})
        if doc and "vector_data" in doc:
            return bytes(doc["vector_data"])
        return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        import bson
        from datetime import datetime
        self.coll.update_one(
            {"_vector_id": key},
            {"$set": {
                "_vector_id": key,
                "vector_data": bson.Binary(value),
                "created_at": datetime.utcnow(),
            }},
            upsert=True,
        )

    def _raw_delete(self, key: str) -> bool:
        result = self.coll.delete_one({"_vector_id": key})
        return result.deleted_count > 0

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        if pattern == "*":
            cursor = self.coll.find({}, {"_vector_id": 1})
        else:
            import re
            regex = pattern.replace("*", ".*")
            cursor = self.coll.find({"_vector_id": {"$regex": f"^{regex}$"}}, {"_vector_id": 1})
        return [doc["_vector_id"] for doc in cursor if "_vector_id" in doc]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        import bson
        from datetime import datetime

        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        doc = {
            "_vector_id": key,
            "vector_data": bson.Binary(data),
            "original_dim": self.encoder.dim,
            "compression_ratio": compressed.compression_ratio(),
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
        }

        if self.store_dense:
            doc["vector_dense"] = vector.tolist()

        self.coll.update_one({"_vector_id": key}, {"$set": doc}, upsert=True)

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
        """Bulk upsert using MongoDB bulk_write."""
        import bson
        from pymongo import UpdateOne
        from datetime import datetime

        ops = []
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            doc = {
                "_vector_id": key,
                "vector_data": bson.Binary(data),
                "original_dim": self.encoder.dim,
                "compression_ratio": compressed.compression_ratio(),
                "metadata": (metadata or {}).get(key, {}),
                "created_at": datetime.utcnow(),
            }
            if self.store_dense:
                doc["vector_dense"] = vector.tolist()

            ops.append(UpdateOne({"_vector_id": key}, {"$set": doc}, upsert=True))

        if ops:
            self.coll.bulk_write(ops, ordered=False)

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
               metadata_filter: Optional[dict] = None) -> List[dict]:
        """
        Search modes:
        - "compressed": Client-side TurboQuant similarity
        - "atlas": MongoDB Atlas Vector Search ($vectorSearch)
        """
        query = np.asarray(query, dtype=np.float32).ravel()

        if mode == "atlas" and self.store_dense:
            return self._search_atlas(query, k, metadata_filter)

        return self._search_compressed(query, k, keys, metadata_filter)

    def _search_compressed(self, query, k, keys=None, metadata_filter=None):
        query_c = self.encoder.encode(query)
        find_filter = {}
        if keys:
            find_filter["_vector_id"] = {"$in": keys}
        if metadata_filter:
            for mkey, mval in metadata_filter.items():
                find_filter[f"metadata.{mkey}"] = mval

        cursor = self.coll.find(find_filter, {"_vector_id": 1, "vector_data": 1, "metadata": 1})

        results = []
        for doc in cursor:
            candidate = CompressedVector.from_bytes(bytes(doc["vector_data"]))
            score = self.encoder.similarity(query_c, candidate)
            results.append({
                "id": doc["_vector_id"],
                "score": score,
                "metadata": doc.get("metadata", {}),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _search_atlas(self, query, k, metadata_filter=None):
        """MongoDB Atlas Vector Search aggregation pipeline."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "vector_dense",
                    "queryVector": query.tolist(),
                    "numCandidates": max(k * 10, 100),
                    "limit": k,
                }
            },
            {
                "$project": {
                    "_vector_id": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

        if metadata_filter:
            pipeline[0]["$vectorSearch"]["filter"] = metadata_filter

        results = list(self.coll.aggregate(pipeline))
        return [{"id": r["_vector_id"], "score": r["score"],
                 "metadata": r.get("metadata", {})} for r in results]

    def collection_stats(self) -> dict:
        stats = self.db.command("collstats", self.coll.name)
        return {
            "count": stats.get("count", 0),
            "storage_bytes": stats.get("storageSize", 0),
            "avg_doc_bytes": stats.get("avgObjSize", 0),
            "index_bytes": stats.get("totalIndexSize", 0),
        }
