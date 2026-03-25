"""
TurboQuant Google Cloud Storage Adapter
=========================================
Requirements: pip install google-cloud-storage

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.gcs import GCSTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = GCSTurboCache(encoder, bucket="my-vectors")

    cache.put("doc:1", vector)
    vec = cache.get("doc:1")
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class GCSTurboCache(BaseTurboAdapter):
    """Google Cloud Storage adapter with TurboQuant compression."""

    def __init__(self, encoder: TurboQuantEncoder,
                 bucket: str,
                 prefix: str = "tq/",
                 client: Any = None):
        super().__init__(encoder)
        self.prefix = prefix

        if client:
            self.storage_client = client
        else:
            from google.cloud import storage
            self.storage_client = storage.Client()

        self.bucket = self.storage_client.bucket(bucket)

    def _blob_name(self, key: str) -> str:
        return f"{self.prefix}{key}.tq"

    def _raw_get(self, key: str) -> Optional[bytes]:
        blob = self.bucket.blob(self._blob_name(key))
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        blob = self.bucket.blob(self._blob_name(key))
        blob.upload_from_string(value, content_type="application/octet-stream")

    def _raw_delete(self, key: str) -> bool:
        blob = self.bucket.blob(self._blob_name(key))
        if blob.exists():
            blob.delete()
            return True
        return False

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        blobs = self.storage_client.list_blobs(self.bucket, prefix=self.prefix)
        keys = []
        for blob in blobs:
            if blob.name.endswith(".tq"):
                keys.append(blob.name[len(self.prefix):-3])
        return keys

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        blob = self.bucket.blob(self._blob_name(key))
        if metadata:
            blob.metadata = {k: str(v) for k, v in metadata.items()}
        blob.upload_from_string(data, content_type="application/octet-stream")

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
        from concurrent.futures import ThreadPoolExecutor
        total_orig = 0
        total_comp = 0

        def _upload(kv):
            return self.put(kv[0], kv[1], ttl=ttl)

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(_upload, items.items()))

        for r in results:
            total_orig += r["original_bytes"]
            total_comp += r["compressed_bytes"]

        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }
