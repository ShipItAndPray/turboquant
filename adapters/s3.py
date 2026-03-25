"""
TurboQuant S3 Adapter
======================
Compressed vector storage in AWS S3.

Requirements: pip install boto3

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.s3 import S3TurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = S3TurboCache(encoder, bucket="my-vectors", prefix="embeddings/")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    vec = cache.get("doc:1")
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class S3TurboCache(BaseTurboAdapter):
    """
    AWS S3 adapter with TurboQuant compression.

    Features:
    - Each vector stored as a single S3 object (~500 bytes vs ~3KB raw)
    - Metadata stored as S3 object metadata headers
    - Batch upload via multipart/concurrent
    - List-based key enumeration
    - Versioning-aware
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 bucket: str,
                 prefix: str = "tq/",
                 region: str = "us-east-1",
                 client: Any = None):
        super().__init__(encoder)
        self.bucket = bucket
        self.prefix = prefix

        if client:
            self.s3 = client
        else:
            import boto3
            self.s3 = boto3.client("s3", region_name=region)

    def _s3key(self, key: str) -> str:
        return f"{self.prefix}{key}.tq"

    def _raw_get(self, key: str) -> Optional[bytes]:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self._s3key(key))
            return resp["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                return None
            raise

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        kwargs = {
            "Bucket": self.bucket,
            "Key": self._s3key(key),
            "Body": value,
            "ContentType": "application/octet-stream",
        }
        if ttl:
            from datetime import datetime, timedelta
            kwargs["Expires"] = datetime.utcnow() + timedelta(seconds=ttl)
        self.s3.put_object(**kwargs)

    def _raw_delete(self, key: str) -> bool:
        self.s3.delete_object(Bucket=self.bucket, Key=self._s3key(key))
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.startswith(self.prefix) and k.endswith(".tq"):
                    clean = k[len(self.prefix):-3]
                    keys.append(clean)
        return keys

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        kwargs = {
            "Bucket": self.bucket,
            "Key": self._s3key(key),
            "Body": data,
            "ContentType": "application/octet-stream",
        }
        if metadata:
            # S3 metadata must be string values, max 2KB total
            kwargs["Metadata"] = {k: str(v) for k, v in metadata.items()}
        if ttl:
            from datetime import datetime, timedelta
            kwargs["Expires"] = datetime.utcnow() + timedelta(seconds=ttl)

        self.s3.put_object(**kwargs)

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
        """Batch upload using ThreadPoolExecutor for concurrency."""
        from concurrent.futures import ThreadPoolExecutor
        total_orig = 0
        total_comp = 0

        def _upload(key_vec):
            key, vector = key_vec
            return self.put(key, vector, ttl=ttl)

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

    def bucket_stats(self) -> dict:
        """Get storage stats for TurboQuant objects in bucket."""
        total_size = 0
        count = 0
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                total_size += obj["Size"]
                count += 1
        return {
            "object_count": count,
            "total_bytes": total_size,
            "total_human": f"{total_size / 1e6:.1f} MB" if total_size > 1e6 else f"{total_size / 1e3:.1f} KB",
            "avg_bytes": total_size // max(count, 1),
        }
