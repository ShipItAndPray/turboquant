"""
TurboQuant Azure Blob Storage Adapter
========================================
Requirements: pip install azure-storage-blob

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.azure_blob import AzureBlobTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = AzureBlobTurboCache(encoder, connection_string="...", container="vectors")

    cache.put("doc:1", vector)
    vec = cache.get("doc:1")
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class AzureBlobTurboCache(BaseTurboAdapter):
    """Azure Blob Storage adapter with TurboQuant compression."""

    def __init__(self, encoder: TurboQuantEncoder,
                 connection_string: str = None,
                 container: str = "turboquant-vectors",
                 prefix: str = "tq/",
                 client: Any = None):
        super().__init__(encoder)
        self.prefix = prefix

        if client:
            self.container_client = client
        else:
            from azure.storage.blob import BlobServiceClient
            service = BlobServiceClient.from_connection_string(connection_string)
            self.container_client = service.get_container_client(container)
            try:
                self.container_client.create_container()
            except Exception:
                pass

    def _blob_name(self, key: str) -> str:
        return f"{self.prefix}{key}.tq"

    def _raw_get(self, key: str) -> Optional[bytes]:
        try:
            blob = self.container_client.get_blob_client(self._blob_name(key))
            return blob.download_blob().readall()
        except Exception:
            return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        blob = self.container_client.get_blob_client(self._blob_name(key))
        blob.upload_blob(value, overwrite=True,
                         content_settings={"content_type": "application/octet-stream"})

    def _raw_delete(self, key: str) -> bool:
        try:
            blob = self.container_client.get_blob_client(self._blob_name(key))
            blob.delete_blob()
            return True
        except Exception:
            return False

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        blobs = self.container_client.list_blobs(name_starts_with=self.prefix)
        return [b.name[len(self.prefix):-3] for b in blobs if b.name.endswith(".tq")]

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        blob = self.container_client.get_blob_client(self._blob_name(key))
        blob_metadata = {k: str(v) for k, v in metadata.items()} if metadata else None
        blob.upload_blob(data, overwrite=True, metadata=blob_metadata)

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
