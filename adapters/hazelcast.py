"""
TurboQuant Hazelcast Adapter
==============================
Distributed cache compression via Hazelcast Python client.

Requirements: pip install hazelcast-python-client

Usage:
    import hazelcast
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.hazelcast import HazelcastTurboCache

    hz = hazelcast.HazelcastClient(cluster_members=["localhost:5701"])
    encoder = TurboQuantEncoder(dim=768)
    cache = HazelcastTurboCache(encoder, hz, map_name="vectors")

    cache.put("doc:1", vector, ttl=3600)
    vec = cache.get("doc:1")
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class HazelcastTurboCache(BaseTurboAdapter):
    """
    Hazelcast adapter with TurboQuant compression.

    Features:
    - Distributed across Hazelcast cluster nodes
    - TTL support via IMap.put(key, value, ttl)
    - Batch via put_all/get_all
    - Near-cache friendly (compressed values are small)
    - Entry processor support for server-side operations
    """

    def __init__(self, encoder: TurboQuantEncoder, client: Any,
                 map_name: str = "turboquant_vectors"):
        """
        Args:
            client: hazelcast.HazelcastClient instance
            map_name: Distributed map name
        """
        super().__init__(encoder)
        self.hz = client
        self.map_name = map_name
        self._map = client.get_map(map_name).blocking()

    def _raw_get(self, key: str) -> Optional[bytes]:
        result = self._map.get(key)
        if result is None:
            return None
        return result if isinstance(result, bytes) else result.encode()

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        if ttl:
            self._map.put(key, value, ttl)
        else:
            self._map.put(key, value)

    def _raw_delete(self, key: str) -> bool:
        return self._map.remove(key) is not None

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        return [str(k) for k in self._map.key_set()]

    def put_batch(self, items: Dict[str, np.ndarray], ttl: Optional[int] = None) -> dict:
        """Batch store using Hazelcast put_all."""
        to_put = {}
        total_orig = 0
        total_comp = 0

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            to_put[key] = data
            total_orig += len(vector) * 4
            total_comp += len(data)

        self._map.put_all(to_put)
        self._stats["puts"] += len(items)
        self._stats["bytes_original"] += total_orig
        self._stats["bytes_compressed"] += total_comp

        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Batch retrieve using Hazelcast get_all."""
        results = self._map.get_all(keys)
        output = {}
        for key in keys:
            self._stats["gets"] += 1
            data = results.get(key)
            if data is not None:
                self._stats["hits"] += 1
                compressed = CompressedVector.from_bytes(
                    data if isinstance(data, bytes) else data.encode()
                )
                output[key] = self.encoder.decode(compressed)
            else:
                self._stats["misses"] += 1
                output[key] = None
        return output

    def size(self) -> int:
        return self._map.size()

    def clear(self) -> None:
        """Clear entire distributed map."""
        self._map.clear()

    def close(self):
        self.hz.shutdown()
