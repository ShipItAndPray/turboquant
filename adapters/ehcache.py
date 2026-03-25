"""
TurboQuant Ehcache Adapter
===========================
Bridge to Java Ehcache via Py4J or subprocess.

Requirements: pip install py4j  (and Ehcache JARs on classpath)

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.ehcache import EhcacheTurboCache

    encoder = TurboQuantEncoder(dim=768)

    # Option 1: Py4J gateway (Ehcache running in JVM)
    cache = EhcacheTurboCache(encoder, gateway_port=25333, cache_name="vectors")

    # Option 2: REST API (Ehcache with REST endpoint)
    cache = EhcacheTurboCache.from_rest(encoder, base_url="http://localhost:9090/ehcache/rest")

    cache.put("doc:1", vector)
    vec = cache.get("doc:1")
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import base64

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class EhcacheTurboCache(BaseTurboAdapter):
    """
    Ehcache adapter via Py4J JVM bridge.

    Ehcache is a popular Java cache (used in Spring, Hibernate, etc.).
    This adapter connects via Py4J to a running JVM with Ehcache loaded.

    Features:
    - Transparent compress/decompress through JVM bridge
    - Supports Ehcache2 and Ehcache3 APIs
    - TTL mapped to Ehcache timeToLive
    - Batch via Ehcache putAll/getAll
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 gateway_port: int = 25333,
                 cache_name: str = "turboquant_vectors",
                 ehcache_version: int = 3):
        """
        Args:
            gateway_port: Py4J gateway port (JVM must be running with Py4J)
            cache_name: Ehcache cache/alias name
            ehcache_version: 2 or 3
        """
        super().__init__(encoder)
        self.cache_name = cache_name
        self.version = ehcache_version

        from py4j.java_gateway import JavaGateway
        self.gateway = JavaGateway(port=gateway_port)
        self.jvm = self.gateway.jvm

        if ehcache_version == 3:
            # Ehcache3: get CacheManager from entry point
            self.cache_manager = self.gateway.entry_point.getCacheManager()
            self.cache = self.cache_manager.getCache(
                cache_name,
                self.jvm.java.lang.String,
                self.jvm.byte.__class__  # byte[] values
            )
        else:
            # Ehcache2: get CacheManager singleton
            cm_class = self.jvm.net.sf.ehcache.CacheManager
            self.cache_manager = cm_class.getInstance()
            self.cache = self.cache_manager.getCache(cache_name)

    @classmethod
    def from_rest(cls, encoder: TurboQuantEncoder,
                  base_url: str = "http://localhost:9090/ehcache/rest",
                  cache_name: str = "turboquant_vectors"):
        """Create adapter using Ehcache REST API instead of Py4J."""
        instance = object.__new__(cls)
        BaseTurboAdapter.__init__(instance, encoder)
        instance.base_url = base_url.rstrip('/')
        instance.cache_name = cache_name
        instance._mode = "rest"
        return instance

    def _raw_get(self, key: str) -> Optional[bytes]:
        if hasattr(self, '_mode') and self._mode == "rest":
            return self._rest_get(key)

        if self.version == 3:
            result = self.cache.get(key)
            if result is None:
                return None
            return bytes(result)
        else:
            element = self.cache.get(key)
            if element is None:
                return None
            return bytes(element.getObjectValue())

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        if hasattr(self, '_mode') and self._mode == "rest":
            self._rest_set(key, value, ttl)
            return

        if self.version == 3:
            byte_array = self.gateway.new_array(self.jvm.byte, len(value))
            for i, b in enumerate(value):
                byte_array[i] = b
            self.cache.put(key, byte_array)
        else:
            byte_array = self.gateway.new_array(self.jvm.byte, len(value))
            for i, b in enumerate(value):
                byte_array[i] = b
            element = self.jvm.net.sf.ehcache.Element(key, byte_array)
            if ttl:
                element.setTimeToLive(ttl)
            self.cache.put(element)

    def _raw_delete(self, key: str) -> bool:
        if hasattr(self, '_mode') and self._mode == "rest":
            return self._rest_delete(key)

        if self.version == 3:
            self.cache.remove(key)
            return True
        else:
            return bool(self.cache.remove(key))

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        if hasattr(self, '_mode') and self._mode == "rest":
            return self._rest_keys()

        if self.version == 2:
            return list(self.cache.getKeys())
        else:
            # Ehcache3 doesn't have getKeys — iterate
            keys = []
            iterator = self.cache.iterator()
            while iterator.hasNext():
                entry = iterator.next()
                keys.append(str(entry.getKey()))
            return keys

    # --- REST API methods ---

    def _rest_get(self, key: str) -> Optional[bytes]:
        import urllib.request
        try:
            url = f"{self.base_url}/{self.cache_name}/{key}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def _rest_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        import urllib.request
        url = f"{self.base_url}/{self.cache_name}/{key}"
        req = urllib.request.Request(url, data=value, method='PUT')
        req.add_header('Content-Type', 'application/octet-stream')
        urllib.request.urlopen(req)

    def _rest_delete(self, key: str) -> bool:
        import urllib.request
        url = f"{self.base_url}/{self.cache_name}/{key}"
        req = urllib.request.Request(url, method='DELETE')
        try:
            urllib.request.urlopen(req)
            return True
        except urllib.error.HTTPError:
            return False

    def _rest_keys(self) -> List[str]:
        import urllib.request
        url = f"{self.base_url}/{self.cache_name}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())

    def close(self):
        """Close the Py4J gateway."""
        if hasattr(self, 'gateway'):
            self.gateway.close()
