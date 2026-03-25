"""
TurboQuant Adapters — Drop-in compression for every major cache, database, and storage system.

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.redis import RedisTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = RedisTurboCache(encoder, redis_client)
    cache.put("doc:1", vector)
"""

ADAPTERS = [
    "redis",
    "memcached",
    "ehcache",
    "hazelcast",
    "elasticsearch",
    "opensearch",
    "postgresql",
    "mysql",
    "sqlite",
    "mongodb",
    "dynamodb",
    "cassandra",
    "s3",
    "gcs",
    "azure_blob",
    "pinecone",
    "qdrant",
    "chromadb",
    "milvus",
    "weaviate",
    "faiss",
    "kafka",
    "lmdb",
    "rocksdb",
]
