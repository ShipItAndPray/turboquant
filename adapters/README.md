# TurboQuant Adapters

Drop-in vector compression for **24 storage systems**. Based on [Google's TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — PolarQuant + QJL for ~6x compression with near-zero accuracy loss.

## Quick Start

```python
from turboquant.core import TurboQuantEncoder, TurboQuantConfig
from turboquant.adapters.redis import RedisTurboCache

# 1. Create encoder (one-time, reuse across all adapters)
encoder = TurboQuantEncoder(dim=768, config=TurboQuantConfig(bits=4, block_size=32))

# 2. Wrap your existing client
import redis
cache = RedisTurboCache(encoder, redis.Redis(), prefix="emb:")

# 3. Use it — vectors are compressed transparently
cache.put("doc:1", embedding_vector)             # Stores ~500 bytes instead of ~3KB
vec = cache.get("doc:1")                          # Returns full float32 vector
results = cache.search(query_vector, k=10)        # Similarity search over compressed data
stats = cache.stats()                             # Compression stats
```

## All Adapters

Every adapter has the same API: `put()`, `get()`, `put_batch()`, `get_batch()`, `search()`, `delete()`, `stats()`.

### Caches
| Adapter | Backend | Install | Key Feature |
|---------|---------|---------|-------------|
| [`redis`](redis.py) | Redis | `pip install redis` | Pipeline batch ops, SCAN search, TTL |
| [`memcached`](memcached.py) | Memcached | `pip install pymemcache` | get_multi/set_multi, CAS atomic updates |
| [`ehcache`](ehcache.py) | Ehcache (Java) | `pip install py4j` | Py4J JVM bridge or REST API |
| [`hazelcast`](hazelcast.py) | Hazelcast | `pip install hazelcast-python-client` | Distributed cluster, put_all/get_all |

### Databases
| Adapter | Backend | Install | Key Feature |
|---------|---------|---------|-------------|
| [`postgresql`](postgresql.py) | PostgreSQL | `pip install psycopg2-binary` | BYTEA + pgvector hybrid search |
| [`mysql`](mysql.py) | MySQL/MariaDB | `pip install mysql-connector-python` | MEDIUMBLOB, executemany bulk |
| [`sqlite`](sqlite.py) | SQLite | *(built-in)* | Zero dependencies, WAL mode |
| [`mongodb`](mongodb.py) | MongoDB | `pip install pymongo` | BSON Binary, Atlas Vector Search |
| [`dynamodb`](dynamodb.py) | AWS DynamoDB | `pip install boto3` | Binary (B) attribute, batch_write, TTL |
| [`cassandra`](cassandra.py) | Cassandra/ScyllaDB | `pip install cassandra-driver` | Prepared stmts, BATCH, native TTL |

### Vector Databases
| Adapter | Backend | Install | Key Feature |
|---------|---------|---------|-------------|
| [`pinecone`](pinecone.py) | Pinecone | `pip install pinecone-client` | ANN + TurboQuant rerank |
| [`qdrant`](qdrant.py) | Qdrant | `pip install qdrant-client` | HNSW + rerank, payload filtering |
| [`chromadb`](chromadb.py) | ChromaDB | `pip install chromadb` | Local/server, metadata filtering |
| [`milvus`](milvus.py) | Milvus | `pip install pymilvus` | IVF/HNSW index + rerank |
| [`weaviate`](weaviate.py) | Weaviate | `pip install weaviate-client` | Schema-based, near_vector search |
| [`faiss`](faiss.py) | FAISS | `pip install faiss-cpu` | Local ANN index, save/load to disk |

### Search Engines
| Adapter | Backend | Install | Key Feature |
|---------|---------|---------|-------------|
| [`elasticsearch`](elasticsearch.py) | Elasticsearch | `pip install elasticsearch` | binary field + kNN rerank |
| [`opensearch`](opensearch.py) | OpenSearch | `pip install opensearch-py` | k-NN plugin (nmslib/faiss) |

### Object Storage
| Adapter | Backend | Install | Key Feature |
|---------|---------|---------|-------------|
| [`s3`](s3.py) | AWS S3 | `pip install boto3` | ~500B objects, concurrent upload |
| [`gcs`](gcs.py) | Google Cloud Storage | `pip install google-cloud-storage` | Blob metadata, concurrent upload |
| [`azure_blob`](azure_blob.py) | Azure Blob Storage | `pip install azure-storage-blob` | Container-based, blob metadata |

### Embedded Stores
| Adapter | Backend | Install | Key Feature |
|---------|---------|---------|-------------|
| [`lmdb`](lmdb.py) | LMDB | `pip install lmdb` | Memory-mapped, zero-copy reads |
| [`rocksdb`](rocksdb.py) | RocksDB | `pip install python-rocksdb` | WriteBatch, LSM-tree optimized |

### Streaming
| Adapter | Backend | Install | Key Feature |
|---------|---------|---------|-------------|
| [`kafka`](kafka.py) | Apache Kafka | `pip install confluent-kafka` | Producer + Consumer, 6x smaller messages |

## Compression Performance

Tested on random unit vectors (typical embedding distribution):

| Config | Dim=128 | Dim=384 | Dim=768 | Dim=1536 |
|--------|---------|---------|---------|----------|
| **4-bit, bs=32** | 5.5x / 0.990 | 6.1x / 0.973 | 6.2x / 0.949 | 6.3x / 0.907 |
| **3-bit, bs=32** | 6.6x / 0.957 | 7.5x / 0.889 | 7.7x / 0.796 | 7.9x / 0.670 |

Format: compression ratio / cosine similarity

**Recommendation:** Use **4-bit, block_size=32** for production. It gives 6x compression with >0.95 cosine similarity up to dim=768.

## Architecture

```
Your App
   |
   v
[TurboQuant Adapter]  <-- Same API for all 24 backends
   |
   ├── encode: vector -> compressed bytes (~6x smaller)
   ├── decode: compressed bytes -> vector (near-lossless)
   └── similarity: compressed vs compressed (no decode needed)
   |
   v
[Your Storage Backend]  <-- Redis, Postgres, S3, Pinecone, etc.
```

### How It Works

1. **PolarQuant** (Stage 1): Random orthogonal rotation spreads information uniformly across all components, then block-wise quantization compresses each block of 32 values with its own scale factor.

2. **QJL** (Stage 2): Quantized Johnson-Lindenstrauss transform projects the residual error into a random subspace and quantizes to 1-bit (sign only), providing unbiased error correction with near-zero overhead.

### Vector DB Reranking

For vector databases (Pinecone, Qdrant, Milvus, etc.), the adapter stores both:
- Full vector for native ANN search (fast candidate generation)
- Compressed vector in metadata/payload (for TurboQuant reranking)

This gives you the best of both worlds: fast approximate search + precise reranking.

```python
# Search modes for vector DBs:
results = cache.search(query, k=10, mode="rerank")     # ANN candidates + TQ rerank (best quality)
results = cache.search(query, k=10, mode="native")     # ANN only (fastest)
results = cache.search(query, k=10, mode="compressed")  # TQ only (no ANN index needed)
```

## Batch Operations

All adapters support batch operations optimized for their backend:

```python
# Batch put (uses Redis pipelines, DynamoDB batch_write, etc.)
cache.put_batch({
    "doc:1": vector1,
    "doc:2": vector2,
    "doc:3": vector3,
})

# Batch get
vectors = cache.get_batch(["doc:1", "doc:2", "doc:3"])
```

## Stats & Monitoring

```python
stats = cache.stats()
# {
#   "puts": 1000,
#   "gets": 5000,
#   "hits": 4800,
#   "hit_rate": "96.0%",
#   "bytes_original": 3072000,
#   "bytes_compressed": 494000,
#   "overall_ratio": "6.2x",
#   "avg_encode_ms": "1.05",
#   "avg_decode_ms": "0.58",
# }
```

## Extending

Create your own adapter by subclassing `BaseTurboAdapter`:

```python
from turboquant.adapters._base import BaseTurboAdapter

class MyCustomCache(BaseTurboAdapter):
    def _raw_get(self, key: str) -> Optional[bytes]:
        return self.backend.get(key)

    def _raw_set(self, key: str, value: bytes, ttl=None) -> None:
        self.backend.set(key, value)

    def _raw_delete(self, key: str) -> bool:
        return self.backend.delete(key)

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        return self.backend.keys()
```

You get `put()`, `get()`, `put_batch()`, `get_batch()`, `search()`, `delete()`, and `stats()` for free.
