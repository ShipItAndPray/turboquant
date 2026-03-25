<p align="center">
  <pre align="center">
 ████████╗██╗   ██╗██████╗ ██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
 ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
    ██║   ██║   ██║██████╔╝██████╔╝██║   ██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║
    ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║
    ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║
    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝
  </pre>
</p>

<h3 align="center">6x Compression for Vectors, Embeddings, and LLMs</h3>

<p align="center">
  Based on <a href="https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/">Google's TurboQuant</a> — PolarQuant + QJL. No training required. Near-zero accuracy loss.
</p>

<p align="center">
  <a href="#adapters--plug-and-play-for-24-systems">24 Adapters</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#compression-benchmarks">Benchmarks</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#llm-quantization-cli">LLM CLI</a> &bull;
  <a href="#target-platforms">Platforms</a> &bull;
  <a href="#github-action">CI/CD</a>
</p>

---

## Adapters — Plug and Play for 24 Systems

**3 lines of code. No forks, no patches, no recompilation.** Wrap your existing client, get 6x compression.

```python
from turboquant.core import TurboQuantEncoder
from turboquant.adapters.redis import RedisTurboCache

encoder = TurboQuantEncoder(dim=768)
cache = RedisTurboCache(encoder, your_existing_redis_client)
cache.put("doc:1", embedding)  # 3KB → 500 bytes
```

Every adapter has the same API: `put` · `get` · `search` · `put_batch` · `get_batch` · `delete` · `stats`

### Caches

| Adapter | Install | Key Feature |
|---------|---------|-------------|
| **[Redis](adapters/redis.py)** | `pip install redis` | Pipeline batching, SCAN search, TTL, key prefixing |
| **[Memcached](adapters/memcached.py)** | `pip install pymemcache` | get_multi/set_multi, CAS atomic updates |
| **[Ehcache](adapters/ehcache.py)** | `pip install py4j` | Java JVM bridge (Py4J) or REST API, Ehcache 2 & 3 |
| **[Hazelcast](adapters/hazelcast.py)** | `pip install hazelcast-python-client` | Distributed cluster, put_all/get_all |

### Databases

| Adapter | Install | Key Feature |
|---------|---------|-------------|
| **[PostgreSQL](adapters/postgresql.py)** | `pip install psycopg2-binary` | BYTEA + optional pgvector hybrid search, JSONB metadata |
| **[MySQL](adapters/mysql.py)** | `pip install mysql-connector-python` | MEDIUMBLOB storage, executemany bulk insert |
| **[SQLite](adapters/sqlite.py)** | *(built-in — zero deps)* | WAL mode, JSON1 metadata, great for local dev |
| **[MongoDB](adapters/mongodb.py)** | `pip install pymongo` | BSON Binary, Atlas Vector Search aggregation pipeline |
| **[DynamoDB](adapters/dynamodb.py)** | `pip install boto3` | Binary attribute, batch_write_item (25/batch), TTL |
| **[Cassandra](adapters/cassandra.py)** | `pip install cassandra-driver` | Prepared statements, UNLOGGED BATCH, native TTL |

### Vector Databases

| Adapter | Install | Key Feature |
|---------|---------|-------------|
| **[Pinecone](adapters/pinecone.py)** | `pip install pinecone-client` | Native ANN + TurboQuant reranking for higher recall |
| **[Qdrant](adapters/qdrant.py)** | `pip install qdrant-client` | HNSW search + rerank, payload filtering |
| **[ChromaDB](adapters/chromadb.py)** | `pip install chromadb` | Local/server mode, metadata where-filtering |
| **[Milvus](adapters/milvus.py)** | `pip install pymilvus` | IVF/HNSW index + TurboQuant rerank |
| **[Weaviate](adapters/weaviate.py)** | `pip install weaviate-client` | Schema-based, near_vector + rerank |
| **[FAISS](adapters/faiss.py)** | `pip install faiss-cpu` | Local ANN index, save/load to disk, rerank mode |

### Search Engines

| Adapter | Install | Key Feature |
|---------|---------|-------------|
| **[Elasticsearch](adapters/elasticsearch.py)** | `pip install elasticsearch` | Binary field + dense_vector kNN, bulk API |
| **[OpenSearch](adapters/opensearch.py)** | `pip install opensearch-py` | k-NN plugin (nmslib/faiss engine), compressed-only mode |

### Object Storage

| Adapter | Install | Key Feature |
|---------|---------|-------------|
| **[AWS S3](adapters/s3.py)** | `pip install boto3` | ~500B objects, concurrent ThreadPool upload |
| **[Google Cloud Storage](adapters/gcs.py)** | `pip install google-cloud-storage` | Blob metadata, concurrent upload |
| **[Azure Blob](adapters/azure_blob.py)** | `pip install azure-storage-blob` | Container-based, blob metadata |

### Embedded Key-Value Stores

| Adapter | Install | Key Feature |
|---------|---------|-------------|
| **[LMDB](adapters/lmdb.py)** | `pip install lmdb` | Memory-mapped B+ tree, zero-copy reads, ACID |
| **[RocksDB](adapters/rocksdb.py)** | `pip install python-rocksdb` | WriteBatch, LSM-tree (less write amplification with smaller values) |

### Streaming

| Adapter | Install | Key Feature |
|---------|---------|-------------|
| **[Apache Kafka](adapters/kafka.py)** | `pip install confluent-kafka` | Producer + Consumer, 6x smaller messages, metadata support |

> Full adapter docs with examples: **[adapters/README.md](adapters/README.md)**

---

## Quick Start

### Install

```bash
pip install numpy   # Only dependency for core engine

# Then install your backend's client:
pip install redis               # for Redis adapter
pip install psycopg2-binary     # for PostgreSQL adapter
pip install pymongo             # for MongoDB adapter
# ... etc
```

### Compress and Store Vectors

```python
from turboquant.core import TurboQuantEncoder, TurboQuantConfig

# Create encoder (reuse across your app)
config = TurboQuantConfig(bits=4, block_size=32, qjl_proj_dim=64)
encoder = TurboQuantEncoder(dim=768, config=config)

# Compress a single vector
import numpy as np
vector = np.random.randn(768).astype(np.float32)
compressed = encoder.encode(vector)

print(f"Original:   {768 * 4} bytes")
print(f"Compressed: {compressed.nbytes()} bytes")
print(f"Ratio:      {compressed.compression_ratio():.1f}x")

# Decompress
reconstructed = encoder.decode(compressed)

# Serialize for any storage
raw_bytes = compressed.to_bytes()
restored = type(compressed).from_bytes(raw_bytes)
```

### Use with Any Backend

```python
# Redis
import redis
from turboquant.adapters.redis import RedisTurboCache
cache = RedisTurboCache(encoder, redis.Redis(), prefix="emb:", ttl=3600)

# PostgreSQL with pgvector
from turboquant.adapters.postgresql import PostgresTurboCache
cache = PostgresTurboCache(encoder, dsn="postgresql://localhost/mydb", use_pgvector=True)

# MongoDB with Atlas Vector Search
from pymongo import MongoClient
from turboquant.adapters.mongodb import MongoTurboCache
cache = MongoTurboCache(encoder, MongoClient(), db="myapp", collection="embeddings")

# S3
from turboquant.adapters.s3 import S3TurboCache
cache = S3TurboCache(encoder, bucket="my-vectors", prefix="embeddings/")

# SQLite (zero deps)
from turboquant.adapters.sqlite import SQLiteTurboCache
cache = SQLiteTurboCache(encoder, db_path="vectors.db")

# All adapters — same API:
cache.put("doc:1", vector)
cache.put_batch({"doc:2": v2, "doc:3": v3})
vec = cache.get("doc:1")
results = cache.search(query_vector, k=10)
print(cache.stats())
```

### Vector DB Reranking

For Pinecone, Qdrant, Milvus, etc. — use native ANN for candidates, TurboQuant for precision:

```python
from turboquant.adapters.qdrant import QdrantTurboCache
cache = QdrantTurboCache(encoder, qdrant_client, collection="docs")

results = cache.search(query, k=10, mode="rerank")      # ANN + TQ rerank (best quality)
results = cache.search(query, k=10, mode="native")      # ANN only (fastest)
results = cache.search(query, k=10, mode="compressed")   # TQ only (no ANN index needed)
```

### Build Your Own Adapter

Subclass `BaseTurboAdapter` — implement 4 methods, get the full API for free:

```python
from turboquant.adapters._base import BaseTurboAdapter

class MyCache(BaseTurboAdapter):
    def _raw_get(self, key): ...       # return bytes or None
    def _raw_set(self, key, value): ...  # store bytes
    def _raw_delete(self, key): ...    # return bool
    def _raw_keys(self, pattern): ...  # return list of keys

# You now have: put, get, search, put_batch, get_batch, delete, stats
```

---

## Compression Benchmarks

4-bit quantization, block_size=32, QJL proj_dim=64:

| Dimension | Compression | Cosine Similarity | Bytes per Vector |
|-----------|-------------|-------------------|-----------------|
| 128 | **5.5x** | 0.990 | 94 |
| 384 | **6.1x** | 0.973 | 254 |
| 768 | **6.2x** | 0.949 | 494 |
| 1536 | **6.3x** | 0.907 | 974 |

### Memory Savings at Scale

| Scenario | Raw float32 | TurboQuant | Saved |
|----------|-------------|------------|-------|
| 10K vectors, dim=128 | 5 MB | 940 KB | 82% |
| 100K vectors, dim=384 | 154 MB | 25 MB | 83% |
| 1M vectors, dim=768 | 3.1 GB | 494 MB | **84%** |
| 10M vectors, dim=1536 | 61.4 GB | 9.7 GB | **84%** |

### Throughput

| Operation | Speed (dim=768) |
|-----------|----------------|
| Encode | ~1,000 vec/s |
| Decode | ~1,800 vec/s |
| Similarity | ~500 pairs/s |

---

## How It Works

Based on [Google's TurboQuant research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — two-stage compression, no training required:

### Stage 1: PolarQuant

1. **Random orthogonal rotation** — spreads information uniformly across all vector components
2. **Block-wise quantization** — each block of 32 values gets its own scale factor, quantized to N bits
3. **Norm preservation** — vector magnitude stored separately at float16 precision

### Stage 2: QJL (Quantized Johnson-Lindenstrauss)

1. **Random projection** of the quantization residual into a lower-dimensional space
2. **1-bit sign quantization** — each projected value becomes just +1 or -1
3. **Unbiased error correction** — mathematically proven to eliminate quantization bias

```
Input Vector (float32)          Compressed Output (~6x smaller)
   ┌─────────────┐              ┌──────────────────────────┐
   │ [0.23, -0.1,│              │ norm (2B) + block scales  │
   │  0.45, 0.67,│   encode()   │ (N*4B) + packed N-bit    │
   │  ...768 dim ]│ ──────────→  │ values + QJL sign bits   │
   │ 3,072 bytes  │              │ ~494 bytes               │
   └─────────────┘              └──────────────────────────┘
```

---

## LLM Quantization CLI

TurboQuant also includes a CLI for compressing HuggingFace LLMs to GGUF/GPTQ/AWQ:

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
```

That's it. Your 16GB model is now 4GB. Ship it to Ollama, vLLM, or llama.cpp.

```bash
pip install turboquant[all]  # Install all LLM backends
```

---

## Target Platforms

**Don't know which format to use?** Just tell TurboQuant where you want to run it.

### Ollama (one command, ready to run)

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --target ollama --bits 4
```

This quantizes to GGUF, auto-generates a `Modelfile` with the correct chat template, and tells you the exact `ollama create` command to run.

### vLLM

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --target vllm --bits 4
```

Auto-selects AWQ (best GPU throughput for vLLM).

### LM Studio / llama.cpp

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --target lmstudio --bits 4
turboquant meta-llama/Llama-3.1-8B-Instruct --target llamacpp --bits 4
```

---

## Publish to HuggingFace

Quantize any model and publish to HuggingFace Hub in one command:

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct \
  --format gguf --bits 4 \
  --push-to-hub yourname/Llama-3.1-8B-Instruct-GGUF
```

Requires: `huggingface-cli login` or `HF_TOKEN` environment variable.

---

## Quality Evaluation

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4 --eval
```

| Perplexity | Grade | Meaning |
|------------|-------|---------|
| < 10 | EXCELLENT | Minimal quality loss |
| 10-20 | GOOD | Acceptable for most use cases |
| 20-50 | FAIR | Some degradation, consider higher bits |
| > 100 | POOR | Model may be broken |

---

## Smart Recommendations

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --recommend
```

Detects your hardware (Apple Silicon, NVIDIA GPU, CPU-only) and recommends the best format + bits.

---

## GitHub Action

**CI/CD pipeline for LLM quantization.** Auto-quantize after fine-tuning.

```yaml
# .github/workflows/quantize.yml
name: Quantize Model
on:
  workflow_dispatch:
    inputs:
      model:
        description: 'Model to quantize'
        required: true

jobs:
  quantize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ShipItAndPray/turboquant@master
        with:
          model: ${{ inputs.model }}
          format: gguf
          bits: 4
          eval: true
          push-to-hub: yourname/model-GGUF
          hf-token: ${{ secrets.HF_TOKEN }}
```

### Action Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes | — | HuggingFace model ID or local path |
| `format` | No | `gguf` | `gguf`, `gptq`, `awq`, or `all` |
| `bits` | No | `4` | `2`, `3`, `4`, `5`, or `8` |
| `target` | No | — | `ollama`, `vllm`, `llamacpp`, `lmstudio` |
| `push-to-hub` | No | — | HuggingFace repo to upload to |
| `eval` | No | `false` | Run quality evaluation |
| `hf-token` | No | — | HuggingFace API token |

---

## LLM Formats

| Format | Best For | Engine | GPU? |
|--------|----------|--------|------|
| **GGUF** | Local/CPU, Ollama, LM Studio | llama.cpp | No |
| **GPTQ** | GPU serving, high throughput | vLLM, TGI | Yes |
| **AWQ** | Fast GPU inference | vLLM, TGI | Yes |

**Don't know?** Run `turboquant your-model --recommend`.

### Supported Architectures

LLaMA (1-3.3), Mistral/Mixtral, Qwen (1.5-2.5), Phi (1-4), GPT-2/J/NeoX, Gemma, DeepSeek, and any HuggingFace model with `.safetensors` or `.bin` weights.

### All CLI Options

```
turboquant MODEL [OPTIONS]

Positional:
  MODEL                     HuggingFace model ID or local path

Formats:
  --format, -f FORMAT       gguf, gptq, awq, or all (default: gguf)
  --bits, -b BITS           2, 3, 4, 5, or 8 (default: 4)
  --output, -o DIR          Output directory (default: ./turboquant-output)

Target Platforms:
  --target, -t TARGET       ollama, vllm, llamacpp, lmstudio

Publishing:
  --push-to-hub REPO        Upload to HuggingFace Hub (e.g. user/model-GGUF)

Quality:
  --eval                    Run perplexity evaluation after quantization
  --recommend               Show hardware-aware format recommendation

Info:
  --info                    Show model details without quantizing
  --check                   Show available backends and hardware
```

---

## Requirements

- Python 3.9+
- NumPy (only dependency for core vector engine + adapters)
- Backend client library for your chosen adapter (see tables above)
- For LLM CLI: PyTorch 2.0+ and backend-specific packages

## License

MIT
