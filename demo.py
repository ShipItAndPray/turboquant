#!/usr/bin/env python3
"""
TurboQuant Demo — Benchmark compression on real-world vector workloads
======================================================================
Run: python3 demo.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import TurboQuantEncoder, TurboQuantConfig
from cache_optimizer import GenericTurboCache, DatabaseTurboCache


def banner(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def test_compression_quality():
    banner("COMPRESSION QUALITY TEST")

    print(f"  {'Config':<28} {'Ratio':>7} {'CosSim':>8} {'RelErr':>8} {'Bytes':>6}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")

    for bits in [3, 4]:
        for dim in [128, 384, 768]:
            config = TurboQuantConfig(bits=bits, block_size=32, qjl_proj_dim=64)
            encoder = TurboQuantEncoder(dim, config)

            rng = np.random.RandomState(42)
            vectors = rng.randn(100, dim).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

            cosines, ratios, errors = [], [], []
            sample_bytes = 0
            for vec in vectors:
                compressed = encoder.encode(vec)
                m = encoder.error(vec, compressed)
                cosines.append(m["cosine_similarity"])
                ratios.append(m["compression_ratio"])
                errors.append(m["relative_error"])
                sample_bytes = m["compressed_bytes"]

            label = f"{bits}-bit dim={dim} bs=32"
            print(f"  {label:<28} {np.mean(ratios):>6.2f}x {np.mean(cosines):>8.4f} {np.mean(errors):>8.4f} {sample_bytes:>6}")


def test_search_accuracy():
    banner("SIMILARITY SEARCH ACCURACY (Recall@K)")

    dim = 128
    n_db = 1000
    n_queries = 50

    config = TurboQuantConfig(bits=4, block_size=32, qjl_proj_dim=64)
    encoder = TurboQuantEncoder(dim, config)

    rng = np.random.RandomState(42)
    database = rng.randn(n_db, dim).astype(np.float32)
    database /= np.linalg.norm(database, axis=1, keepdims=True)
    queries = rng.randn(n_queries, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    gt_scores = queries @ database.T
    compressed_db = [encoder.encode(v) for v in database]

    for k in [1, 5, 10, 20]:
        recalls = []
        for qi in range(n_queries):
            gt_topk = set(np.argsort(gt_scores[qi])[-k:][::-1])
            query_c = encoder.encode(queries[qi])
            scores = [encoder.similarity(query_c, c) for c in compressed_db]
            comp_topk = set(np.argsort(scores)[-k:][::-1])
            recalls.append(len(gt_topk & comp_topk) / k)

        print(f"  Recall@{k:2d}: {np.mean(recalls):.3f} (dim={dim}, db={n_db}, queries={n_queries})")


def test_throughput():
    banner("THROUGHPUT BENCHMARK")

    dim = 768
    config = TurboQuantConfig(bits=4, block_size=32, qjl_proj_dim=64)
    encoder = TurboQuantEncoder(dim, config)

    rng = np.random.RandomState(42)
    vectors = rng.randn(200, dim).astype(np.float32)

    start = time.time()
    compressed = [encoder.encode(v) for v in vectors]
    encode_time = time.time() - start

    start = time.time()
    decoded = [encoder.decode(c) for c in compressed]
    decode_time = time.time() - start

    print(f"  Dimension:       {dim}")
    print(f"  Vectors:         {len(vectors)}")
    print(f"  Encode:          {len(vectors)/encode_time:.0f} vec/s ({encode_time*1000/len(vectors):.1f} ms/vec)")
    print(f"  Decode:          {len(vectors)/decode_time:.0f} vec/s ({decode_time*1000/len(vectors):.1f} ms/vec)")
    print(f"  Avg compression: {np.mean([c.compression_ratio() for c in compressed]):.1f}x")


def test_cache_adapters():
    banner("CACHE ADAPTER DEMO")

    dim = 256
    encoder = TurboQuantEncoder(dim, TurboQuantConfig(bits=4, block_size=32, qjl_proj_dim=64))
    rng = np.random.RandomState(42)

    # --- In-Memory Cache ---
    print("  [GenericTurboCache - In-Memory]")
    cache = GenericTurboCache(encoder)

    vectors = {}
    for i in range(100):
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vectors[f"doc:{i}"] = vec

    result = cache.put_batch(vectors)
    print(f"    Stored {result['count']} vectors: {result['original_bytes']} -> {result['compressed_bytes']} bytes ({result['ratio']})")

    retrieved = cache.get("doc:0")
    original = vectors["doc:0"]
    cos_sim = float(np.dot(original, retrieved) / (np.linalg.norm(original) * np.linalg.norm(retrieved)))
    print(f"    Retrieval cosine similarity: {cos_sim:.4f}")

    query = rng.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    results = cache.search(query, list(vectors.keys()), k=5)
    print(f"    Search top-5: {[(k, f'{s:.3f}') for k, s in results]}")
    print(f"    Stats: {cache.stats()}")

    # --- Serialization round-trip ---
    print("\n  [Serialization Round-Trip]")
    c = encoder.encode(vectors["doc:0"])
    raw = c.to_bytes()
    c2 = type(c).from_bytes(raw)
    v_restored = encoder.decode(c2)
    cos_rt = float(np.dot(original, v_restored) / (np.linalg.norm(original) * np.linalg.norm(v_restored)))
    print(f"    Serialized size: {len(raw)} bytes, round-trip cosine: {cos_rt:.4f}")

    # --- SQLite Database Cache ---
    print("\n  [DatabaseTurboCache - SQLite]")
    db_path = "/tmp/turboquant_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db_cache = DatabaseTurboCache(encoder, db_url=f"sqlite:///{db_path}")
    for key, vec in list(vectors.items())[:20]:
        db_cache.put(key, vec, metadata={"source": "demo"})

    retrieved_db = db_cache.get("doc:0")
    cos_db = float(np.dot(original, retrieved_db) / (np.linalg.norm(original) * np.linalg.norm(retrieved_db)))
    print(f"    Stored 20 vectors, retrieval cosine: {cos_db:.4f}")
    print(f"    Stats: {db_cache.stats()}")

    results_db = db_cache.search(query, k=5)
    print(f"    Search top-5: {[(k, f'{s:.3f}') for k, s in results_db]}")

    db_cache.close()
    os.remove(db_path)


def test_memory_comparison():
    banner("MEMORY SAVINGS PROJECTION")

    scenarios = [
        ("10K vecs, dim=128", 10_000, 128),
        ("100K vecs, dim=384", 100_000, 384),
        ("1M vecs, dim=768", 1_000_000, 768),
        ("10M vecs, dim=1536", 10_000_000, 1536),
    ]

    print(f"  {'Scenario':<28} {'Raw float32':<14} {'TurboQuant':<14} {'Saved':<8}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*8}")

    for name, n, dim in scenarios:
        raw = n * dim * 4
        config = TurboQuantConfig(bits=4, block_size=32, qjl_proj_dim=64)
        enc = TurboQuantEncoder(dim, config)
        sample = np.random.randn(dim).astype(np.float32)
        ratio = enc.encode(sample).compression_ratio()
        compressed = int(raw / ratio)

        def fmt(b):
            if b >= 1e9: return f"{b/1e9:.1f} GB"
            if b >= 1e6: return f"{b/1e6:.0f} MB"
            return f"{b/1e3:.0f} KB"

        print(f"  {name:<28} {fmt(raw):<14} {fmt(compressed):<14} {(1 - compressed/raw)*100:.0f}%")


if __name__ == "__main__":
    print("""
 ████████╗██╗   ██╗██████╗ ██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
 ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
    ██║   ██║   ██║██████╔╝██████╔╝██║   ██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║
    ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║
    ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║
    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝

    Cache Optimizer Benchmark — PolarQuant + QJL
    """)

    test_compression_quality()
    test_memory_comparison()
    test_throughput()
    test_search_accuracy()
    test_cache_adapters()

    banner("ALL BENCHMARKS COMPLETE")
