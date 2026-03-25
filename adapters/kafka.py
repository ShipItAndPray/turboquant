"""
TurboQuant Kafka Adapter
==========================
Compressed vector streaming via Kafka.

Requirements: pip install confluent-kafka

Usage:
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.kafka import KafkaTurboProducer, KafkaTurboConsumer

    encoder = TurboQuantEncoder(dim=768)

    # Producer: send compressed vectors
    producer = KafkaTurboProducer(encoder, bootstrap_servers="localhost:9092")
    producer.send("embeddings", key="doc:1", vector=vector)

    # Consumer: receive and decompress
    consumer = KafkaTurboConsumer(encoder, bootstrap_servers="localhost:9092",
                                  topic="embeddings", group_id="my-group")
    for key, vector, metadata in consumer.consume(max_messages=100):
        process(key, vector)
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Iterator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core import TurboQuantEncoder, CompressedVector


class KafkaTurboProducer:
    """
    Kafka producer that sends TurboQuant-compressed vectors.

    Reduces Kafka message sizes by ~6x, lowering broker storage
    and network bandwidth.
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 bootstrap_servers: str = "localhost:9092",
                 producer: Any = None,
                 **kafka_config):
        self.encoder = encoder

        if producer:
            self.producer = producer
        else:
            from confluent_kafka import Producer
            config = {"bootstrap.servers": bootstrap_servers, **kafka_config}
            self.producer = Producer(config)

        self._stats = {"sent": 0, "bytes_original": 0, "bytes_compressed": 0}

    def send(self, topic: str, key: str, vector: np.ndarray,
             metadata: Optional[dict] = None,
             partition: Optional[int] = None,
             callback: Optional[callable] = None) -> dict:
        """Send a compressed vector to Kafka topic."""
        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        # Prepend metadata length + metadata JSON if present
        if metadata:
            meta_bytes = json.dumps(metadata).encode()
            payload = len(meta_bytes).to_bytes(4, 'big') + meta_bytes + data
        else:
            payload = (0).to_bytes(4, 'big') + data

        kwargs = {"topic": topic, "key": key.encode(), "value": payload}
        if partition is not None:
            kwargs["partition"] = partition
        if callback:
            kwargs["callback"] = callback

        self.producer.produce(**kwargs)

        original_bytes = len(vector) * 4
        self._stats["sent"] += 1
        self._stats["bytes_original"] += original_bytes
        self._stats["bytes_compressed"] += len(payload)

        return {
            "key": key,
            "original_bytes": original_bytes,
            "message_bytes": len(payload),
            "ratio": f"{original_bytes / len(payload):.1f}x",
        }

    def send_batch(self, topic: str, items: Dict[str, np.ndarray],
                   metadata: Optional[Dict[str, dict]] = None) -> dict:
        total_orig = 0
        total_comp = 0
        for key, vector in items.items():
            meta = (metadata or {}).get(key)
            info = self.send(topic, key, vector, metadata=meta)
            total_orig += info["original_bytes"]
            total_comp += info["message_bytes"]

        self.producer.flush()
        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def flush(self):
        self.producer.flush()

    def stats(self) -> dict:
        return dict(self._stats)


class KafkaTurboConsumer:
    """
    Kafka consumer that receives and decompresses TurboQuant vectors.
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "embeddings",
                 group_id: str = "turboquant-consumer",
                 consumer: Any = None,
                 **kafka_config):
        self.encoder = encoder
        self.topic = topic

        if consumer:
            self.consumer = consumer
        else:
            from confluent_kafka import Consumer
            config = {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                **kafka_config,
            }
            self.consumer = Consumer(config)
            self.consumer.subscribe([topic])

    def consume(self, max_messages: int = 100,
                timeout: float = 1.0) -> Iterator[Tuple[str, np.ndarray, Optional[dict]]]:
        """Yield (key, decompressed_vector, metadata) tuples."""
        count = 0
        while count < max_messages:
            msg = self.consumer.poll(timeout)
            if msg is None:
                break
            if msg.error():
                continue

            key = msg.key().decode() if msg.key() else None
            payload = msg.value()

            # Parse metadata
            meta_len = int.from_bytes(payload[:4], 'big')
            metadata = None
            if meta_len > 0:
                metadata = json.loads(payload[4:4 + meta_len].decode())

            # Decompress vector
            compressed_data = payload[4 + meta_len:]
            compressed = CompressedVector.from_bytes(compressed_data)
            vector = self.encoder.decode(compressed)

            yield key, vector, metadata
            count += 1

    def close(self):
        self.consumer.close()
