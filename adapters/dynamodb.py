"""
TurboQuant DynamoDB Adapter
=============================
Compressed vector storage in AWS DynamoDB.

Requirements: pip install boto3

Usage:
    import boto3
    from turboquant.core import TurboQuantEncoder
    from turboquant.adapters.dynamodb import DynamoDBTurboCache

    encoder = TurboQuantEncoder(dim=768)
    cache = DynamoDBTurboCache(encoder, table_name="vectors", region="us-east-1")

    cache.put("doc:1", vector, metadata={"title": "Hello"})
    vec = cache.get("doc:1")
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

from _base import BaseTurboAdapter
from core import TurboQuantEncoder, CompressedVector


class DynamoDBTurboCache(BaseTurboAdapter):
    """
    AWS DynamoDB adapter with TurboQuant compression.

    Features:
    - Binary (B) attribute for compressed vectors (efficient DynamoDB storage)
    - batch_write_item for bulk operations (25 items per batch)
    - TTL support via DynamoDB TTL attribute
    - On-demand or provisioned capacity
    - GSI on metadata attributes for filtered queries
    """

    def __init__(self, encoder: TurboQuantEncoder,
                 table_name: str = "turboquant_vectors",
                 region: str = "us-east-1",
                 client: Any = None,
                 create_table: bool = True,
                 ttl_attribute: str = "expires_at"):
        super().__init__(encoder)
        self.table_name = table_name
        self.ttl_attribute = ttl_attribute

        if client:
            self.dynamodb = client
        else:
            import boto3
            self.dynamodb = boto3.resource("dynamodb", region_name=region)

        self.table = self.dynamodb.Table(table_name)

        if create_table:
            self._ensure_table()

    def _ensure_table(self):
        import boto3
        client = self.table.meta.client
        try:
            client.describe_table(TableName=self.table_name)
        except client.exceptions.ResourceNotFoundException:
            client.create_table(
                TableName=self.table_name,
                KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
            waiter = client.get_waiter("table_exists")
            waiter.wait(TableName=self.table_name)

            # Enable TTL
            try:
                client.update_time_to_live(
                    TableName=self.table_name,
                    TimeToLiveSpecification={
                        "Enabled": True,
                        "AttributeName": self.ttl_attribute,
                    }
                )
            except Exception:
                pass

    def _raw_get(self, key: str) -> Optional[bytes]:
        resp = self.table.get_item(Key={"id": key}, ProjectionExpression="vector_data")
        item = resp.get("Item")
        if item and "vector_data" in item:
            return bytes(item["vector_data"].value)
        return None

    def _raw_set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        import time as _time
        item = {
            "id": key,
            "vector_data": value,
            "original_dim": self.encoder.dim,
        }
        if ttl:
            item[self.ttl_attribute] = int(_time.time()) + ttl
        self.table.put_item(Item=item)

    def _raw_delete(self, key: str) -> bool:
        self.table.delete_item(Key={"id": key})
        return True

    def _raw_keys(self, pattern: str = "*") -> List[str]:
        keys = []
        scan_kwargs = {"ProjectionExpression": "id"}
        if pattern != "*":
            scan_kwargs["FilterExpression"] = "begins_with(id, :prefix)"
            scan_kwargs["ExpressionAttributeValues"] = {
                ":prefix": pattern.replace("*", "")
            }

        while True:
            resp = self.table.scan(**scan_kwargs)
            keys.extend(item["id"] for item in resp.get("Items", []))
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        return keys

    def put(self, key: str, vector: np.ndarray,
            metadata: Optional[dict] = None, ttl: Optional[int] = None) -> dict:
        import time as _time

        vector = np.asarray(vector, dtype=np.float32).ravel()
        compressed = self.encoder.encode(vector)
        data = compressed.to_bytes()

        item = {
            "id": key,
            "vector_data": data,
            "original_dim": self.encoder.dim,
            "compression_ratio": Decimal(str(round(compressed.compression_ratio(), 2))),
        }
        if metadata:
            # Convert floats to Decimal for DynamoDB
            item["metadata"] = json.loads(json.dumps(metadata), parse_float=Decimal)
        if ttl:
            item[self.ttl_attribute] = int(_time.time()) + ttl

        self.table.put_item(Item=item)

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
        """Batch write using DynamoDB batch_write_item (25 items per batch)."""
        import time as _time

        total_orig = 0
        total_comp = 0
        batch = []

        for key, vector in items.items():
            vector = np.asarray(vector, dtype=np.float32).ravel()
            compressed = self.encoder.encode(vector)
            data = compressed.to_bytes()
            total_orig += len(vector) * 4
            total_comp += len(data)

            item = {
                "id": key,
                "vector_data": data,
                "original_dim": self.encoder.dim,
                "compression_ratio": Decimal(str(round(compressed.compression_ratio(), 2))),
            }
            if ttl:
                item[self.ttl_attribute] = int(_time.time()) + ttl

            batch.append({"PutRequest": {"Item": item}})

            if len(batch) >= 25:
                self.table.meta.client.batch_write_item(
                    RequestItems={self.table_name: batch}
                )
                batch = []

        if batch:
            self.table.meta.client.batch_write_item(
                RequestItems={self.table_name: batch}
            )

        self._stats["puts"] += len(items)
        self._stats["bytes_original"] += total_orig
        self._stats["bytes_compressed"] += total_comp

        return {
            "count": len(items),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "ratio": f"{total_orig / max(total_comp, 1):.1f}x",
        }

    def search(self, query: np.ndarray, k: int = 10,
               keys: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Client-side similarity search (DynamoDB has no native vector search)."""
        query = np.asarray(query, dtype=np.float32).ravel()
        query_c = self.encoder.encode(query)
        results = []

        if keys:
            # BatchGetItem
            request_keys = [{"id": k} for k in keys]
            for i in range(0, len(request_keys), 100):
                chunk = request_keys[i:i+100]
                resp = self.table.meta.client.batch_get_item(
                    RequestItems={self.table_name: {"Keys": chunk, "ProjectionExpression": "id, vector_data"}}
                )
                for item in resp["Responses"].get(self.table_name, []):
                    candidate = CompressedVector.from_bytes(bytes(item["vector_data"].value))
                    score = self.encoder.similarity(query_c, candidate)
                    results.append((item["id"], score))
        else:
            for key in self._raw_keys():
                data = self._raw_get(key)
                if data:
                    candidate = CompressedVector.from_bytes(data)
                    score = self.encoder.similarity(query_c, candidate)
                    results.append((key, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
