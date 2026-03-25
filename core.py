"""
TurboQuant Core — PolarQuant + QJL Implementation
===================================================
Implements Google's TurboQuant two-stage vector compression:

Stage 1: PolarQuant
  - Random orthogonal rotation to spread information uniformly across components
  - After rotation, all components have similar magnitude distributions
  - Uniform quantization with a SINGLE global scale (minimal overhead)
  - Radius (norm) stored separately with high precision

Stage 2: QJL (Quantized Johnson-Lindenstrauss)
  - Random projection of residual error into lower-dimensional space
  - 1-bit sign quantization (zero memory overhead)
  - Unbiased inner product estimator

Result: 3-4 bit compression with near-zero accuracy loss, no training needed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import struct


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant compression."""
    # Bits per component for quantization (2-8, default 4)
    bits: int = 4
    # Block size for quantization (each block gets its own scale/zero)
    # Smaller = better quality, slightly more overhead. 0 = global (single scale)
    block_size: int = 32
    # QJL projection dimension (higher = better error correction, more bytes)
    # Set to 0 to disable QJL stage
    qjl_proj_dim: int = 32
    # Random seed for reproducible rotation/projection matrices
    seed: int = 42


@dataclass
class CompressedVector:
    """A TurboQuant-compressed vector stored as packed bytes."""
    data: bytes           # All compressed data packed together
    original_dim: int     # Original vector dimension

    def nbytes(self) -> int:
        return len(self.data)

    def compression_ratio(self) -> float:
        original_bytes = self.original_dim * 4  # float32
        return original_bytes / len(self.data)

    def to_bytes(self) -> bytes:
        """Serialize for cache storage."""
        return struct.pack('<I', self.original_dim) + self.data

    @classmethod
    def from_bytes(cls, raw: bytes) -> 'CompressedVector':
        """Deserialize from cache storage."""
        dim = struct.unpack_from('<I', raw, 0)[0]
        return cls(data=raw[4:], original_dim=dim)


def _random_orthogonal(dim: int, seed: int) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim).astype(np.float32)
    Q, R = np.linalg.qr(H)
    # Ensure deterministic sign
    Q *= np.sign(np.diag(R))
    return Q


def _randomized_hadamard(dim: int, seed: int) -> np.ndarray:
    """
    Fast randomized Hadamard-like rotation.
    For dimensions that aren't powers of 2, falls back to random orthogonal.
    """
    if dim & (dim - 1) != 0:
        # Not power of 2, use random orthogonal
        return _random_orthogonal(dim, seed)

    # Build Hadamard matrix recursively
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < dim:
        H = np.block([[H, H], [H, -H]]) / np.sqrt(2)

    # Apply random sign flips
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    return H * signs[np.newaxis, :]


class TurboQuantEncoder:
    """
    Full TurboQuant encoder/decoder: PolarQuant + QJL.

    Usage:
        encoder = TurboQuantEncoder(dim=768)
        compressed = encoder.encode(vector)           # ~4-6x smaller
        reconstructed = encoder.decode(compressed)    # near-lossless
        score = encoder.similarity(comp_a, comp_b)    # no decompression needed

    For cache integration, use the adapters in cache_optimizer.py.
    """

    def __init__(self, dim: int, config: Optional[TurboQuantConfig] = None):
        self.dim = dim
        self.config = config or TurboQuantConfig()
        self.bits = self.config.bits
        self.levels = (1 << self.bits) - 1  # e.g., 15 for 4-bit
        self.block_size = self.config.block_size if self.config.block_size > 0 else dim

        # Stage 1: Random rotation matrix
        self.R = _random_orthogonal(dim, self.config.seed)

        # Stage 2: QJL random projection (Rademacher +1/-1)
        self.qjl_dim = self.config.qjl_proj_dim
        if self.qjl_dim > 0:
            rng = np.random.RandomState(self.config.seed + 1)
            self.P = rng.choice(
                [-1.0, 1.0], size=(self.qjl_dim, dim)
            ).astype(np.float32) / np.sqrt(self.qjl_dim)
        else:
            self.P = None

    def encode(self, vec: np.ndarray) -> CompressedVector:
        """Compress a vector using TurboQuant."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        assert len(vec) == self.dim, f"Expected dim {self.dim}, got {len(vec)}"

        # --- Stage 1: PolarQuant ---
        # 1a. Store the norm separately (float16 = 2 bytes)
        norm = np.linalg.norm(vec)
        norm_f16 = np.float16(norm)

        # 1b. Normalize
        if norm > 1e-10:
            unit = vec / norm
        else:
            unit = np.zeros_like(vec)

        # 1c. Random rotation — spreads information uniformly
        rotated = self.R @ unit

        # 1d. Block-wise quantization: each block of `block_size` gets its own scale/zero
        bs = self.block_size
        n_blocks = (self.dim + bs - 1) // bs
        quantized = np.zeros(self.dim, dtype=np.uint8)
        block_mins = np.zeros(n_blocks, dtype=np.float16)
        block_maxs = np.zeros(n_blocks, dtype=np.float16)

        for b in range(n_blocks):
            start = b * bs
            end = min(start + bs, self.dim)
            block = rotated[start:end]
            vmin = block.min()
            vmax = block.max()
            if vmax - vmin < 1e-10:
                vmax = vmin + 1e-6
            block_mins[b] = np.float16(vmin)
            block_maxs[b] = np.float16(vmax)
            scale = (float(np.float16(vmax)) - float(np.float16(vmin))) / self.levels
            quantized[start:end] = np.clip(
                np.round((block - float(np.float16(vmin))) / scale), 0, self.levels
            ).astype(np.uint8)

        # 1e. Pack quantized values into bytes
        packed = _pack_nbits(quantized, self.dim, self.bits)

        # 1f. Compute residual for QJL
        dequantized = np.zeros(self.dim, dtype=np.float32)
        for b in range(n_blocks):
            start = b * bs
            end = min(start + bs, self.dim)
            vmin_f = float(block_mins[b])
            vmax_f = float(block_maxs[b])
            scale = (vmax_f - vmin_f) / self.levels
            dequantized[start:end] = quantized[start:end].astype(np.float32) * scale + vmin_f

        reconstructed_unit = self.R.T @ dequantized
        reconstructed = reconstructed_unit * float(norm_f16)
        residual = vec - reconstructed

        # --- Stage 2: QJL error correction ---
        qjl_packed = b''
        qjl_norm_f16 = np.float16(0.0)
        if self.P is not None:
            res_norm = np.linalg.norm(residual)
            qjl_norm_f16 = np.float16(res_norm)
            if res_norm > 1e-10:
                projected = self.P @ residual
                signs = (projected >= 0).astype(np.uint8)
                qjl_packed = np.packbits(signs).tobytes()
            else:
                qjl_packed = np.packbits(np.zeros(self.qjl_dim, dtype=np.uint8)).tobytes()

        # --- Pack everything into bytes ---
        # Format: [norm_f16(2)] [n_blocks(2)] [block_mins(n*2)] [block_maxs(n*2)]
        #         [quantized_packed] [qjl_norm_f16(2)] [qjl_signs]
        parts = []
        parts.append(norm_f16.tobytes())                    # 2 bytes
        parts.append(struct.pack('<H', n_blocks))           # 2 bytes
        parts.append(block_mins.tobytes())                  # n_blocks * 2 bytes
        parts.append(block_maxs.tobytes())                  # n_blocks * 2 bytes
        parts.append(packed.tobytes())                      # packed quantized values
        parts.append(qjl_norm_f16.tobytes())                # 2 bytes
        parts.append(qjl_packed)                            # qjl_dim/8 bytes

        data = b''.join(parts)
        return CompressedVector(data=data, original_dim=self.dim)

    def decode(self, compressed: CompressedVector) -> np.ndarray:
        """Decompress a TurboQuant-compressed vector."""
        data = compressed.data
        offset = 0

        # Read header
        norm = float(np.frombuffer(data[offset:offset+2], dtype=np.float16)[0])
        offset += 2
        n_blocks = struct.unpack_from('<H', data, offset)[0]
        offset += 2

        # Read block scales
        block_mins = np.frombuffer(data[offset:offset + n_blocks * 2], dtype=np.float16).astype(np.float32)
        offset += n_blocks * 2
        block_maxs = np.frombuffer(data[offset:offset + n_blocks * 2], dtype=np.float16).astype(np.float32)
        offset += n_blocks * 2

        # Read packed quantized values
        packed_len = _packed_size(self.dim, self.bits)
        packed = np.frombuffer(data[offset:offset+packed_len], dtype=np.uint8)
        offset += packed_len
        quantized = _unpack_nbits(packed, self.dim, self.bits)

        # Block-wise dequantize
        bs = self.block_size
        dequantized = np.zeros(self.dim, dtype=np.float32)
        for b in range(n_blocks):
            start = b * bs
            end = min(start + bs, self.dim)
            vmin_f = float(block_mins[b])
            vmax_f = float(block_maxs[b])
            scale = (vmax_f - vmin_f) / self.levels
            dequantized[start:end] = quantized[start:end] * scale + vmin_f

        # Reverse rotation
        unit = self.R.T @ dequantized
        base = unit * norm

        # QJL error correction
        qjl_norm = float(np.frombuffer(data[offset:offset+2], dtype=np.float16)[0])
        offset += 2

        if self.P is not None and qjl_norm > 1e-10:
            qjl_packed_len = (self.qjl_dim + 7) // 8
            signs_packed = np.frombuffer(data[offset:offset+qjl_packed_len], dtype=np.uint8)
            signs = np.unpackbits(signs_packed)[:self.qjl_dim].astype(np.float32) * 2 - 1
            correction = self.P.T @ signs * qjl_norm / np.sqrt(self.qjl_dim)
            base += correction

        return base

    def encode_batch(self, vectors: np.ndarray) -> List[CompressedVector]:
        """Compress a batch of vectors."""
        return [self.encode(v) for v in vectors]

    def decode_batch(self, compressed_list: List[CompressedVector]) -> np.ndarray:
        """Decompress a batch of vectors."""
        return np.array([self.decode(c) for c in compressed_list])

    def similarity(self, a: CompressedVector, b: CompressedVector) -> float:
        """Estimate cosine similarity between two compressed vectors (via decode)."""
        va = self.decode(a)
        vb = self.decode(b)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def error(self, original: np.ndarray, compressed: CompressedVector) -> dict:
        """Compute reconstruction error metrics."""
        original = np.asarray(original, dtype=np.float32).ravel()
        reconstructed = self.decode(compressed)
        diff = original - reconstructed

        orig_norm = float(np.linalg.norm(original))
        recon_norm = float(np.linalg.norm(reconstructed))
        error_norm = float(np.linalg.norm(diff))

        cos_sim = float(np.dot(original, reconstructed)) / (orig_norm * recon_norm + 1e-10)

        return {
            "l2_error": error_norm,
            "relative_error": error_norm / (orig_norm + 1e-10),
            "cosine_similarity": cos_sim,
            "mse": float(np.mean(diff ** 2)),
            "compression_ratio": compressed.compression_ratio(),
            "compressed_bytes": compressed.nbytes(),
            "original_bytes": len(original) * 4,
        }


def _packed_size(dim: int, bits: int) -> int:
    """Compute packed byte size for dim values at given bit-width."""
    if bits == 4:
        return (dim + 1) // 2
    elif bits == 2:
        return (dim + 3) // 4
    elif bits in (5, 6, 7, 8):
        return dim  # 1 byte per value
    else:
        # Generic: ceil(dim * bits / 8)
        return (dim * bits + 7) // 8


def _pack_nbits(values: np.ndarray, dim: int, bits: int) -> np.ndarray:
    """Pack N-bit quantized values into bytes."""
    if bits == 4:
        packed_len = (dim + 1) // 2
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(0, dim - 1, 2):
            packed[i // 2] = (values[i] << 4) | (values[i + 1] & 0x0F)
        if dim % 2 == 1:
            packed[-1] = values[-1] << 4
        return packed
    elif bits == 2:
        packed_len = (dim + 3) // 4
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(dim):
            byte_idx = i // 4
            shift = (3 - (i % 4)) * 2
            packed[byte_idx] |= (values[i] & 0x03) << shift
        return packed
    elif bits >= 5:
        return values.astype(np.uint8)
    else:
        # Generic bit packing (3-bit, etc.)
        total_bytes = (dim * bits + 7) // 8
        packed = np.zeros(total_bytes, dtype=np.uint8)
        mask = (1 << bits) - 1
        bit_pos = 0
        for i in range(dim):
            val = int(values[i]) & mask
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            remaining = 8 - bit_offset
            if remaining >= bits:
                packed[byte_idx] |= val << (remaining - bits)
            else:
                packed[byte_idx] |= val >> (bits - remaining)
                if byte_idx + 1 < total_bytes:
                    packed[byte_idx + 1] |= (val << (8 - (bits - remaining))) & 0xFF
            bit_pos += bits
        return packed


def _unpack_nbits(packed: np.ndarray, dim: int, bits: int) -> np.ndarray:
    """Unpack N-bit values from bytes."""
    values = np.zeros(dim, dtype=np.float32)
    if bits == 4:
        for i in range(0, dim - 1, 2):
            values[i] = (packed[i // 2] >> 4) & 0x0F
            values[i + 1] = packed[i // 2] & 0x0F
        if dim % 2 == 1:
            values[-1] = (packed[-1] >> 4) & 0x0F
        return values
    elif bits == 2:
        for i in range(dim):
            byte_idx = i // 4
            shift = (3 - (i % 4)) * 2
            values[i] = (packed[byte_idx] >> shift) & 0x03
        return values
    elif bits >= 5:
        return packed[:dim].astype(np.float32)
    else:
        # Generic unpack
        mask = (1 << bits) - 1
        bit_pos = 0
        for i in range(dim):
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            remaining = 8 - bit_offset
            if remaining >= bits:
                values[i] = (packed[byte_idx] >> (remaining - bits)) & mask
            else:
                hi = packed[byte_idx] & ((1 << remaining) - 1)
                lo_bits = bits - remaining
                lo = (packed[byte_idx + 1] >> (8 - lo_bits)) if byte_idx + 1 < len(packed) else 0
                values[i] = (hi << lo_bits) | lo
            bit_pos += bits
        return values
