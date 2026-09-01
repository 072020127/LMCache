# SPDX-License-Identifier: Apache-2.0

"""Training-free KV Quantization Drift Meter (QDM) shadow implementation.

The observer is deliberately downstream of MaKV quantization: it consumes the
payload and scales returned by the production quantizer and uses the existing
reference dequantization semantics to form a small block-level witness.  This
module is for explicit diagnostics and offline validation only; it is not a
dependency of precision planning, production attention, or cache restore.
The reference estimator never materializes attention weights or a shadow KV
cache.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping
import math

# Third Party
import torch
import torch.nn.functional as F

# First Party
from lmcache.v1.storage_backend.makv.reference_dequant import (
    dequantize_bucket_vectors,
)

QDM_VERSION = "qdm_v1"
QDM_LAYOUT = "layer_block_kv_head"
QDM_QUANTIZER_VERSION = "makv_per_token_head_symmetric_narrow_v1"
QDM_BLOCK_SIZE = 32

PRECISION_ID_K2V2 = 0
PRECISION_ID_K4V2 = 1
PRECISION_ID_K8V4 = 2
PRECISION_ID_BF16 = 3
PRECISION_ID_MIXED = 255

_PAIR_TO_PRECISION_ID = {
    (2, 2): PRECISION_ID_K2V2,
    (4, 2): PRECISION_ID_K4V2,
    (8, 4): PRECISION_ID_K8V4,
    (16, 16): PRECISION_ID_BF16,
}
_PRECISION_ID_TO_NAME = {
    PRECISION_ID_K2V2: "K2V2",
    PRECISION_ID_K4V2: "K4V2",
    PRECISION_ID_K8V4: "K8V4",
    PRECISION_ID_BF16: "BF16",
    PRECISION_ID_MIXED: "MIXED",
}


def _plan_value(plan: Any, name: str) -> Any:
    if isinstance(plan, Mapping):
        return plan[name]
    return getattr(plan, name)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def _tensor_from_payload(payload: Any, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload.to(dtype=dtype)
    if isinstance(payload, memoryview):
        raw = payload.tobytes()
    else:
        raw = bytes(payload)
    if not raw:
        return torch.empty(0, dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype)


def _precision_ids_from_plan(
    plan: Any,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return block precision IDs, valid token counts and block origin."""
    chunk_start = int(_plan_value(plan, "chunk_start"))
    chunk_length = int(_plan_value(plan, "chunk_length"))
    num_layers = int(_plan_value(plan, "num_layers"))
    num_heads = int(_plan_value(plan, "num_kv_heads"))
    bucket_bits = tuple(int(value) for value in _plan_value(plan, "bucket_bits"))
    bucket_ids = torch.tensor(
        list(_plan_value(plan, "bucket_ids")), dtype=torch.long
    )
    layout = str(_plan_value(plan, "importance_layout"))
    if block_size <= 0:
        raise ValueError("QDM block_size must be positive")
    if chunk_length <= 0:
        raise ValueError("QDM requires a non-empty KV chunk")

    block_start = chunk_start // block_size
    block_end = (chunk_start + chunk_length + block_size - 1) // block_size
    num_blocks = block_end - block_start
    valid_tokens = torch.zeros(num_blocks, dtype=torch.int32)
    token_block = torch.empty(chunk_length, dtype=torch.long)
    for token in range(chunk_length):
        block = (chunk_start + token) // block_size - block_start
        token_block[token] = block
        valid_tokens[block] += 1

    if layout == "token":
        if bucket_ids.numel() != chunk_length:
            raise ValueError("QDM token plan length does not match chunk length")
        bits = torch.tensor(
            [bucket_bits[int(value)] for value in bucket_ids.tolist()],
            dtype=torch.long,
        )
        pairs = torch.stack((bits, bits), dim=1)
        pairs_by_layer = pairs.unsqueeze(0).expand(num_layers, -1, -1)
    elif layout == "layer_kv_token":
        expected = num_layers * 2 * chunk_length
        if bucket_ids.numel() != expected:
            raise ValueError("QDM layer/KV plan length does not match chunk length")
        bucket_ids = bucket_ids.view(num_layers, 2, chunk_length)
        bits = torch.empty_like(bucket_ids)
        for bucket_index, bit in enumerate(bucket_bits):
            bits[bucket_ids == bucket_index] = bit
        pairs_by_layer = bits.permute(0, 2, 1).contiguous()
    else:
        raise ValueError(f"unsupported QDM plan layout: {layout!r}")

    precision = torch.full(
        (num_layers, num_blocks), PRECISION_ID_MIXED, dtype=torch.uint8
    )
    for layer in range(num_layers):
        for block in range(num_blocks):
            token_mask = token_block == block
            pairs = {
                tuple(pair)
                for pair in pairs_by_layer[layer][token_mask].tolist()
            }
            if len(pairs) == 1:
                precision[layer, block] = _PAIR_TO_PRECISION_ID.get(
                    next(iter(pairs)), PRECISION_ID_MIXED
                )
    return (
        precision.unsqueeze(-1).expand(-1, -1, num_heads).contiguous(),
        valid_tokens,
        block_start,
        chunk_start,
    )


@dataclass(frozen=True)
class QDMMetadata:
    """Persistent QDM witness with ``[layer, block, kv_head]`` layout."""

    qdm_version: str
    quantizer_version: str
    block_size: int
    k_error: torch.Tensor
    v_error: torch.Tensor
    v_norm: torch.Tensor
    precision_id: torch.Tensor
    token_start: int = 0
    token_count: int = 0
    block_start: int = 0
    valid_tokens: torch.Tensor | None = None
    layout: str = QDM_LAYOUT

    def __post_init__(self) -> None:
        if self.layout != QDM_LAYOUT:
            raise ValueError(f"unsupported QDM metadata layout: {self.layout!r}")
        if self.block_size <= 0:
            raise ValueError("QDM block_size must be positive")
        shapes = {
            tuple(self.k_error.shape),
            tuple(self.v_error.shape),
            tuple(self.v_norm.shape),
            tuple(self.precision_id.shape),
        }
        if len(shapes) != 1 or len(next(iter(shapes))) != 3:
            raise ValueError(
                "QDM witness tensors must share [layer, block, kv_head] shape"
            )
        if self.valid_tokens is not None:
            if (
                self.valid_tokens.ndim != 1
                or self.valid_tokens.shape[0] != self.k_error.shape[1]
            ):
                raise ValueError("QDM valid_tokens must have one entry per block")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.k_error.shape)

    @property
    def num_layers(self) -> int:
        return self.shape[0]

    @property
    def num_blocks(self) -> int:
        return self.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.shape[2]

    def precision_name(self, precision_id: int) -> str:
        return _PRECISION_ID_TO_NAME.get(int(precision_id), "UNKNOWN")

    def to_descriptor(self) -> dict[str, Any]:
        """Return JSON-safe metadata; numeric witnesses live in payloads."""
        return {
            "qdm_version": self.qdm_version,
            "quantizer_version": self.quantizer_version,
            "block_size": int(self.block_size),
            "layout": self.layout,
            "shape": list(self.shape),
            "token_start": int(self.token_start),
            "token_count": int(self.token_count),
            "block_start": int(self.block_start),
            "k_error": "qdm_k_error",
            "v_error": "qdm_v_error",
            "v_norm": "qdm_v_norm",
            "precision_id": "qdm_precision_id",
            "valid_tokens": "qdm_valid_tokens",
            "precision_table": dict(_PRECISION_ID_TO_NAME),
        }

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        """Return a human-readable representation for diagnostics and tests."""
        result = self.to_descriptor()
        if include_values:
            result.update(
                {
                    "k_error_values": self.k_error.detach().cpu().tolist(),
                    "v_error_values": self.v_error.detach().cpu().tolist(),
                    "v_norm_values": self.v_norm.detach().cpu().tolist(),
                    "precision_id_values": self.precision_id.detach().cpu().tolist(),
                    "valid_tokens_values": (
                        None
                        if self.valid_tokens is None
                        else self.valid_tokens.detach().cpu().tolist()
                    ),
                }
            )
        return result

    def to_payloads(self) -> dict[str, bytes]:
        valid_tokens = self.valid_tokens
        if valid_tokens is None:
            valid_tokens = torch.zeros(self.num_blocks, dtype=torch.int32)
        return {
            "qdm_k_error": _tensor_bytes(self.k_error.to(torch.float32)),
            "qdm_v_error": _tensor_bytes(self.v_error.to(torch.float32)),
            "qdm_v_norm": _tensor_bytes(self.v_norm.to(torch.float32)),
            "qdm_precision_id": _tensor_bytes(self.precision_id.to(torch.uint8)),
            "qdm_valid_tokens": _tensor_bytes(valid_tokens.to(torch.int32)),
        }

    @classmethod
    def from_descriptor(
        cls,
        descriptor: Mapping[str, Any],
        payloads: Mapping[str, Any],
    ) -> "QDMMetadata":
        shape = tuple(int(value) for value in descriptor["shape"])

        def load(name: str, dtype: torch.dtype) -> torch.Tensor:
            reference = descriptor[name]
            if isinstance(reference, str):
                if reference not in payloads:
                    raise ValueError(f"QDM payload is missing: {reference}")
                tensor = _tensor_from_payload(payloads[reference], dtype)
            else:
                tensor = torch.as_tensor(reference, dtype=dtype)
            expected = math.prod(shape) if name != "valid_tokens" else shape[1]
            if tensor.numel() != expected:
                raise ValueError(f"QDM payload {name!r} has an invalid size")
            return tensor.reshape(shape if name != "valid_tokens" else (shape[1],))

        return cls(
            qdm_version=str(descriptor["qdm_version"]),
            quantizer_version=str(descriptor["quantizer_version"]),
            block_size=int(descriptor["block_size"]),
            k_error=load("k_error", torch.float32),
            v_error=load("v_error", torch.float32),
            v_norm=load("v_norm", torch.float32),
            precision_id=load("precision_id", torch.uint8),
            token_start=int(descriptor.get("token_start", 0)),
            token_count=int(descriptor.get("token_count", 0)),
            block_start=int(descriptor.get("block_start", 0)),
            valid_tokens=load("valid_tokens", torch.int32),
            layout=str(descriptor.get("layout", QDM_LAYOUT)),
        )


class QDMObserver:
    """Reduce real production quantizer round trips to a block witness."""

    def __init__(
        self,
        plan: Any,
        *,
        block_size: int = QDM_BLOCK_SIZE,
        quantizer_version: str = QDM_QUANTIZER_VERSION,
    ) -> None:
        self.plan = plan
        self.block_size = int(block_size)
        self.quantizer_version = str(quantizer_version)
        if self.block_size <= 0:
            raise ValueError("QDM block_size must be positive")
        self.layout = str(_plan_value(plan, "importance_layout"))
        self.chunk_start = int(_plan_value(plan, "chunk_start"))
        self.chunk_length = int(_plan_value(plan, "chunk_length"))
        self.num_layers = int(_plan_value(plan, "num_layers"))
        self.num_kv_heads = int(_plan_value(plan, "num_kv_heads"))
        self.head_dim = int(_plan_value(plan, "head_dim"))
        (
            self._precision_id_cpu,
            self._valid_tokens_cpu,
            self.block_start,
            self.token_start,
        ) = _precision_ids_from_plan(plan, self.block_size)
        self._device: torch.device | None = None
        self._k_error: torch.Tensor | None = None
        self._v_error: torch.Tensor | None = None
        self._v_norm: torch.Tensor | None = None

    @property
    def num_blocks(self) -> int:
        return int(self._precision_id_cpu.shape[1])

    def _ensure_buffers(self, device: torch.device) -> None:
        if self._device is not None:
            if self._device != device:
                raise ValueError(
                    "QDM observer cannot combine tensors on different devices"
                )
            return
        shape = (self.num_layers, self.num_blocks, self.num_kv_heads)
        self._device = device
        self._k_error = torch.zeros(shape, dtype=torch.float32, device=device)
        self._v_error = torch.zeros_like(self._k_error)
        self._v_norm = torch.zeros_like(self._k_error)

    @staticmethod
    def _scatter_max(
        target: torch.Tensor,
        values: torch.Tensor,
        block_indices: torch.Tensor,
    ) -> None:
        if values.numel() == 0:
            return
        index = block_indices.view(1, -1, 1).expand(
            values.shape[0], -1, values.shape[-1]
        )
        reduced = torch.zeros_like(target)
        reduced.scatter_reduce_(1, index, values, reduce="amax", include_self=True)
        target.copy_(torch.maximum(target, reduced))

    @staticmethod
    def _scatter_layer_block_max(
        target: torch.Tensor,
        values: torch.Tensor,
        layer_indices: torch.Tensor,
        block_indices: torch.Tensor,
    ) -> None:
        if values.numel() == 0:
            return
        num_blocks = target.shape[1]
        flat_target = target.view(target.shape[0] * num_blocks, target.shape[2])
        flat_indices = layer_indices * num_blocks + block_indices
        index = flat_indices.view(-1, 1).expand(-1, values.shape[-1])
        reduced = torch.zeros_like(flat_target)
        reduced.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        flat_target.copy_(torch.maximum(flat_target, reduced))

    def observe_bucket(
        self,
        *,
        vectors: torch.Tensor,
        q_payload: torch.Tensor,
        scales: torch.Tensor,
        bits: int,
        positions: torch.Tensor,
    ) -> None:
        """Observe one exact production quantizer bucket.

        ``q_payload`` and ``scales`` are the tensors returned by the existing
        quantizer.  No scale or quantization operation is created here.
        """
        if vectors.shape[-1] != self.head_dim:
            raise ValueError("QDM vector head dimension does not match the plan")
        if int(bits) not in tuple(
            int(value) for value in _plan_value(self.plan, "bucket_bits")
        ):
            raise ValueError(f"QDM received an unknown precision bucket: {bits}")
        positions = torch.as_tensor(
            positions, dtype=torch.long, device=vectors.device
        ).reshape(-1)
        self._ensure_buffers(vectors.device)
        assert self._k_error is not None
        assert self._v_error is not None
        assert self._v_norm is not None
        restored = dequantize_bucket_vectors(
            q_payload,
            scales,
            int(bits),
            vector_shape=tuple(vectors.shape),
            head_dim=self.head_dim,
            output_dtype=vectors.dtype,
        )
        if restored.shape != vectors.shape:
            raise ValueError("QDM dequantized bucket shape mismatch")
        residual = torch.sqrt(
            (vectors.float() - restored.float()).square().sum(dim=-1)
        )
        if self.layout == "token":
            if vectors.ndim != 5 or vectors.shape[1] != 2:
                raise ValueError("token-layout QDM vectors must be [L,2,T,H,D]")
            if positions.numel() != vectors.shape[2]:
                raise ValueError("QDM bucket position count mismatch")
            global_positions = positions + self.chunk_start
            block_indices = global_positions // self.block_size - self.block_start
            self._scatter_max(self._k_error, residual[:, 0], block_indices)
            self._scatter_max(self._v_error, residual[:, 1], block_indices)
            value_norm = torch.sqrt(
                vectors[:, 1].float().square().sum(dim=-1)
            )
            self._scatter_max(self._v_norm, value_norm, block_indices)
            return

        if self.layout != "layer_kv_token":
            raise ValueError(f"unsupported QDM plan layout: {self.layout!r}")
        if vectors.ndim != 3 or positions.numel() != vectors.shape[0]:
            raise ValueError("layer/KV-layout QDM vectors must be [T,H,D]")
        layer_indices = positions // (2 * self.chunk_length)
        remainder = positions % (2 * self.chunk_length)
        kv_indices = remainder // self.chunk_length
        token_indices = remainder % self.chunk_length
        block_indices = (
            token_indices + self.chunk_start
        ) // self.block_size - self.block_start
        value_norm = torch.sqrt(vectors.float().square().sum(dim=-1))
        for kv_index, target in ((0, self._k_error), (1, self._v_error)):
            mask = kv_indices == kv_index
            self._scatter_layer_block_max(
                target,
                residual[mask],
                layer_indices[mask],
                block_indices[mask],
            )
            if kv_index == 1:
                self._scatter_layer_block_max(
                    self._v_norm,
                    value_norm[mask],
                    layer_indices[mask],
                    block_indices[mask],
                )

    # A descriptive alias makes the hook easy to use from packers that call
    # their payload output "quantized" rather than "q_payload".
    observe_quantized_bucket = observe_bucket

    def observe_block(
        self,
        *,
        layer: int,
        token_positions: torch.Tensor,
        original_k: torch.Tensor,
        original_v: torch.Tensor,
        dequantized_k: torch.Tensor,
        dequantized_v: torch.Tensor,
        quantized_k: torch.Tensor | None = None,
        quantized_v: torch.Tensor | None = None,
    ) -> None:
        """Observe already unpacked K/V tensors for a reference/test hook."""
        del quantized_k, quantized_v
        if (
            original_k.shape != original_v.shape
            or original_k.shape != dequantized_k.shape
        ):
            raise ValueError("QDM K/V witness tensors must have matching shapes")
        if original_k.shape != dequantized_v.shape:
            raise ValueError("QDM dequantized V shape mismatch")
        if original_k.ndim != 3:
            raise ValueError("QDM block tensors must have shape [tokens, heads, dim]")
        if not 0 <= layer < self.num_layers:
            raise ValueError("QDM layer index is out of range")
        positions = torch.as_tensor(
            token_positions, dtype=torch.long, device=original_k.device
        )
        self._ensure_buffers(original_k.device)
        assert self._k_error is not None
        assert self._v_error is not None
        assert self._v_norm is not None
        block_indices = (
            (positions + self.chunk_start) // self.block_size - self.block_start
        )
        k_norm = torch.sqrt(
            (original_k.float() - dequantized_k.float()).square().sum(dim=-1)
        )
        v_norm_error = torch.sqrt(
            (original_v.float() - dequantized_v.float()).square().sum(dim=-1)
        )
        value_norm = torch.sqrt(original_v.float().square().sum(dim=-1))
        target_k = self._k_error[layer : layer + 1]
        target_v = self._v_error[layer : layer + 1]
        target_norm = self._v_norm[layer : layer + 1]
        self._scatter_max(target_k, k_norm.unsqueeze(0), block_indices)
        self._scatter_max(target_v, v_norm_error.unsqueeze(0), block_indices)
        self._scatter_max(target_norm, value_norm.unsqueeze(0), block_indices)

    def finalize(self) -> QDMMetadata:
        if self._device is None:
            self._ensure_buffers(torch.device("cpu"))
        assert self._k_error is not None
        assert self._v_error is not None
        assert self._v_norm is not None
        return QDMMetadata(
            qdm_version=QDM_VERSION,
            quantizer_version=self.quantizer_version,
            block_size=self.block_size,
            k_error=self._k_error.detach().clone(),
            v_error=self._v_error.detach().clone(),
            v_norm=self._v_norm.detach().clone(),
            precision_id=self._precision_id_cpu.to(self._k_error.device),
            token_start=self.token_start,
            token_count=self.chunk_length,
            block_start=self.block_start,
            valid_tokens=self._valid_tokens_cpu.to(self._k_error.device),
        )


def load_qdm_metadata(
    metadata: Mapping[str, Any],
    payloads: Mapping[str, Any],
) -> QDMMetadata | None:
    """Load QDM witness from a decoded MaKV object, if present."""
    descriptor = metadata.get("qdm")
    if descriptor is None and "qdm_version" in metadata:
        descriptor = metadata
    if descriptor is None:
        return None
    if not isinstance(descriptor, Mapping):
        raise ValueError("MaKV QDM metadata must be an object")
    return QDMMetadata.from_descriptor(descriptor, payloads)


@dataclass(frozen=True)
class QDMDriftEstimate:
    """Reference attention drift estimate for one layer/KV head."""

    k_tv_bound: float
    v_error: float
    attention_error_bound: float
    v_norm_max: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "k_tv_bound": self.k_tv_bound,
            "v_error": self.v_error,
            "attention_error_bound": self.attention_error_bound,
        }


class QDMScalarAccumulator:
    """Decode-side scalar accumulator suitable for a fused attention hook."""

    def __init__(self, query: torch.Tensor, *, head_dim: int | None = None) -> None:
        if query.ndim != 1:
            raise ValueError("QDM scalar accumulator expects one query head")
        self.query_norm = torch.linalg.vector_norm(query.float())
        self.head_dim = int(head_dim or query.shape[-1])
        if self.head_dim <= 0:
            raise ValueError("QDM head_dim must be positive")
        self._a = torch.zeros((), dtype=torch.float32, device=query.device)
        self._v_error = torch.zeros_like(self._a)
        self._v_norm_max = torch.zeros_like(self._a)

    def update_block(
        self,
        attention_prob: torch.Tensor | float,
        k_error: torch.Tensor | float,
        v_error: torch.Tensor | float,
        v_norm: torch.Tensor | float,
    ) -> None:
        probability = torch.as_tensor(
            attention_prob, dtype=torch.float32, device=self._a.device
        )
        if probability.numel() != 1:
            raise ValueError("QDM block probability must be scalar")
        if bool((probability < 0).item()):
            raise ValueError("QDM attention probabilities must be non-negative")
        k_value = torch.as_tensor(k_error, dtype=torch.float32, device=self._a.device)
        v_value = torch.as_tensor(v_error, dtype=torch.float32, device=self._a.device)
        norm_value = torch.as_tensor(v_norm, dtype=torch.float32, device=self._a.device)
        c_block = self.query_norm * k_value / math.sqrt(self.head_dim)
        self._a = self._a + probability * torch.exp(c_block)
        self._v_error = self._v_error + probability * v_value
        if bool((probability > 0).item()):
            self._v_norm_max = torch.maximum(self._v_norm_max, norm_value)

    def finalize(
        self,
        *,
        visible_v_norm_max: torch.Tensor | float | None = None,
    ) -> QDMDriftEstimate:
        if not bool(torch.isfinite(self._a).item()):
            tv = torch.ones_like(self._a)
        else:
            tv = torch.clamp((self._a.square() - 1.0) / 2.0, min=0.0, max=1.0)
        if visible_v_norm_max is not None:
            visible_norm = torch.as_tensor(
                visible_v_norm_max,
                dtype=torch.float32,
                device=self._v_norm_max.device,
            )
            if visible_norm.numel() != 1:
                raise ValueError("visible_v_norm_max must be scalar")
            self._v_norm_max = torch.maximum(self._v_norm_max, visible_norm)
        attention_error = 2.0 * tv * self._v_norm_max + self._v_error
        return QDMDriftEstimate(
            k_tv_bound=float(tv.item()),
            v_error=float(self._v_error.item()),
            attention_error_bound=float(attention_error.item()),
            v_norm_max=float(self._v_norm_max.item()),
        )


QDMDecodeAccumulator = QDMScalarAccumulator


class QDMRuntimeEstimator:
    """Estimate drift from witness metadata and per-block probabilities."""

    def __init__(self, metadata: QDMMetadata) -> None:
        self.metadata = metadata

    def new_accumulator(
        self,
        query: torch.Tensor,
        *,
        layer: int,
        kv_head: int,
    ) -> QDMScalarAccumulator:
        del layer, kv_head
        return QDMScalarAccumulator(query, head_dim=query.shape[-1])

    def estimate(
        self,
        query: torch.Tensor,
        attention_prob: torch.Tensor,
        *,
        layer: int,
        kv_head: int,
        visible_v_norm_max: torch.Tensor | float | None = None,
    ) -> QDMDriftEstimate:
        if query.ndim != 1:
            raise ValueError("QDM estimate expects one query head")
        if attention_prob.ndim != 1:
            raise ValueError("QDM attention probabilities must be [block]")
        if not 0 <= layer < self.metadata.num_layers:
            raise ValueError("QDM layer index is out of range")
        if not 0 <= kv_head < self.metadata.num_kv_heads:
            raise ValueError("QDM KV head index is out of range")
        if attention_prob.shape[0] != self.metadata.num_blocks:
            raise ValueError("QDM probability count does not match witness blocks")
        device = query.device
        k_error = self.metadata.k_error[layer, :, kv_head].to(device=device)
        v_error = self.metadata.v_error[layer, :, kv_head].to(device=device)
        v_norm = self.metadata.v_norm[layer, :, kv_head].to(device=device)
        accumulator = QDMScalarAccumulator(query, head_dim=query.shape[-1])
        for block in range(self.metadata.num_blocks):
            accumulator.update_block(
                attention_prob[block], k_error[block], v_error[block], v_norm[block]
            )
        return accumulator.finalize(visible_v_norm_max=visible_v_norm_max)

    def estimate_all(
        self,
        query: torch.Tensor,
        attention_prob: torch.Tensor,
        *,
        layer: int,
        kv_head_for_query: torch.Tensor | None = None,
    ) -> list[tuple[int, QDMDriftEstimate]]:
        """Estimate one record per query head without storing attention data."""
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2:
            raise ValueError("QDM query must be [head, head_dim]")
        if attention_prob.ndim == 1:
            attention_prob = attention_prob.unsqueeze(0).expand(query.shape[0], -1)
        if attention_prob.shape != (query.shape[0], self.metadata.num_blocks):
            raise ValueError("QDM probability shape does not match query heads")
        if kv_head_for_query is None:
            if query.shape[0] != self.metadata.num_kv_heads:
                raise ValueError("query/KV head count mismatch; provide a head map")
            kv_head_for_query = torch.arange(query.shape[0], device=query.device)
        if kv_head_for_query.numel() != query.shape[0]:
            raise ValueError("QDM query-to-KV head map length mismatch")
        result = []
        for query_head in range(query.shape[0]):
            kv_head = int(kv_head_for_query[query_head].item())
            result.append(
                (
                    kv_head,
                    self.estimate(
                        query[query_head],
                        attention_prob[query_head],
                        layer=layer,
                        kv_head=kv_head,
                    ),
                )
            )
        return result


def estimate_qdm_drift(
    query: torch.Tensor,
    attention_prob: torch.Tensor,
    metadata: QDMMetadata,
    *,
    layer: int,
    kv_head: int,
) -> QDMDriftEstimate:
    return QDMRuntimeEstimator(metadata).estimate(
        query, attention_prob, layer=layer, kv_head=kv_head
    )


def compute_qdm_witness(
    original_k: torch.Tensor,
    original_v: torch.Tensor,
    dequantized_k: torch.Tensor,
    dequantized_v: torch.Tensor,
    *,
    token_start: int = 0,
    block_size: int = QDM_BLOCK_SIZE,
    quantized_k: torch.Tensor | None = None,
    quantized_v: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute one-layer witness from an already materialized reference block.

    This helper is for Phase 1 reference experiments. The production hook
    should use :class:`QDMObserver.observe_bucket`, where the quantized input
    is the actual packed payload returned by MaKV.
    """
    del quantized_k, quantized_v
    if (
        original_k.shape != original_v.shape
        or original_k.shape != dequantized_k.shape
        or original_k.shape != dequantized_v.shape
    ):
        raise ValueError("QDM witness tensors must have matching shapes")
    if original_k.ndim != 3:
        raise ValueError("QDM witness tensors must have shape [tokens, heads, dim]")
    if token_start < 0 or block_size <= 0:
        raise ValueError("QDM token_start and block_size must be valid")
    tokens, heads, _ = original_k.shape
    block_start = token_start // block_size
    block_end = (token_start + tokens + block_size - 1) // block_size
    blocks = block_end - block_start
    k_error = torch.zeros(
        (1, blocks, heads), dtype=torch.float32, device=original_k.device
    )
    v_error = torch.zeros_like(k_error)
    v_norm = torch.zeros_like(k_error)
    k_residual = torch.sqrt(
        (original_k.float() - dequantized_k.float()).square().sum(dim=-1)
    )
    v_residual = torch.sqrt(
        (original_v.float() - dequantized_v.float()).square().sum(dim=-1)
    )
    value_norm = torch.sqrt(original_v.float().square().sum(dim=-1))
    for block in range(blocks):
        global_start = max(token_start, (block_start + block) * block_size)
        global_end = min(token_start + tokens, global_start + block_size)
        start = global_start - token_start
        end = global_end - token_start
        k_error[:, block] = k_residual[start:end].amax(dim=0)
        v_error[:, block] = v_residual[start:end].amax(dim=0)
        v_norm[:, block] = value_norm[start:end].amax(dim=0)
    return {"k_error": k_error, "v_error": v_error, "v_norm": v_norm}


@dataclass(frozen=True)
class QDMLogitMetrics:
    top1_top2_margin: float
    topk_entropy: float

    @property
    def top1_margin(self) -> float:
        return self.top1_top2_margin

    def to_dict(self) -> dict[str, float]:
        return {
            "top1_margin": self.top1_top2_margin,
            "top1_top2_margin": self.top1_top2_margin,
            "topK_entropy": self.topk_entropy,
        }


def compute_logit_metrics(
    logits: torch.Tensor,
    *,
    top_k: int = 50,
) -> QDMLogitMetrics | list[QDMLogitMetrics]:
    """Compute top-1/top-2 margin and entropy over the normalized top-K."""
    if logits.ndim == 2:
        return [compute_logit_metrics(row, top_k=top_k) for row in logits]
    if logits.ndim != 1 or logits.shape[0] < 2:
        raise ValueError("logits must be [vocab] with at least two entries")
    k = min(int(top_k), int(logits.shape[0]))
    if k <= 0:
        raise ValueError("top_k must be positive")
    values = torch.topk(logits.float(), k=k, dim=-1).values
    margin = values[0] - values[1]
    log_prob = values - torch.logsumexp(values, dim=-1)
    probability = log_prob.exp()
    entropy = -(probability * log_prob).sum()
    return QDMLogitMetrics(float(margin.item()), float(entropy.item()))


def top1_top2_margin(logits: torch.Tensor) -> torch.Tensor:
    """Return the top-1/top-2 margin for one or more logit rows."""
    if logits.ndim == 1:
        return torch.topk(logits.float(), k=2, dim=-1).values.diff().neg().squeeze(-1)
    if logits.ndim == 2:
        return torch.topk(logits.float(), k=2, dim=-1).values[:, 0].sub(
            torch.topk(logits.float(), k=2, dim=-1).values[:, 1]
        )
    raise ValueError("logits must be [vocab] or [tokens, vocab]")


class QDMRiskState(str, Enum):
    """Diagnostic label only; never a production controller decision."""

    SAFE = "SAFE"
    MODEL_FRAGILE = "MODEL_FRAGILE"
    KV_DRIFT_ROBUST = "KV_DRIFT_ROBUST"
    KV_TOKEN_RISK = "KV_TOKEN_RISK"


@dataclass(frozen=True)
class QDMRiskThresholds:
    """Reference-only thresholds for diagnostic labeling."""

    drift_high_threshold: float = 1.0
    margin_small_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.drift_high_threshold < 0 or self.margin_small_threshold < 0:
            raise ValueError("QDM risk thresholds must be non-negative")


def classify_risk_state(
    attention_error_bound: float,
    top1_margin: float,
    *,
    thresholds: QDMRiskThresholds | None = None,
) -> QDMRiskState:
    """Classify a reference observation for diagnostics, not control flow."""
    thresholds = thresholds or QDMRiskThresholds()
    drift_high = (
        not math.isfinite(float(attention_error_bound))
        or float(attention_error_bound) >= thresholds.drift_high_threshold
    )
    margin_small = (
        not math.isfinite(float(top1_margin))
        or float(top1_margin) <= thresholds.margin_small_threshold
    )
    if drift_high and margin_small:
        return QDMRiskState.KV_TOKEN_RISK
    if drift_high:
        return QDMRiskState.KV_DRIFT_ROBUST
    if margin_small:
        return QDMRiskState.MODEL_FRAGILE
    return QDMRiskState.SAFE


def make_qdm_output(
    *,
    step: int,
    layer: int,
    kv_head: int,
    drift: QDMDriftEstimate,
    logits: torch.Tensor,
    top_k: int = 50,
    thresholds: QDMRiskThresholds | None = None,
) -> dict[str, Any]:
    """Build a diagnostic record; the label is not a calibrated risk signal."""
    metrics = compute_logit_metrics(logits, top_k=top_k)
    if not isinstance(metrics, QDMLogitMetrics):
        raise ValueError("make_qdm_output expects one logit row")
    result: dict[str, Any] = {
        "step": int(step),
        "layer": int(layer),
        "kv_head": int(kv_head),
        **drift.to_dict(),
        **metrics.to_dict(),
    }
    result["risk_state"] = classify_risk_state(
        drift.attention_error_bound,
        metrics.top1_top2_margin,
        thresholds=thresholds,
    ).value
    return result


@dataclass(frozen=True)
class TokenDecisionDiagnostics:
    """Reference-only validation signals computed from already available logits."""

    kl_divergence: torch.Tensor
    top1_flip: torch.Tensor
    reference_margin: torch.Tensor
    quantized_margin: torch.Tensor


def compute_token_decision_diagnostics(
    reference_logits: torch.Tensor,
    quantized_logits: torch.Tensor,
) -> TokenDecisionDiagnostics:
    """Compare BF16/reference and quantized logits without another forward."""
    if reference_logits.shape != quantized_logits.shape:
        raise ValueError("reference and quantized logits must have the same shape")
    if reference_logits.ndim not in (1, 2) or reference_logits.shape[-1] < 2:
        raise ValueError("logits must be [vocab] or [tokens, vocab]")
    ref = reference_logits.float()
    quant = quantized_logits.float()
    ref_log_prob = F.log_softmax(ref, dim=-1)
    quant_log_prob = F.log_softmax(quant, dim=-1)
    kl = (ref_log_prob.exp() * (ref_log_prob - quant_log_prob)).sum(dim=-1)
    ref_top = torch.topk(ref, k=2, dim=-1).values
    quant_top = torch.topk(quant, k=2, dim=-1).values
    return TokenDecisionDiagnostics(
        kl_divergence=kl,
        top1_flip=ref.argmax(dim=-1).ne(quant.argmax(dim=-1)),
        reference_margin=ref_top[..., 0] - ref_top[..., 1],
        quantized_margin=quant_top[..., 0] - quant_top[..., 1],
    )


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    if left.numel() != right.numel() or left.numel() < 2:
        raise ValueError("QDM correlation inputs must contain at least two values")
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((left * right).sum().div(denominator).item())


def qdm_validation_correlations(
    qdm_score: torch.Tensor,
    diagnostics: TokenDecisionDiagnostics,
) -> dict[str, float]:
    """Return reference correlations used by the Phase 1 validation study."""
    score = qdm_score.reshape(-1)
    kl = diagnostics.kl_divergence.reshape(-1)
    flip = diagnostics.top1_flip.float().reshape(-1)
    margin_drop = (
        diagnostics.reference_margin - diagnostics.quantized_margin
    ).reshape(-1)
    return {
        "qdm_kl_pearson": _pearson_correlation(score, kl),
        "qdm_top1_flip_pearson": _pearson_correlation(score, flip),
        "qdm_margin_drop_pearson": _pearson_correlation(score, margin_drop),
    }


estimate_attention_drift = estimate_qdm_drift

__all__ = [
    "QDM_BLOCK_SIZE",
    "QDM_LAYOUT",
    "QDM_QUANTIZER_VERSION",
    "QDM_VERSION",
    "PRECISION_ID_BF16",
    "PRECISION_ID_K2V2",
    "PRECISION_ID_K4V2",
    "PRECISION_ID_K8V4",
    "PRECISION_ID_MIXED",
    "QDMDecodeAccumulator",
    "QDMDriftEstimate",
    "QDMLogitMetrics",
    "QDMMetadata",
    "QDMObserver",
    "QDMRiskState",
    "QDMRiskThresholds",
    "QDMRuntimeEstimator",
    "QDMScalarAccumulator",
    "TokenDecisionDiagnostics",
    "classify_risk_state",
    "compute_logit_metrics",
    "compute_qdm_witness",
    "compute_token_decision_diagnostics",
    "estimate_attention_drift",
    "estimate_qdm_drift",
    "load_qdm_metadata",
    "make_qdm_output",
    "qdm_validation_correlations",
    "top1_top2_margin",
]
