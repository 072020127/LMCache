# SPDX-License-Identifier: Apache-2.0

"""Optional elementwise MaKV quantization residuals.

The normal MaKV restore path intentionally ignores this module.  When enabled
on the remote manager, a residual is stored next to each low-bit bucket as
``original - dequantized``.  A manager-side precision upgrade can then rebuild
an approximate source tensor without retaining a second raw KV copy.
"""

from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import Any
import math
import warnings

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.makv.entropy import decode_entropy_payloads
from lmcache.v1.storage_backend.makv.reference_dequant import _unpack_low_bit

RESIDUAL_VERSION = 1
RESIDUAL_SEMANTICS = "original_minus_dequantized"
SUPPORTED_RESIDUAL_DTYPES = ("none", "float16", "float32")


def residual_payload_name(bits: int) -> str:
    """Return the serialized residual payload name for one precision bucket."""
    return f"residual_{int(bits)}"


def residual_torch_dtype(dtype: str) -> torch.dtype:
    """Map a configured residual dtype to its PyTorch dtype."""
    value = str(dtype).strip().lower()
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"unsupported MaKV residual dtype: {dtype!r}")


def _plan_value(plan: Mapping[str, Any], name: str) -> Any:
    return plan[name]


def _tensor_from_payload(payload: Any, dtype: torch.dtype) -> torch.Tensor:
    """Create an owned CPU tensor view over a serialized payload."""
    if torch.is_tensor(payload):
        return payload.to(dtype=dtype)
    if isinstance(payload, memoryview):
        raw = payload.tobytes()
    else:
        raw = bytes(payload)
    element_size = torch.empty((), dtype=dtype).element_size()
    if len(raw) % element_size:
        raise ValueError("MaKV residual payload is not aligned to its dtype")
    if not raw:
        return torch.empty((0,), dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(bytearray(raw), dtype=dtype)


def _payload_length(payload: Any) -> int:
    if torch.is_tensor(payload):
        return int(payload.numel() * payload.element_size())
    return len(payload)


def residual_shape(plan: Mapping[str, Any], count: int) -> tuple[int, ...]:
    """Return the vector shape used by one residual bucket."""
    layout = str(_plan_value(plan, "importance_layout"))
    if layout == "token":
        return (
            int(_plan_value(plan, "num_layers")),
            2,
            int(count),
            int(_plan_value(plan, "num_kv_heads")),
            int(_plan_value(plan, "head_dim")),
        )
    if layout == "layer_kv_token":
        return (
            int(count),
            int(_plan_value(plan, "num_kv_heads")),
            int(_plan_value(plan, "head_dim")),
        )
    raise ValueError(f"unsupported MaKV residual layout: {layout!r}")


def validate_residual_metadata(
    metadata: Mapping[str, Any], payloads: Mapping[str, Any]
) -> None:
    """Validate residual descriptors and payload lengths.

    Missing residual metadata is valid for legacy MaKV objects.  If a
    descriptor is present, every non-16-bit physical bucket must have exactly
    one correctly sized residual payload.
    """
    descriptor = metadata.get("residual")
    if descriptor is None:
        return
    if not isinstance(descriptor, Mapping):
        raise ValueError("MaKV residual metadata must be an object")
    if int(descriptor.get("version", -1)) != RESIDUAL_VERSION:
        raise ValueError("unsupported MaKV residual version")
    if descriptor.get("semantics") != RESIDUAL_SEMANTICS:
        raise ValueError("unsupported MaKV residual semantics")
    dtype_name = str(descriptor.get("dtype", ""))
    if dtype_name not in SUPPORTED_RESIDUAL_DTYPES[1:]:
        raise ValueError("MaKV residual dtype must be float16 or float32")
    plan = metadata.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("MaKV residual metadata is missing its plan")
    if str(descriptor.get("source_dtype", "")) != str(
        plan.get("original_dtype", "")
    ):
        raise ValueError("MaKV residual source dtype mismatch")
    bucket_bits = tuple(int(value) for value in plan.get("bucket_bits", ()))
    bucket_entries = metadata.get("bucket_entries")
    if not isinstance(bucket_entries, list):
        raise ValueError("MaKV residual metadata is missing bucket entries")
    bucket_counts: dict[int, int] = {}
    for entry in bucket_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid MaKV bucket entry")
        bit = int(entry.get("bits", -1))
        if bit in bucket_counts:
            raise ValueError("duplicate MaKV bucket entry")
        bucket_counts[bit] = int(entry.get("count", -1))

    entries = descriptor.get("buckets")
    if not isinstance(entries, list):
        raise ValueError("MaKV residual bucket table must be a list")
    expected_bits = {bit for bit in bucket_bits if bit != 16}
    seen: set[int] = set()
    residual_dtype = residual_torch_dtype(dtype_name)
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid MaKV residual bucket entry")
        bit = int(entry.get("bits", -1))
        if bit == 16 or bit not in expected_bits or bit in seen:
            raise ValueError("invalid or duplicate MaKV residual bucket")
        seen.add(bit)
        count = int(entry.get("count", -1))
        if count < 0:
            raise ValueError("MaKV residual bucket count must be non-negative")
        if count != bucket_counts.get(bit):
            raise ValueError("MaKV residual bucket count mismatch")
        shape = tuple(int(value) for value in entry.get("shape", ()))
        if shape != residual_shape(plan, count):
            raise ValueError("MaKV residual shape mismatch")
        element_count = math.prod(shape)
        if int(entry.get("element_count", -1)) != element_count:
            raise ValueError("MaKV residual element count mismatch")
        name = str(entry.get("payload_name", ""))
        if name != residual_payload_name(bit) or name not in payloads:
            raise ValueError("MaKV residual payload is missing")
        if _payload_length(payloads[name]) != element_count * residual_dtype.itemsize:
            raise ValueError("MaKV residual payload length mismatch")
    if seen != expected_bits:
        raise ValueError("MaKV residual buckets do not match the quantized buckets")


def _load_payloads(
    metadata: Mapping[str, Any], payloads: Mapping[str, Any]
) -> dict[int, dict[str, torch.Tensor]]:
    """Load compact bucket tensors, decoding optional arithmetic coding."""
    plan = metadata.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("MaKV object is missing its plan")
    raw_dtype = getattr(torch, str(plan["original_dtype"]).replace("torch.", ""))
    scale_dtype = (
        torch.float16 if str(plan["scale_dtype"]) == "float16" else torch.float32
    )

    def get_payload(name: str, dtype: torch.dtype) -> torch.Tensor:
        if name not in payloads:
            return torch.empty((0,), dtype=dtype)
        return _tensor_from_payload(payloads[name], dtype)

    decoded_entropy: dict[int, torch.Tensor] = {}
    if isinstance(metadata.get("entropy"), Mapping):
        decoded_entropy = decode_entropy_payloads(metadata, get_payload)

    result: dict[int, dict[str, torch.Tensor]] = {}
    for bit_value in plan["bucket_bits"]:
        bit = int(bit_value)
        payload_dtype = (
            torch.uint8 if bit in (2, 4) else torch.int8 if bit == 8 else raw_dtype
        )
        result[bit] = {
            "positions": get_payload(f"positions_{bit}", torch.int32),
            "payload": decoded_entropy.get(
                bit, get_payload(f"payload_{bit}", payload_dtype)
            ),
            "scales": get_payload(f"scales_{bit}", scale_dtype),
        }
    return result


def _decode_bucket_values(
    bit: int,
    tensors: Mapping[str, torch.Tensor],
    shape: tuple[int, ...],
    head_dim: int,
) -> torch.Tensor:
    """Decode one compact bucket to FP32 vectors for an upgrade operation."""
    rows = math.prod(shape[:-1])
    payload = tensors["payload"]
    scales = tensors["scales"]
    if bit == 16:
        if payload.numel() != math.prod(shape):
            raise ValueError("MaKV 16-bit payload element count mismatch")
        return payload.reshape(shape).float()
    if bit == 8:
        if payload.numel() != math.prod(shape):
            raise ValueError("MaKV INT8 payload element count mismatch")
        if scales.numel() != rows:
            raise ValueError("MaKV INT8 scale count mismatch")
        quantized = payload.reshape(shape).float()
    elif bit in (2, 4):
        bytes_per_row = (head_dim + (8 // bit) - 1) // (8 // bit)
        if payload.numel() != rows * bytes_per_row:
            raise ValueError(f"MaKV INT{bit} payload length mismatch")
        if scales.numel() != rows:
            raise ValueError(f"MaKV INT{bit} scale count mismatch")
        if rows == 0:
            return torch.empty(shape, dtype=torch.float32)
        quantized = _unpack_low_bit(
            payload.to(torch.uint8), rows, head_dim, bit
        ).reshape(shape).float()
    else:
        raise ValueError(f"unsupported MaKV bucket width: {bit}")
    return quantized * scales.reshape(shape[:-1]).float().unsqueeze(-1)


def reconstruct_with_residual(
    metadata: Mapping[str, Any],
    payloads: Mapping[str, Any],
) -> torch.Tensor:
    """Reconstruct the source KV layout using quantized values plus residuals.

    This function is intentionally manager-side and may allocate a full CPU
    tensor.  It is never called by the normal GPU restore path.
    """
    validate_residual_metadata(metadata, payloads)
    plan = metadata.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("MaKV object is missing its plan")
    num_layers = int(plan["num_layers"])
    num_heads = int(plan["num_kv_heads"])
    head_dim = int(plan["head_dim"])
    chunk_length = int(plan["chunk_length"])
    output_dtype = getattr(torch, str(plan["original_dtype"]).replace("torch.", ""))
    output = torch.zeros(
        (2, num_layers, chunk_length, num_heads * head_dim), dtype=output_dtype
    )
    tensors_by_bit = _load_payloads(metadata, payloads)
    residual_descriptor = metadata.get("residual")
    residual_by_bit: dict[int, Mapping[str, Any]] = {}
    if isinstance(residual_descriptor, Mapping):
        residual_by_bit = {
            int(entry["bits"]): entry
            for entry in residual_descriptor.get("buckets", [])
        }
    residual_dtype = (
        residual_torch_dtype(str(residual_descriptor["dtype"]))
        if isinstance(residual_descriptor, Mapping)
        else None
    )
    layout = str(plan["importance_layout"])
    seen: set[int] = set()
    expected_positions = (
        chunk_length if layout == "token" else num_layers * 2 * chunk_length
    )
    for bit_value in plan["bucket_bits"]:
        bit = int(bit_value)
        tensors = tensors_by_bit[bit]
        positions = tensors["positions"].to(torch.int64).reshape(-1)
        count = int(positions.numel())
        shape = residual_shape(plan, count)
        if layout == "token":
            if any(
                int(value) < 0 or int(value) >= chunk_length
                for value in positions.tolist()
            ):
                raise ValueError("MaKV token position is out of range")
            for value in positions.tolist():
                if int(value) in seen:
                    raise ValueError("MaKV token positions overlap")
                seen.add(int(value))
        else:
            if any(
                int(value) < 0 or int(value) >= expected_positions
                for value in positions.tolist()
            ):
                raise ValueError("MaKV layer/KV position is out of range")
            for value in positions.tolist():
                if int(value) in seen:
                    raise ValueError("MaKV layer/KV positions overlap")
                seen.add(int(value))
        values = _decode_bucket_values(bit, tensors, shape, head_dim)
        if bit != 16 and residual_dtype is not None:
            entry = residual_by_bit.get(bit)
            if entry is None:
                raise ValueError("MaKV residual payload is missing for a bucket")
            residual = _tensor_from_payload(
                payloads[str(entry["payload_name"])], residual_dtype
            ).reshape(shape).float()
            values = values + residual
        if layout == "token":
            restored = values.permute(1, 0, 2, 3, 4).reshape(
                2, num_layers, count, num_heads * head_dim
            )
            output[:, :, positions, :] = restored.to(output_dtype)
        else:
            layer = positions // (2 * chunk_length)
            remainder = positions % (2 * chunk_length)
            kv = remainder // chunk_length
            token = remainder % chunk_length
            output[kv, layer, token, :] = values.reshape(
                count, num_heads * head_dim
            ).to(output_dtype)
    if len(seen) != expected_positions:
        raise ValueError("MaKV bucket positions do not cover the KV object")
    return output


__all__ = [
    "RESIDUAL_SEMANTICS",
    "RESIDUAL_VERSION",
    "SUPPORTED_RESIDUAL_DTYPES",
    "reconstruct_with_residual",
    "residual_payload_name",
    "residual_shape",
    "residual_torch_dtype",
    "validate_residual_metadata",
]
