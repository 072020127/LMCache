# SPDX-License-Identifier: Apache-2.0

"""Reference MaKV dequantizer used for tests and CPU fallback."""

# Standard
from typing import Any

# Third Party
import torch


def _unpack_low_bit(
    packed: torch.Tensor, logical_count: int, head_dim: int, bits: int
) -> torch.Tensor:
    if bits not in (2, 4):
        raise ValueError(f"Unsupported packed MaKV width: {bits}")
    rows = logical_count
    values_per_byte = 8 // bits
    bytes_per_row = (head_dim + values_per_byte - 1) // values_per_byte
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    packed = packed.view(rows, bytes_per_row).to(torch.int16)
    shifts = (
        torch.arange(values_per_byte, dtype=torch.int16, device=packed.device)
        * bits
    )
    fields = ((packed.unsqueeze(-1) >> shifts) & mask).reshape(rows, -1)
    signed = (fields ^ sign) - sign
    return signed[:, :head_dim].to(torch.int8)


def dequantize_bucket_vectors(
    payload: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    *,
    vector_shape: tuple[int, ...],
    head_dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize one production bucket without restoring a full KV tensor.

    The payload is already produced by ``quantizer._quantize_vector_bucket``.
    This helper only applies the same unpack, scale and output-cast semantics
    used by the reference restore path.
    """
    if not vector_shape or vector_shape[-1] != head_dim:
        raise ValueError("vector_shape must end in head_dim")
    rows = 1
    for dimension in vector_shape[:-1]:
        rows *= int(dimension)
    if bits == 16:
        return payload.reshape(vector_shape).to(output_dtype)
    if bits == 8:
        quantized = payload.reshape(vector_shape).to(torch.float32)
    elif bits in (2, 4):
        unpacked = _unpack_low_bit(
            payload.to(torch.uint8), rows, head_dim, bits
        )
        quantized = unpacked.reshape(vector_shape).to(torch.float32)
    else:
        raise ValueError(f"Unsupported MaKV dequantization width: {bits}")
    scale_shape = tuple(int(dimension) for dimension in vector_shape[:-1])
    scale = scales.reshape(scale_shape).to(torch.float32)
    return (quantized * scale.unsqueeze(-1)).to(output_dtype)


def dequantize_reference(
    *,
    plan: dict[str, Any],
    bucket_payloads: dict[int, dict[str, torch.Tensor]],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Restore a canonical ``[2,L,T,H*D]`` KV tensor from MaKV buckets."""
    num_layers = int(plan["num_layers"])
    num_heads = int(plan["num_kv_heads"])
    head_dim = int(plan["head_dim"])
    chunk_length = int(plan["chunk_length"])
    output = torch.zeros(
        (2, num_layers, chunk_length, num_heads * head_dim),
        dtype=output_dtype,
    )
    layout = str(plan["importance_layout"])
    for bits, tensors in bucket_payloads.items():
        positions = tensors["positions"].to(torch.int64)
        if positions.numel() == 0:
            continue
        if layout == "token":
            if bits == 16:
                values = (
                    tensors["payload"]
                    .view(num_layers, 2, positions.numel(), num_heads, head_dim)
                    .permute(1, 0, 2, 3, 4)
                )
            elif bits == 8:
                q = (
                    tensors["payload"]
                    .view(num_layers, 2, positions.numel(), num_heads, head_dim)
                    .to(torch.float32)
                )
                scales = tensors["scales"].view(
                    num_layers, 2, positions.numel(), num_heads
                )
                values = (q * scales.unsqueeze(-1)).permute(1, 0, 2, 3, 4)
            else:
                logical_rows = num_layers * 2 * positions.numel() * num_heads
                unpacked = _unpack_low_bit(
                    tensors["payload"].to(torch.uint8),
                    logical_rows,
                    head_dim,
                    bits,
                ).view(num_layers, 2, positions.numel(), num_heads, head_dim)
                scales = tensors["scales"].view(
                    num_layers, 2, positions.numel(), num_heads
                )
                values = (unpacked.to(torch.float32) * scales.unsqueeze(-1)).permute(
                    1, 0, 2, 3, 4
                )
            for idx, token_pos in enumerate(positions.tolist()):
                output[:, :, token_pos, :] = (
                    values[:, :, idx]
                    .reshape(2, num_layers, num_heads * head_dim)
                    .to(output_dtype)
                )
        else:
            if bits == 16:
                values = tensors["payload"].view(positions.numel(), num_heads, head_dim)
            elif bits == 8:
                values = tensors["payload"].view(
                    positions.numel(), num_heads, head_dim
                ).to(torch.float32) * tensors["scales"].view(
                    positions.numel(), num_heads, 1
                )
            else:
                unpacked = _unpack_low_bit(
                    tensors["payload"].to(torch.uint8),
                    positions.numel() * num_heads,
                    head_dim,
                    bits,
                ).view(positions.numel(), num_heads, head_dim)
                values = unpacked.to(torch.float32) * tensors["scales"].view(
                    positions.numel(), num_heads, 1
                )
            for idx, flat_pos in enumerate(positions.tolist()):
                layer_idx, rem = divmod(flat_pos, 2 * chunk_length)
                kv_idx, token_pos = divmod(rem, chunk_length)
                output[kv_idx, layer_idx, token_pos, :] = (
                    values[idx].reshape(num_heads * head_dim).to(output_dtype)
                )
    return output
