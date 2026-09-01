# SPDX-License-Identifier: Apache-2.0

# Standard
import time
import warnings

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.makv.memory import MaKVQuantizedMemoryObj
from lmcache.v1.storage_backend.makv.entropy import decode_entropy_payloads
from lmcache.v1.storage_backend.makv.metrics import RESTORE_METRICS
from lmcache.v1.storage_backend.makv.reference_dequant import dequantize_reference

logger = init_logger(__name__)


def makv_cuda_op_available() -> bool:
    return hasattr(torch.ops, "lmcache_makv") and hasattr(
        torch.ops.lmcache_makv, "dequantize_scatter_out"
    )


def _empty_tensor(
    dtype: torch.dtype, device: torch.device | str = "cpu"
) -> torch.Tensor:
    return torch.empty((0,), dtype=dtype, device=device)


def _buffer_to_tensor(
    buffer: bytes | memoryview, dtype: torch.dtype
) -> torch.Tensor:
    if len(buffer) == 0:
        return _empty_tensor(dtype)
    # Keep the network response as the backing storage until the one required
    # pin-memory copy. Converting through bytearray would copy every payload.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(buffer, dtype=dtype)


def _pinned_blob_tensor(memory_obj: MaKVQuantizedMemoryObj) -> torch.Tensor:
    """Make one pinned byte copy for all payload views in one MaKV object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        blob = torch.frombuffer(memory_obj.byte_array, dtype=torch.uint8)
    if blob.numel() == 0 or blob.is_pinned():
        return blob
    return blob.pin_memory()


def payload_tensors_from_obj(
    memory_obj: MaKVQuantizedMemoryObj,
    *,
    pinned_blob: torch.Tensor | None = None,
    decode_entropy: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    plan = memory_obj.makv_metadata["plan"]
    raw_dtype = getattr(torch, plan["original_dtype"].replace("torch.", ""))
    scale_dtype = torch.float16 if plan["scale_dtype"] == "float16" else torch.float32
    payload_table = {
        str(entry["name"]): entry
        for entry in memory_obj.makv_metadata.get("_payload_table", [])
    }
    pinned_buffer = (
        memoryview(pinned_blob.numpy()) if pinned_blob is not None else None
    )

    def make_tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        entry = payload_table.get(name)
        if pinned_buffer is not None and entry is not None:
            offset = int(entry["offset"])
            length = int(entry["length"])
            if length == 0:
                return _empty_tensor(dtype)
            # frombuffer accepts byte-aligned slices; Tensor.view(dtype) does
            # not, while JSON metadata is not required to align payloads.
            return _buffer_to_tensor(
                pinned_buffer[offset : offset + length], dtype
            )
        return _buffer_to_tensor(memory_obj.makv_payloads.get(name, b""), dtype)

    tensors: dict[int, dict[str, torch.Tensor]] = {}
    for bit in plan["bucket_bits"]:
        bit = int(bit)
        payload_dtype = (
            torch.uint8 if bit in (2, 4) else (torch.int8 if bit == 8 else raw_dtype)
        )
        tensors[bit] = {
            "positions": make_tensor(f"positions_{bit}", torch.int32),
            "payload": make_tensor(f"payload_{bit}", payload_dtype),
            "scales": make_tensor(f"scales_{bit}", scale_dtype),
        }
    for bit in (16, 8, 4, 2):
        if bit not in tensors:
            payload_dtype = (
                torch.uint8
                if bit in (2, 4)
                else (torch.int8 if bit == 8 else raw_dtype)
            )
            tensors[bit] = {
                "positions": _empty_tensor(torch.int32),
                "payload": _empty_tensor(payload_dtype),
                "scales": _empty_tensor(scale_dtype),
            }
    if decode_entropy and isinstance(memory_obj.makv_metadata.get("entropy"), dict):
        decode_started = time.perf_counter()
        decoded = decode_entropy_payloads(memory_obj.makv_metadata, make_tensor)
        for bit, payload in decoded.items():
            tensors[bit]["payload"] = payload
        entropy = memory_obj.makv_metadata["entropy"]
        entropy_bytes = sum(
            int(plane.get("stream_bytes", 0))
            + int(plane.get("cdf_bytes", 0))
            + int(plane.get("lengths_bytes", 0))
            for bucket in entropy.get("buckets", {}).values()
            for plane in bucket.get("planes", [])
        )
        RESTORE_METRICS.add(
            makv_entropy_decode_calls=len(decoded),
            makv_entropy_decode_time_ms=(time.perf_counter() - decode_started) * 1000,
            makv_entropy_decode_bytes=entropy_bytes,
        )
    return tensors


def payload_tensors_from_gpu_blob(
    memory_obj: MaKVQuantizedMemoryObj,
    gpu_blob: torch.Tensor,
    *,
    cpu_payloads: dict[int, dict[str, torch.Tensor]] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Create device views into one contiguous GPU MaKV blob.

    New MaKV objects align every payload segment, allowing the view to avoid
    one H2D allocation per bucket. Objects written by older protocol-v1
    encoders may have unaligned offsets; those individual segments use the
    existing copy path instead of making the whole restore fail.
    """
    if gpu_blob.device.type != "cuda" or gpu_blob.dtype != torch.uint8:
        raise ValueError("gpu_blob must be a CUDA uint8 tensor")
    if not gpu_blob.is_contiguous():
        raise ValueError("gpu_blob must be contiguous")
    plan = memory_obj.makv_metadata["plan"]
    raw_dtype = getattr(torch, plan["original_dtype"].replace("torch.", ""))
    scale_dtype = torch.float16 if plan["scale_dtype"] == "float16" else torch.float32
    payload_table = {
        str(entry["name"]): entry
        for entry in memory_obj.makv_metadata.get("_payload_table", [])
    }
    entropy_enabled = isinstance(memory_obj.makv_metadata.get("entropy"), dict)
    cpu_payloads = cpu_payloads or payload_tensors_from_obj(
        memory_obj, decode_entropy=False if entropy_enabled else True
    )

    def make_named_tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        entry = payload_table.get(name)
        if entry is None:
            raw = memory_obj.makv_payloads.get(name, b"")
            return _buffer_to_tensor(raw, dtype).to(
                device=gpu_blob.device, non_blocking=True
            )
        offset = int(entry["offset"])
        length = int(entry["length"])
        element_size = torch.empty((), dtype=dtype).element_size()
        if length == 0:
            return _empty_tensor(dtype, gpu_blob.device)
        if (
            offset % element_size == 0
            and length % element_size == 0
            and offset >= 0
            and offset + length <= gpu_blob.numel()
        ):
            return gpu_blob.narrow(0, offset, length).view(dtype)
        raw = memory_obj.makv_payloads.get(name, b"")
        return _buffer_to_tensor(raw, dtype).to(
            device=gpu_blob.device, non_blocking=True
        )

    def make_tensor(
        bits: int, name: str, dtype: torch.dtype
    ) -> torch.Tensor:
        entry = payload_table.get(name)
        if entry is None:
            # Unit-created/legacy objects may retain decoded payloads without
            # the serialized offset table. Keep their established copy path.
            field = name.rsplit("_", 1)[0]
            return cpu_payloads[bits][field].to(
                device=gpu_blob.device, non_blocking=True
            )
        return make_named_tensor(name, dtype)

    tensors: dict[int, dict[str, torch.Tensor]] = {}
    for bit in plan["bucket_bits"]:
        bit = int(bit)
        payload_dtype = (
            torch.uint8
            if bit in (2, 4)
            else (torch.int8 if bit == 8 else raw_dtype)
        )
        tensors[bit] = {
            "positions": make_tensor(bit, f"positions_{bit}", torch.int32),
            "payload": make_tensor(bit, f"payload_{bit}", payload_dtype),
            "scales": make_tensor(bit, f"scales_{bit}", scale_dtype),
        }
    for bit in (16, 8, 4, 2):
        if bit not in tensors:
            payload_dtype = (
                torch.uint8
                if bit in (2, 4)
                else (torch.int8 if bit == 8 else raw_dtype)
            )
            tensors[bit] = {
                "positions": _empty_tensor(torch.int32, gpu_blob.device),
                "payload": _empty_tensor(payload_dtype, gpu_blob.device),
                "scales": _empty_tensor(scale_dtype, gpu_blob.device),
            }
    if entropy_enabled:
        decode_started = time.perf_counter()
        decoded = decode_entropy_payloads(
            memory_obj.makv_metadata, make_named_tensor
        )
        for bit, payload in decoded.items():
            tensors[bit]["payload"] = payload
        entropy = memory_obj.makv_metadata["entropy"]
        entropy_bytes = sum(
            int(plane.get("stream_bytes", 0))
            + int(plane.get("cdf_bytes", 0))
            + int(plane.get("lengths_bytes", 0))
            for bucket in entropy.get("buckets", {}).values()
            for plane in bucket.get("planes", [])
        )
        RESTORE_METRICS.add(
            makv_entropy_decode_calls=len(decoded),
            makv_entropy_decode_time_ms=(time.perf_counter() - decode_started) * 1000,
            makv_entropy_decode_bytes=entropy_bytes,
        )
    return tensors


def restore_makv_quantized_to_tensor(
    memory_obj: MaKVQuantizedMemoryObj,
    device: torch.device,
    require_cuda: bool,
) -> TensorMemoryObj:
    plan = memory_obj.makv_metadata["plan"]
    output_dtype = getattr(torch, plan["original_dtype"].replace("torch.", ""))
    start_time = time.perf_counter()
    cuda_path = device.type == "cuda" and makv_cuda_op_available()
    pinned_blob = _pinned_blob_tensor(memory_obj) if cuda_path else None
    if cuda_path:
        gpu_blob = pinned_blob.to(device=device, non_blocking=True)
        payload_tensors = payload_tensors_from_gpu_blob(memory_obj, gpu_blob)
    else:
        payload_tensors = payload_tensors_from_obj(memory_obj)
    output = torch.empty(
        (
            2,
            plan["num_layers"],
            plan["chunk_length"],
            plan["num_kv_heads"] * plan["head_dim"],
        ),
        dtype=output_dtype,
        device=device,
    )
    if device.type == "cuda" and makv_cuda_op_available():
        h2d_start = time.perf_counter()
        pos16 = payload_tensors[16]["positions"].to(device=device, non_blocking=True)
        pos8 = payload_tensors[8]["positions"].to(device=device, non_blocking=True)
        pos4 = payload_tensors[4]["positions"].to(device=device, non_blocking=True)
        pos2 = payload_tensors[2]["positions"].to(device=device, non_blocking=True)
        raw16 = payload_tensors[16]["payload"].to(device=device, non_blocking=True)
        int8_payload = payload_tensors[8]["payload"].to(
            device=device, non_blocking=True
        )
        int4_payload = payload_tensors[4]["payload"].to(
            device=device, non_blocking=True
        )
        int2_payload = payload_tensors[2]["payload"].to(
            device=device, non_blocking=True
        )
        int8_scales = payload_tensors[8]["scales"].to(device=device, non_blocking=True)
        int4_scales = payload_tensors[4]["scales"].to(device=device, non_blocking=True)
        int2_scales = payload_tensors[2]["scales"].to(device=device, non_blocking=True)
        RESTORE_METRICS.add(makv_h2d_time_ms=(time.perf_counter() - h2d_start) * 1000)
        kernel_start = time.perf_counter()
        torch.ops.lmcache_makv.dequantize_scatter_out(
            raw16,
            int8_payload,
            int4_payload,
            int2_payload,
            int8_scales,
            int4_scales,
            int2_scales,
            pos16,
            pos8,
            pos4,
            pos2,
            output,
            int(plan["importance_layout"] == "layer_kv_token"),
            int(plan["num_layers"]),
            int(plan["num_kv_heads"]),
            int(plan["head_dim"]),
        )
        RESTORE_METRICS.add(
            makv_dequant_kernel_time_ms=(time.perf_counter() - kernel_start) * 1000
        )
    else:
        if device.type == "cuda" and require_cuda:
            raise RuntimeError(
                "MaKV CUDA dequantization is required but the operator is unavailable"
            )
        restored = dequantize_reference(
            plan=plan,
            bucket_payloads=payload_tensors,
            output_dtype=output_dtype,
        )
        output.copy_(restored.to(device))
    RESTORE_METRICS.add(
        makv_restore_total_time_ms=(time.perf_counter() - start_time) * 1000
    )
    result = TensorMemoryObj(
        raw_data=output,
        metadata=MemoryObjMetadata(
            shape=output.shape,
            dtype=output.dtype,
            address=-1,
            phy_size=output.numel() * output.element_size(),
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )
    if cuda_path:
        # The contiguous path is retained for compatibility and differential
        # tests.  Its dequant kernel is asynchronous, so keep decoded views
        # alive and attach them to the returned object until the consumer's
        # stream has completed.
        stream = torch.cuda.current_stream(device)
        keepalive = [pinned_blob, gpu_blob]
        keepalive.extend(
            tensor
            for bucket in payload_tensors.values()
            for tensor in bucket.values()
            if tensor.device.type == "cuda"
        )
        for tensor in keepalive:
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                tensor.record_stream(stream)
        result._makv_restore_keepalive = tuple(keepalive)
    return result
