# SPDX-License-Identifier: Apache-2.0
"""Benchmarks for MaKV plan/build, remote round-trip, and restore paths.

Run with:

    PYTHONPATH=$PWD/LMCache .venv/bin/python -m pytest \
        LMCache/tests/benchmarks/test_makv.py --benchmark-only
"""

# Standard
import asyncio

# Third Party
import pytest
import torch

# First Party
import lmcache.c_ops as lmc_ops
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.makv.config import get_makv_config
from lmcache.v1.storage_backend.makv.gpu_restore import (
    restore_makv_quantized_to_tensor,
)
from lmcache.v1.storage_backend.makv.paged_restore import (
    makv_paged_cuda_op_available,
    restore_makv_quantized_to_paged,
)
from lmcache.v1.storage_backend.makv.plan import build_chunk_quant_plan
from lmcache.v1.storage_backend.makv.serde import MaKVDeserializer, MaKVSerializer
from tests.v1.utils import close_asyncio_loop, dumb_cache_engine_key, init_asyncio_loop


def _make_config(tmp_path, **extra) -> LMCacheEngineConfig:
    extra_config = {
        "makv_storage_url": f"file://{tmp_path}",
        "makv_bucket_ratios": [0.20, 0.30, 0.50],
        "makv_bucket_bits": [16, 8, 4],
        "makv_protect_prefix_tokens": 0,
        "makv_protect_tail_tokens": 0,
        "makv_require_cuda_dequant": False,
        "makv_fallback": "naive",
    }
    extra_config.update(extra)
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        remote_url="makv://127.0.0.1:65432",
        remote_serde="makv",
        extra_config=extra_config,
    )


def _metadata(chunk_size: int) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="makv-bench-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, chunk_size, 8, 16),
        chunk_size=chunk_size,
    )


def _memory_obj(metadata: LMCacheMetadata) -> TensorMemoryObj:
    shape = metadata.get_shapes()[0]
    tensor = torch.arange(shape.numel(), dtype=metadata.kv_dtype).view(shape)
    return TensorMemoryObj(
        raw_data=tensor,
        metadata=MemoryObjMetadata(
            shape=shape,
            dtype=tensor.dtype,
            address=-1,
            phy_size=tensor.numel() * tensor.element_size(),
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )


def _importance(chunk_size: int) -> list[float]:
    return [float(chunk_size - i) / float(chunk_size) for i in range(chunk_size)]


@pytest.mark.benchmark(group="makv_plan")
@pytest.mark.parametrize("chunk_size", [64, 256, 1024])
def test_makv_quant_plan_bench(benchmark, tmp_path, chunk_size):
    config = _make_config(tmp_path)
    metadata = _metadata(chunk_size)
    makv_config = get_makv_config(config)

    def run():
        return build_chunk_quant_plan(
            importance=_importance(chunk_size),
            importance_layout_hint="token",
            chunk_start=0,
            chunk_end=chunk_size,
            original_shape=(
                2,
                metadata.kv_shape[0],
                chunk_size,
                metadata.kv_shape[3] * metadata.kv_shape[4],
            ),
            original_strides=(
                metadata.kv_shape[0]
                * chunk_size
                * metadata.kv_shape[3]
                * metadata.kv_shape[4],
                chunk_size * metadata.kv_shape[3] * metadata.kv_shape[4],
                metadata.kv_shape[3] * metadata.kv_shape[4],
                1,
            ),
            original_dtype="torch.float16",
            token_dim=2,
            num_layers=metadata.kv_shape[0],
            num_kv_heads=metadata.kv_shape[3],
            head_dim=metadata.kv_shape[4],
            model_name=metadata.model_name,
            world_size=metadata.world_size,
            worker_id=metadata.worker_id,
            config=makv_config,
        )

    benchmark(run)


@pytest.mark.benchmark(group="makv_remote_roundtrip")
@pytest.mark.parametrize("chunk_size", [64, 256, 1024])
def test_makv_remote_roundtrip_bench(benchmark, tmp_path, chunk_size):
    config = _make_config(tmp_path)
    metadata = _metadata(chunk_size)
    serializer = MaKVSerializer(config, metadata)
    deserializer = MaKVDeserializer(config, metadata)
    memory_obj = _memory_obj(metadata)
    transfer_spec = {
        "chunk_start": 0,
        "chunk_end": chunk_size,
        "request_token_count": chunk_size,
        "makv_importance": _importance(chunk_size),
        "makv_importance_layout": "token",
        "request_configs": {},
    }
    async_loop, async_thread = init_asyncio_loop()
    connector = CreateConnector(config.remote_url, async_loop, None, config, metadata)
    key = dumb_cache_engine_key(chunk_size)

    def run():
        envelope = serializer.serialize(
            memory_obj, transfer_spec=transfer_spec, key=key
        )
        asyncio.run_coroutine_threadsafe(
            connector.put(key, envelope), async_loop
        ).result()
        stored = asyncio.run_coroutine_threadsafe(
            connector.get(key), async_loop
        ).result()
        return deserializer.deserialize(stored)

    try:
        benchmark(run)
    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.benchmark(group="makv_restore_cpu")
@pytest.mark.parametrize("chunk_size", [64, 256, 1024])
def test_makv_restore_reference_bench(benchmark, tmp_path, chunk_size):
    config = _make_config(tmp_path)
    metadata = _metadata(chunk_size)
    serializer = MaKVSerializer(config, metadata)
    deserializer = MaKVDeserializer(config, metadata)
    memory_obj = _memory_obj(metadata)
    transfer_spec = {
        "chunk_start": 0,
        "chunk_end": chunk_size,
        "request_token_count": chunk_size,
        "makv_importance": _importance(chunk_size),
        "makv_importance_layout": "token",
        "request_configs": {},
    }
    async_loop, async_thread = init_asyncio_loop()
    connector = CreateConnector(config.remote_url, async_loop, None, config, metadata)
    key = dumb_cache_engine_key(chunk_size + 1)
    try:
        envelope = serializer.serialize(
            memory_obj, transfer_spec=transfer_spec, key=key
        )
        asyncio.run_coroutine_threadsafe(
            connector.put(key, envelope), async_loop
        ).result()
        stored = asyncio.run_coroutine_threadsafe(
            connector.get(key), async_loop
        ).result()
        quantized_obj = deserializer.deserialize(stored)
        benchmark(
            lambda: restore_makv_quantized_to_tensor(
                quantized_obj, torch.device("cpu"), require_cuda=False
            )
        )
    finally:
        close_asyncio_loop(async_loop, async_thread)


@pytest.mark.benchmark(group="makv_restore_cuda")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA runtime")
@pytest.mark.skipif(
    not makv_paged_cuda_op_available(), reason="Requires MaKV paged CUDA op"
)
@pytest.mark.parametrize("chunk_size", [64, 256, 1024])
def test_makv_restore_cuda_bench(benchmark, tmp_path, chunk_size):
    config = _make_config(tmp_path, makv_require_cuda_dequant=True)
    metadata = _metadata(chunk_size)
    serializer = MaKVSerializer(config, metadata)
    deserializer = MaKVDeserializer(config, metadata)
    memory_obj = _memory_obj(metadata)
    transfer_spec = {
        "chunk_start": 0,
        "chunk_end": chunk_size,
        "request_token_count": chunk_size,
        "makv_importance": _importance(chunk_size),
        "makv_importance_layout": "token",
        "request_configs": {},
    }
    async_loop, async_thread = init_asyncio_loop()
    connector = CreateConnector(config.remote_url, async_loop, None, config, metadata)
    key = dumb_cache_engine_key(chunk_size + 2)
    try:
        envelope = serializer.serialize(
            memory_obj, transfer_spec=transfer_spec, key=key
        )
        asyncio.run_coroutine_threadsafe(
            connector.put(key, envelope), async_loop
        ).result()
        stored = asyncio.run_coroutine_threadsafe(
            connector.get(key), async_loop
        ).result()
        quantized_obj = deserializer.deserialize(stored)

        block_size = 16
        num_blocks = (chunk_size + block_size - 1) // block_size
        num_heads = metadata.kv_shape[3]
        head_dim = metadata.kv_shape[4]
        caches = [
            torch.empty(
                (2, num_blocks, block_size, num_heads, head_dim),
                dtype=metadata.kv_dtype,
                device="cuda",
            )
            for _ in range(metadata.kv_shape[0])
        ]
        page_ptrs = torch.tensor(
            [cache.data_ptr() for cache in caches],
            dtype=torch.int64,
            device="cuda",
        )
        slots = torch.arange(chunk_size, dtype=torch.int64, device="cuda")

        def run():
            restore_makv_quantized_to_paged(
                quantized_obj,
                device=torch.device("cuda"),
                page_ptrs=page_ptrs,
                slot_mapping=slots,
                page_buffer_size=num_blocks * block_size,
                block_size=block_size,
                head_size=head_dim,
                engine_kv_format=(lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS),
            )
            torch.cuda.synchronize()

        benchmark(run)
    finally:
        close_asyncio_loop(async_loop, async_thread)
