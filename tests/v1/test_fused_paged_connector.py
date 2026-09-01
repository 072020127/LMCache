# SPDX-License-Identifier: Apache-2.0

"""Regression tests for split LMCache tensors and fused vLLM paged caches."""

import pytest
import torch

from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _memory_obj(tensor: torch.Tensor) -> TensorMemoryObj:
    return TensorMemoryObj(
        raw_data=tensor,
        metadata=MemoryObjMetadata(
            shape=tensor.shape,
            dtype=tensor.dtype,
            address=-1,
            phy_size=tensor.numel() * tensor.element_size(),
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("layout", ("hnd", "nhd"))
def test_v2_fused_paged_cpu_round_trip(dtype: torch.dtype, layout: str) -> None:
    layers, blocks, block_size, heads, head_dim, tokens = 2, 5, 4, 3, 8, 11
    device = torch.device("cuda")
    source = (
        torch.arange(2 * layers * tokens * heads * head_dim, device=device)
        .reshape(2, layers, tokens, heads * head_dim)
        .to(dtype)
    )
    slots = torch.tensor(
        [13, 0, 7, 18, 3, 11, 1, 15, 6, 9, 4],
        device=device,
        dtype=torch.int64,
    )
    if layout == "hnd":
        cache_shape = (blocks, heads, block_size, 2 * head_dim)
    else:
        cache_shape = (blocks, block_size, heads, 2 * head_dim)
    caches = [
        torch.full(cache_shape, -1, device=device, dtype=dtype) for _ in range(layers)
    ]
    connector = VLLMPagedMemGPUConnectorV2(
        hidden_dim_size=heads * head_dim,
        num_layers=layers,
        use_gpu=False,
        dtype=dtype,
        device=device,
        layout_hints={"kv_layout": layout.upper()},
    )

    connector.to_gpu(
        _memory_obj(source), 0, tokens, kvcaches=caches, slot_mapping=slots
    )
    torch.cuda.synchronize()
    cpu_obj = _memory_obj(torch.empty_like(source, device="cpu"))
    connector.from_gpu(cpu_obj, 0, tokens, kvcaches=caches, slot_mapping=slots)
    assert torch.equal(cpu_obj.tensor, source.cpu())

    for cache in caches:
        cache.zero_()
    connector.to_gpu(cpu_obj, 0, tokens, kvcaches=caches, slot_mapping=slots)
    torch.cuda.synchronize()
    gpu_obj = _memory_obj(torch.empty_like(source, device=device))
    connector.from_gpu(gpu_obj, 0, tokens, kvcaches=caches, slot_mapping=slots)
    torch.cuda.synchronize()
    assert torch.equal(gpu_obj.tensor, source)
