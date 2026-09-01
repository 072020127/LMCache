# SPDX-License-Identifier: Apache-2.0

"""Tests for the K/V-separated MaKV precision schemes."""

# Standard
from types import SimpleNamespace
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.storage_backend.makv.config import (
    get_makv_config,
    validate_makv_runtime_config,
)
from lmcache.v1.storage_backend.makv.format import (
    decode_makv_object,
    encode_client_put_envelope,
)
from lmcache.v1.storage_backend.makv.plan import build_chunk_quant_plan
from lmcache.v1.storage_backend.makv.quantizer import quantize_canonical_kv
from lmcache.v1.storage_backend.makv.reference_dequant import dequantize_reference
from lmcache.v1.storage_backend.makv_remote.manager import MaKVRemoteManager


def _runtime_config(**overrides):
    extra = {
        "makv_storage_url": "file:///tmp/makv-kv-separate-test",
        "makv_precision_scheme": "kv_separate_3tier",
        "makv_bucket_ratios": [0.2, 0.3, 0.5],
        "makv_bucket_bits": [8, 4, 2],
        "makv_protect_prefix_tokens": 0,
        "makv_protect_tail_tokens": 0,
        "makv_require_cuda_dequant": False,
    }
    extra.update(overrides)
    return SimpleNamespace(remote_serde="makv", extra_config=extra)


def _runtime_config_4tier(**overrides):
    extra = {
        "makv_storage_url": "file:///tmp/makv-kv-separate-4tier-test",
        "makv_precision_scheme": "kv_separate_4tier",
        "makv_bucket_ratios": [0.1, 0.2, 0.5, 0.2],
        "makv_bucket_bits": [16, 8, 4, 2],
        "makv_protect_prefix_tokens": 0,
        "makv_protect_tail_tokens": 0,
        "makv_require_cuda_dequant": False,
    }
    extra.update(overrides)
    return SimpleNamespace(remote_serde="makv", extra_config=extra)


def _plan(config, *, importance, start=0, end=10, token_count=10, layers=1):
    return build_chunk_quant_plan(
        importance=importance,
        importance_layout_hint=None,
        chunk_start=start,
        chunk_end=end,
        original_shape=(2, layers, end - start, 3),
        original_strides=(layers * (end - start) * 3, (end - start) * 3, 3, 1),
        original_dtype="torch.float16",
        token_dim=2,
        num_layers=layers,
        num_kv_heads=1,
        head_dim=3,
        model_name="makv-kv-separate-test",
        world_size=1,
        worker_id=0,
        config=config,
        request_token_count=token_count,
    )


def test_kv_separate_config_defaults_and_validation():
    config = get_makv_config(_runtime_config(makv_bucket_bits=None))
    assert config.precision_scheme == "kv_separate_3tier"
    assert config.bucket_bits == (8, 4, 2)

    with pytest.raises(ValueError, match=r"requires makv_bucket_bits=\[8,4,2\]"):
        validate_makv_runtime_config(_runtime_config(makv_bucket_bits=[16, 8, 4]))


def test_kv_separate_4tier_defaults_and_mapping():
    config = get_makv_config(_runtime_config_4tier(makv_bucket_bits=None))
    assert config.precision_scheme == "kv_separate_4tier"
    assert config.bucket_ratios == (0.1, 0.2, 0.5, 0.2)
    assert config.bucket_bits == (16, 8, 4, 2)

    plan = _plan(
        config,
        importance=list(range(20, 0, -1)),
        end=20,
        token_count=20,
    )
    assert plan.precision_scheme == "kv_separate_4tier"
    assert plan.importance_layout == "layer_kv_token"
    assert plan.bucket_bits == (16, 8, 4, 2)
    assert list(plan.bucket_ids) == (
        [0, 0, 1, 1, 1, 1] + [2] * 10 + [3] * 4
        + [0, 0] + [2] * 4 + [3] * 14
    )

    with pytest.raises(ValueError, match=r"requires makv_bucket_bits=\[16,8,4,2\]"):
        validate_makv_runtime_config(
            _runtime_config_4tier(
                makv_bucket_ratios=[0.2, 0.3, 0.5],
                makv_bucket_bits=[8, 4, 2],
            )
        )


def test_score_tiers_are_ranked_request_wide_and_expanded_to_kv_positions():
    config = get_makv_config(_runtime_config())
    importance = list(range(10, 0, -1))
    plan = _plan(config, importance=importance, start=3, end=8)

    assert plan.precision_scheme == "kv_separate_3tier"
    assert plan.importance_layout == "layer_kv_token"
    assert plan.bucket_bits == (8, 4, 2)
    assert len(plan.bucket_ids) == 2 * 5

    # Request-wide tiers are [0, 0, 1, 1, 1, 2, 2, 2, 2, 2].  The chunk
    # [3:8] therefore contains K=[4,4,2,2,2] and V=[2,2,2,2,2].
    assert list(plan.bucket_ids) == [
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]


def test_layer_kv_scores_keep_independent_kv_bucket_maps():
    config = get_makv_config(_runtime_config())
    importance = torch.tensor(
        [
            [
                [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        ]
    )
    plan = _plan(config, importance=importance, layers=1)

    assert plan.importance_layout == "layer_kv_token"
    rows = torch.tensor(list(plan.bucket_ids)).view(1, 2, 10)
    assert not torch.equal(rows[0, 0], rows[0, 1])
    # The K and V rows are independently ranked, rather than sharing the
    # token bucket selected by the other plane.
    assert rows[0, 0, 0].item() == 0
    assert rows[0, 1, 0].item() == 2


def test_quantizer_and_reference_restore_use_separate_kv_positions():
    config = get_makv_config(_runtime_config())
    plan = _plan(config, importance=list(range(10, 0, -1)))
    canonical = torch.arange(1 * 2 * 10 * 1 * 3, dtype=torch.float32).view(
        1, 2, 10, 1, 3
    )
    metadata, payloads = quantize_canonical_kv(canonical, plan, config)

    entries = {int(entry["bits"]): entry for entry in metadata["bucket_entries"]}
    assert {bits: entries[bits]["count"] for bits in (8, 4, 2)} == {
        8: 2,
        4: 5,
        2: 13,
    }
    positions_8 = torch.frombuffer(
        bytearray(payloads["positions_8"]), dtype=torch.int32
    )
    assert list(positions_8) == [0, 1]

    bucket_payloads = {}
    scale_dtype = torch.float16
    for bits in (8, 4, 2):
        bucket_payloads[bits] = {
            "positions": torch.frombuffer(
                bytearray(payloads[f"positions_{bits}"]), dtype=torch.int32
            ),
            "payload": torch.frombuffer(
                bytearray(payloads[f"payload_{bits}"]),
                dtype=torch.int8 if bits == 8 else torch.uint8,
            ),
            "scales": torch.frombuffer(
                bytearray(payloads[f"scales_{bits}"]), dtype=scale_dtype
            ),
        }
    restored = dequantize_reference(
        plan=metadata["plan"],
        bucket_payloads=bucket_payloads,
        output_dtype=torch.float16,
    )
    assert restored.shape == (2, 1, 10, 3)
    # K token 0 is 8-bit and V token 0 is 4-bit, so both planes are restored
    # from different payload buckets.
    assert torch.count_nonzero(restored[0, 0, 0]) > 0
    assert torch.count_nonzero(restored[1, 0, 0]) > 0


def test_quantizer_and_reference_restore_use_4tier_raw16_bucket():
    config = get_makv_config(_runtime_config_4tier())
    plan = _plan(
        config,
        importance=list(range(20, 0, -1)),
        end=20,
        token_count=20,
    )
    canonical = torch.arange(1 * 2 * 20 * 1 * 3, dtype=torch.float16).view(
        1, 2, 20, 1, 3
    )
    metadata, payloads = quantize_canonical_kv(canonical, plan, config)
    entries = {int(entry["bits"]): entry for entry in metadata["bucket_entries"]}

    # K/V mapping yields 4, 4, 14 and 18 physical vectors in the 16/8/4/2
    # buckets respectively. The 16-bit bucket has no scales.
    assert {bits: entries[bits]["count"] for bits in (16, 8, 4, 2)} == {
        16: 4,
        8: 4,
        4: 14,
        2: 18,
    }
    assert payloads["scales_16"] == b""
    bucket_payloads = {
        bits: {
            "positions": torch.frombuffer(
                bytearray(payloads[f"positions_{bits}"]), dtype=torch.int32
            ),
            "payload": torch.frombuffer(
                bytearray(payloads[f"payload_{bits}"]),
                dtype=torch.float16
                if bits == 16
                else (torch.int8 if bits == 8 else torch.uint8),
            ),
            "scales": (
                torch.empty(0, dtype=torch.float16)
                if bits == 16
                else torch.frombuffer(
                    bytearray(payloads[f"scales_{bits}"]), dtype=torch.float16
                )
            ),
        }
        for bits in (16, 8, 4, 2)
    }
    restored = dequantize_reference(
        plan=metadata["plan"],
        bucket_payloads=bucket_payloads,
        output_dtype=torch.float16,
    )
    assert restored.shape == (2, 1, 20, 3)
    expected = canonical.permute(1, 0, 2, 3, 4).reshape(2, 1, 20, 3)
    assert torch.equal(restored[:, :, :2], expected[:, :, :2])


def test_remote_manager_is_the_only_quantizer_for_kv_separate_plan():
    class InMemoryAdapter:
        def __init__(self):
            self.objects = {}

        async def put(self, key, data):
            self.objects[key] = data

        async def get(self, key):
            return self.objects.get(key)

        async def close(self):
            return None

    config = get_makv_config(_runtime_config())
    plan = _plan(config, importance=list(range(10, 0, -1)))
    canonical = torch.arange(1 * 2 * 10 * 1 * 3, dtype=torch.float16).view(
        1, 2, 10, 1, 3
    )
    raw = (
        canonical.permute(1, 0, 2, 3, 4)
        .reshape(2, 1, 10, 3)
        .contiguous()
        .numpy()
        .tobytes()
    )
    envelope = encode_client_put_envelope(
        key="kv-separate-key",
        object_type="raw_with_plan",
        plan=plan,
        raw_kv_payload=raw,
        # Legacy request-side QDM controls must not opt in the production
        # manager when its shadow observer is disabled.
        extra_metadata={"qdm_enable": True, "qdm_block_size": 1},
    )
    manager = MaKVRemoteManager(config, InMemoryAdapter())
    stored_size = asyncio.run(manager.put("kv-separate-key", envelope, 0.0))
    stored = asyncio.run(manager.get("kv-separate-key"))
    asyncio.run(manager.close())

    assert manager.quantize_calls == 1
    assert stored is not None
    assert stored_size == len(stored)
    decoded = decode_makv_object(stored)
    assert decoded.object_type == "quantized"
    assert "qdm" not in decoded.metadata
