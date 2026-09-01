# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from lmcache.v1.storage_backend.makv.config import MaKVConfig, get_makv_config
from lmcache.v1.storage_backend.makv.plan import MaKVQuantPlan
from lmcache.v1.storage_backend.makv.qdm import (
    PRECISION_ID_K8V4,
    PRECISION_ID_MIXED,
    QDMMetadata,
    QDMRiskState,
    QDMRiskThresholds,
    QDMRuntimeEstimator,
    classify_risk_state,
    compute_logit_metrics,
    compute_token_decision_diagnostics,
    load_qdm_metadata,
    make_qdm_output,
    qdm_validation_correlations,
)
from lmcache.v1.storage_backend.makv.quantizer import quantize_canonical_kv
from lmcache.v1.storage_backend.makv.reference_dequant import dequantize_reference


def _config(bits: tuple[int, ...]) -> MaKVConfig:
    return MaKVConfig(
        storage_url="file:///tmp/makv-qdm",
        bucket_ratios=tuple(1.0 / len(bits) for _ in bits),
        bucket_bits=bits,
        importance_layout="layer_kv_token",
        quant_granularity="per_token_head",
        scale_dtype="float16",
        protect_prefix_tokens=0,
        protect_tail_tokens=0,
        dequant_backend="reference",
        require_cuda_dequant=False,
        fallback="miss",
        enable_checksum=True,
    )


def _plan(
    *,
    layout: str,
    bucket_bits: tuple[int, ...],
    bucket_ids: list[int],
    layers: int = 1,
    tokens: int = 40,
    heads: int = 2,
    head_dim: int = 4,
) -> MaKVQuantPlan:
    return MaKVQuantPlan(
        protocol_version=1,
        importance_layout=layout,
        token_count=tokens,
        chunk_start=0,
        chunk_length=tokens,
        bucket_bits=bucket_bits,
        bucket_ids=bytes(bucket_ids),
        original_shape=(2, layers, tokens, heads * head_dim),
        original_strides=(
            layers * tokens * heads * head_dim,
            tokens * heads * head_dim,
            heads * head_dim,
            1,
        ),
        original_dtype="torch.bfloat16",
        token_dim=2,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        quant_granularity="per_token_head",
        scale_dtype="float16",
        model_fingerprint="qdm-test-model",
        parallel_fingerprint="qdm-test-worker",
        checksum=0,
    )


def _bucket_payloads(payloads: dict[str, bytes], bits: tuple[int, ...]):
    result = {}
    for bit in bits:
        payload_dtype = (
            torch.float16
            if bit == 16
            else torch.int8
            if bit == 8
            else torch.uint8
        )
        result[bit] = {
            "positions": torch.frombuffer(
                bytearray(payloads[f"positions_{bit}"]), dtype=torch.int32
            ),
            "payload": torch.frombuffer(
                bytearray(payloads[f"payload_{bit}"]), dtype=payload_dtype
            ),
            "scales": torch.frombuffer(
                bytearray(payloads[f"scales_{bit}"]), dtype=torch.float16
            )
            if payloads[f"scales_{bit}"]
            else torch.empty(0, dtype=torch.float16),
        }
    return result


def test_qdm_witness_uses_production_payload_and_preserves_payload_bytes():
    torch.manual_seed(12)
    layers, tokens, heads, head_dim = 1, 40, 2, 4
    bits = (8, 4, 2)
    ids = [0] * tokens + [1] * 32 + [2] * 8
    plan = _plan(
        layout="layer_kv_token",
        bucket_bits=bits,
        bucket_ids=ids,
        layers=layers,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
    )
    source = torch.randn(
        layers, 2, tokens, heads, head_dim, dtype=torch.bfloat16
    )
    config = _config(bits)
    _, payloads_without_qdm = quantize_canonical_kv(source, plan, config)
    metadata, payloads = quantize_canonical_kv(
        source, plan, config, enable_qdm=True, qdm_block_size=32
    )

    assert all(
        payloads[name] == payload
        for name, payload in payloads_without_qdm.items()
    )
    assert metadata["qdm"]["layout"] == "layer_block_kv_head"
    qdm = load_qdm_metadata(metadata, payloads)
    assert qdm is not None
    assert qdm.shape == (layers, 2, heads)
    assert qdm.valid_tokens.tolist() == [32, 8]
    assert qdm.precision_id[0, 0, 0].item() == PRECISION_ID_K8V4
    assert qdm.precision_id[0, 1, 0].item() == PRECISION_ID_MIXED

    restored = dequantize_reference(
        plan=metadata["plan"],
        bucket_payloads=_bucket_payloads(payloads, bits),
        output_dtype=source.dtype,
    ).view(2, layers, tokens, heads, head_dim)
    source_kv = source.permute(1, 0, 2, 3, 4)
    expected_k = torch.zeros_like(qdm.k_error)
    expected_v = torch.zeros_like(qdm.v_error)
    expected_norm = torch.zeros_like(qdm.v_norm)
    for block in range(2):
        start, end = block * 32, min(tokens, (block + 1) * 32)
        expected_k[:, block] = torch.sqrt(
            (source_kv[0, :, start:end].float() - restored[0, :, start:end].float())
            .square()
            .sum(dim=-1)
        ).amax(dim=1)
        expected_v[:, block] = torch.sqrt(
            (source_kv[1, :, start:end].float() - restored[1, :, start:end].float())
            .square()
            .sum(dim=-1)
        ).amax(dim=1)
        expected_norm[:, block] = torch.sqrt(
            source_kv[1, :, start:end].float().square().sum(dim=-1)
        ).amax(dim=1)
    torch.testing.assert_close(qdm.k_error, expected_k)
    torch.testing.assert_close(qdm.v_error, expected_v)
    torch.testing.assert_close(qdm.v_norm, expected_norm)

    roundtrip = QDMMetadata.from_descriptor(
        qdm.to_descriptor(), qdm.to_payloads()
    )
    assert roundtrip is not None
    torch.testing.assert_close(roundtrip.k_error, qdm.k_error)
    torch.testing.assert_close(roundtrip.v_error, qdm.v_error)
    torch.testing.assert_close(roundtrip.v_norm, qdm.v_norm)


def test_qdm_bf16_witness_is_zero_and_disabled_path_has_no_qdm():
    bits = (16,)
    tokens = 33
    plan = _plan(
        layout="token",
        bucket_bits=bits,
        bucket_ids=[0] * tokens,
        tokens=tokens,
    )
    source = torch.randn(1, 2, tokens, 2, 4, dtype=torch.bfloat16)
    config = _config(bits)
    plain_metadata, plain_payloads = quantize_canonical_kv(source, plan, config)
    qdm_metadata, qdm_payloads = quantize_canonical_kv(
        source, plan, config, enable_qdm=True
    )
    assert "qdm" not in plain_metadata
    assert all(plain_payloads[key] == qdm_payloads[key] for key in plain_payloads)
    qdm = load_qdm_metadata(qdm_metadata, qdm_payloads)
    assert qdm is not None
    assert qdm.precision_id.unique().tolist() == [3]
    assert torch.count_nonzero(qdm.k_error) == 0
    assert torch.count_nonzero(qdm.v_error) == 0


def test_qdm_disabled_hard_gates_supplied_observer_and_legacy_metadata():
    bits = (8, 4, 2)
    tokens = 40
    plan = _plan(
        layout="token",
        bucket_bits=bits,
        bucket_ids=[0] * tokens,
        tokens=tokens,
    )
    source = torch.randn(1, 2, tokens, 2, 4, dtype=torch.bfloat16)
    config = _config(bits)
    plain_metadata, plain_payloads = quantize_canonical_kv(source, plan, config)

    class ObserverMustNotRun:
        def observe_bucket(self, **kwargs):
            raise AssertionError("QDM observer ran while QDM was disabled")

        def finalize(self):
            raise AssertionError("QDM observer finalized while QDM was disabled")

    disabled_metadata, disabled_payloads = quantize_canonical_kv(
        source,
        plan,
        config,
        qdm_observer=ObserverMustNotRun(),
        enable_qdm=False,
    )
    assert disabled_metadata == plain_metadata
    assert disabled_payloads == plain_payloads
    assert load_qdm_metadata({"plan": plain_metadata["plan"]}, plain_payloads) is None


def test_qdm_config_defaults_to_disabled_and_disabled_config_ignores_bad_block_size():
    class RuntimeConfig:
        remote_serde = "makv"
        remote_url = ""
        extra_config = {
            "makv_storage_url": "file:///tmp/makv-qdm-config",
            "makv_require_cuda_dequant": False,
            "makv_qdm_block_size": 0,
        }

    config = get_makv_config(RuntimeConfig())
    assert config.enable_qdm is False


def test_qdm_runtime_bound_matches_scalar_formula():
    metadata = QDMMetadata(
        qdm_version="qdm_v1",
        quantizer_version="test",
        block_size=32,
        k_error=torch.tensor([[[0.0], [1.0]]]),
        v_error=torch.tensor([[[2.0], [4.0]]]),
        v_norm=torch.tensor([[[3.0], [5.0]]]),
        precision_id=torch.zeros((1, 2, 1), dtype=torch.uint8),
    )
    probability = torch.tensor([0.25, 0.75])
    estimate = QDMRuntimeEstimator(metadata).estimate(
        torch.ones(4), probability, layer=0, kv_head=0
    )
    a = 0.25 + 0.75 * math.exp(1.0)
    tv = min(1.0, (a * a - 1.0) / 2.0)
    assert estimate.k_tv_bound == pytest.approx(tv)
    assert estimate.v_error == pytest.approx(3.5)
    assert estimate.attention_error_bound == pytest.approx(2.0 * tv * 5.0 + 3.5)
    visible_range_estimate = QDMRuntimeEstimator(metadata).estimate(
        torch.ones(4),
        probability,
        layer=0,
        kv_head=0,
        visible_v_norm_max=10.0,
    )
    assert visible_range_estimate.v_norm_max == pytest.approx(10.0)
    assert visible_range_estimate.attention_error_bound == pytest.approx(
        2.0 * tv * 10.0 + 3.5
    )


def test_qdm_token_metrics_states_and_phase1_diagnostics():
    thresholds = QDMRiskThresholds(
        drift_high_threshold=1.0, margin_small_threshold=0.2
    )
    assert classify_risk_state(0.1, 1.0, thresholds=thresholds) is QDMRiskState.SAFE
    assert (
        classify_risk_state(0.1, 0.1, thresholds=thresholds)
        is QDMRiskState.MODEL_FRAGILE
    )
    assert (
        classify_risk_state(2.0, 1.0, thresholds=thresholds)
        is QDMRiskState.KV_DRIFT_ROBUST
    )
    assert (
        classify_risk_state(2.0, 0.1, thresholds=thresholds)
        is QDMRiskState.KV_TOKEN_RISK
    )

    metrics = compute_logit_metrics(torch.tensor([2.0, 1.0, 0.0]), top_k=2)
    assert metrics.top1_top2_margin == pytest.approx(1.0)
    assert metrics.topk_entropy == pytest.approx(0.58220315, rel=1e-5)

    from lmcache.v1.storage_backend.makv.qdm import QDMDriftEstimate

    output = make_qdm_output(
        step=7,
        layer=3,
        kv_head=2,
        drift=QDMDriftEstimate(0.2, 0.3, 0.4),
        logits=torch.tensor([2.0, 1.0, 0.0]),
        top_k=2,
        thresholds=thresholds,
    )
    assert output["step"] == 7
    assert output["layer"] == 3
    assert output["kv_head"] == 2
    assert output["risk_state"] == "SAFE"
    assert {
        "k_tv_bound",
        "v_error",
        "attention_error_bound",
        "top1_margin",
        "risk_state",
    } <= output.keys()

    reference = torch.tensor([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0]])
    quantized = torch.tensor([[3.0, 1.0, 0.0], [3.1, 2.9, 0.0]])
    diagnostics = compute_token_decision_diagnostics(reference, quantized)
    correlations = qdm_validation_correlations(
        torch.tensor([0.1, 0.9]), diagnostics
    )
    assert diagnostics.top1_flip.tolist() == [False, True]
    assert set(correlations) == {
        "qdm_kl_pearson",
        "qdm_top1_flip_pearson",
        "qdm_margin_drop_pearson",
    }
