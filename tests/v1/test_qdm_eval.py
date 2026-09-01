# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from lmcache.v1.storage_backend.makv.qdm import QDMMetadata
from lmcache.v1.storage_backend.makv.qdm_eval import (
    QDMBlockObservation,
    aggregate_exact_attention_step,
    aggregate_qdm_step,
    assess_qdm_validation,
    assert_teacher_forced_prefix_alignment,
    assert_teacher_forced_sequence_alignment,
    build_downstream_sensitivity_analysis,
    build_paired_precision_analysis,
    compute_exact_attention_drift,
    compute_layer_drift_trace,
    make_teacher_forced_rows,
    teacher_forced_logit_metrics,
    write_qdm_validation_artifacts,
)
from lmcache.v1.storage_backend.makv.qdm_eval import (
    StreamingQDMCollector,
    qdm_reference_attention,
    qdm_streaming_attention,
)


def _metadata() -> QDMMetadata:
    return QDMMetadata(
        qdm_version="qdm_v1",
        quantizer_version="test-production",
        block_size=32,
        k_error=torch.tensor([[[1.0], [0.0]]]),
        v_error=torch.tensor([[[2.0], [0.0]]]),
        # The second witness block is intentionally smaller than the value
        # norm in the newly appended visible block below.
        v_norm=torch.tensor([[[3.0], [5.0]]]),
        precision_id=torch.zeros((1, 2, 1), dtype=torch.uint8),
    )


def test_teacher_forced_metrics_and_prefix_alignment():
    assert_teacher_forced_prefix_alignment([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="prefixes"):
        assert_teacher_forced_prefix_alignment([1, 2], [1, 3])
    hashes = assert_teacher_forced_sequence_alignment(
        [1, 2],
        [1, 2],
        reference_suffix_input_ids=[3, 4],
        quantized_suffix_input_ids=[3, 4],
        reference_target_token_ids=[4, 5],
        quantized_target_token_ids=[4, 5],
    )
    assert all(isinstance(value, str) for value in hashes.values())
    with pytest.raises(ValueError, match="suffixes"):
        assert_teacher_forced_sequence_alignment(
            [1, 2],
            [1, 2],
            reference_suffix_input_ids=[3],
            quantized_suffix_input_ids=[4],
        )

    reference = torch.tensor([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0]])
    quantized = torch.tensor([[2.0, 1.0, 0.0], [3.2, 2.8, 0.0]])
    metrics = teacher_forced_logit_metrics(reference, quantized, top_k=2)
    assert metrics[0]["top1_flip"] is False
    assert metrics[1]["top1_flip"] is True
    assert metrics[0]["kl_bf16_quantized"] == pytest.approx(0.0, abs=1e-7)
    assert metrics[1]["js_divergence"] >= 0.0
    assert metrics[0]["top1_top2_margin"] == pytest.approx(1.0)
    assert metrics[0]["topK_entropy"] == pytest.approx(0.58220315, rel=1e-5)


def test_qdm_v_norm_uses_all_visible_blocks_and_formula():
    observation = QDMBlockObservation(
        layer=0,
        step=0,
        query_head=0,
        kv_head=0,
        query_norm=2.0,
        block_probability=torch.tensor([0.25, 0.25, 0.50]),
        visible_v_norm=torch.tensor([3.0, 5.0, 10.0]),
    )
    result = aggregate_qdm_step(_metadata(), [observation], head_dim=4, expected_step=0)
    a = 0.25 * torch.exp(torch.tensor(1.0)) + 0.75
    tv = (a.square() - 1.0) / 2.0
    expected = 2.0 * tv * 10.0 + 0.25 * 2.0
    assert result["max_tv_bound"] == pytest.approx(float(tv))
    assert result["max_v_error"] == pytest.approx(0.5)
    assert result["v_norm_max"] == pytest.approx(10.0)
    assert result["max_attention_error"] == pytest.approx(float(expected))
    assert result["attention_error_formula_max_abs_error"] == pytest.approx(0.0)
    assert result["raw_A"] == pytest.approx(float(a))
    assert result["raw_tv_bound"] == pytest.approx(float(tv))
    assert result["log_A"] == pytest.approx(float(torch.log(a)))
    assert result["saturated"] is False
    assert result["saturation_rate"] == pytest.approx(0.0)
    assert result["mean_attention_error"] == pytest.approx(
        result["max_attention_error"]
    )
    assert result["p90_attention_error"] == pytest.approx(
        result["max_attention_error"]
    )
    assert result["p95_attention_error"] == pytest.approx(
        result["max_attention_error"]
    )
    assert result["top_k_layer_mean"] == pytest.approx(
        result["max_attention_error"]
    )
    assert result["saturation_by_layer"]["0"]["observation_count"] == 1


def test_qdm_v_norm_includes_visible_zero_mass_blocks():
    metadata = QDMMetadata(
        qdm_version="qdm_v1",
        quantizer_version="test-production",
        block_size=32,
        k_error=torch.tensor([[[0.0], [1.0]]]),
        v_error=torch.zeros((1, 2, 1)),
        v_norm=torch.tensor([[[100.0], [1.0]]]),
        precision_id=torch.zeros((1, 2, 1), dtype=torch.uint8),
    )
    observation = QDMBlockObservation(
        layer=0,
        step=0,
        query_head=0,
        kv_head=0,
        query_norm=2.0,
        block_probability=torch.tensor([0.0, 1.0]),
        visible_v_norm=torch.tensor([100.0, 1.0]),
    )
    result = aggregate_qdm_step(metadata, [observation], head_dim=4, expected_step=0)
    tv = torch.clamp((torch.exp(torch.tensor(1.0)).square() - 1.0) / 2.0, max=1.0)
    assert result["v_norm_max"] == pytest.approx(100.0)
    assert result["max_attention_error"] == pytest.approx(float(2.0 * tv * 100.0))


def test_qdm_saturation_keeps_raw_tv_diagnostic_separate_from_clamp():
    metadata = QDMMetadata(
        qdm_version="qdm_v1",
        quantizer_version="test-production",
        block_size=32,
        k_error=torch.tensor([[[5.0]]]),
        v_error=torch.zeros((1, 1, 1)),
        v_norm=torch.ones((1, 1, 1)),
        precision_id=torch.zeros((1, 1, 1), dtype=torch.uint8),
    )
    observation = QDMBlockObservation(
        layer=0,
        step=0,
        query_head=0,
        kv_head=0,
        query_norm=2.0,
        block_probability=torch.tensor([1.0]),
        visible_v_norm=torch.tensor([1.0]),
    )
    result = aggregate_qdm_step(metadata, [observation], head_dim=4, expected_step=0)
    assert result["raw_tv_bound"] > 1.0
    assert result["saturated"] is True
    assert result["saturation_rate"] == pytest.approx(1.0)
    assert result["max_tv_bound"] == pytest.approx(1.0)


def test_streaming_attention_matches_eager_without_attention_matrix():
    torch.manual_seed(4)
    query = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    key = torch.randn(1, 1, 5, 4, dtype=torch.float32)
    value = torch.randn(1, 1, 5, 4, dtype=torch.float32)
    value[0, 0, 4] = 100.0
    mask = torch.full((1, 1, 3, 5), torch.finfo(torch.float32).min)
    mask[:, :, 0, :3] = 0.0
    mask[:, :, 1, :4] = 0.0
    mask[:, :, 2, :5] = 0.0
    module = SimpleNamespace(layer_idx=2)
    collector = StreamingQDMCollector()
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    with qdm_reference_attention(model, collector):
        streaming, weights = qdm_streaming_attention(
            module,
            query,
            key,
            value,
            mask,
            scaling=0.5,
        )
    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    eager_weights = torch.softmax(
        torch.matmul(query, repeated_key.transpose(-2, -1)) * 0.5 + mask,
        dim=-1,
    )
    eager = torch.matmul(eager_weights, repeated_value).transpose(1, 2).contiguous()
    assert weights is None
    torch.testing.assert_close(streaming, eager, atol=1e-5, rtol=1e-5)
    assert len(collector.for_step(0)) == 2
    assert len(collector.for_step(1)) == 2
    assert len(collector.for_step(2)) == 2
    for step in range(3):
        for observation in collector.for_step(step):
            assert observation.block_probability.sum().item() == pytest.approx(1.0)
    assert collector.for_step(0)[0].visible_v_norm[0].item() < 100.0


def test_exact_attention_oracle_uses_query_direction_and_prefix_restore():
    query = torch.tensor([[[[1.0, 0.0]]]])
    reference_key = torch.tensor(
        [[[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]]]
    )
    reference_value = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]]
    )
    mask = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    collector = StreamingQDMCollector(capture_exact=True)
    module = SimpleNamespace(layer_idx=0)
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    with qdm_reference_attention(model, collector):
        qdm_streaming_attention(
            module,
            query,
            reference_key,
            reference_value,
            mask,
            scaling=1.0 / torch.sqrt(torch.tensor(2.0)).item(),
        )

    source = torch.zeros((1, 2, 3, 1, 2), dtype=torch.float32)
    source[0, 0, :, 0] = reference_key[0, 0, :3]
    source[0, 1, :, 0] = reference_value[0, 0, :3]
    restored = source.clone()
    restored[0, 0, 0, 0, 0] = 1.0
    restored[0, 1, 0, 0, 0] = 0.5
    metadata = QDMMetadata(
        qdm_version="qdm_v1",
        quantizer_version="test-production",
        block_size=32,
        k_error=torch.ones((1, 1, 1)),
        v_error=torch.full((1, 1, 1), 0.5),
        v_norm=torch.ones((1, 1, 1)),
        precision_id=torch.zeros((1, 1, 1), dtype=torch.uint8),
    )
    observations = compute_exact_attention_drift(
        metadata,
        collector,
        source,
        restored,
        prefix_length=3,
        head_dim=2,
    )
    assert len(observations) == 1
    assert observations[0]["exact_max_score_error"] == pytest.approx(1.0 / 2**0.5)
    assert observations[0]["exact_attention_TV"] > 0.0
    assert observations[0]["exact_attention_output_error"] > 0.0
    probability_error = observations[0]["reference_block_probability_max_abs_error"]
    assert probability_error == pytest.approx(0.0, abs=1e-6)
    aggregate = aggregate_exact_attention_step(observations)
    assert aggregate["exact_worst_layer_identity"] == "layer=0,kv_head=0"


def _qdm_metric(score: float) -> dict[str, float | int]:
    return {
        "qdm_score": score,
        "max_tv_bound": score,
        "p95_tv_bound": score,
        "max_v_error": score / 2.0,
        "max_attention_error": score,
        "mean_attention_error": score,
        "p90_attention_error": score,
        "p95_attention_error": score,
        "top_k_layer_mean": score,
        "worst_layer": 0,
        "worst_kv_head": 0,
        "v_norm_max": 7.0,
        "attention_error_formula_max_abs_error": 0.0,
        "qdm_observation_count": 1,
        "saturated_observation_count": 0,
        "saturation_rate": 0.0,
        "raw_A": 1.0,
        "raw_tv_bound": 0.0,
        "log_A": 0.0,
        "saturated": False,
        "saturation_by_layer": {
            "0": {"saturated_count": 0, "observation_count": 1}
        },
        "saturation_by_kv_head": {
            "0": {"saturated_count": 0, "observation_count": 1}
        },
        "saturation_by_layer_head": {
            "layer=0,kv_head=0": {
                "saturated_count": 0,
                "observation_count": 1,
            }
        },
    }


def test_qdm_artifacts_have_precision_buckets_and_diagnostic_enrichment(tmp_path):
    reference = torch.tensor(
        [[6.0, 1.0, 0.0], [3.0, 2.9, 0.0], [5.0, 1.0, 0.0], [2.01, 2.0, 0.0]]
    )
    flipped = torch.tensor(
        [[6.0, 1.0, 0.0], [2.8, 3.0, 0.0], [5.0, 1.0, 0.0], [1.9, 2.1, 0.0]]
    )
    rows = []
    for precision, scores in (
        ("K2V2", [0.1, 1.2, 0.2, 1.4]),
        ("K4V2", [0.05, 0.8, 0.1, 1.0]),
        ("K8V4", [0.01, 0.3, 0.02, 0.4]),
        ("MIXED", [0.08, 0.7, 0.12, 0.9]),
        ("BF16", [0.0, 0.0, 0.0, 0.0]),
    ):
        rows.extend(
            make_teacher_forced_rows(
                sample="synthetic",
                prefix_token_ids=[1, 2, 3],
                quantized_prefix_token_ids=[1, 2, 3],
                precision_composition=precision,
                reference_logits=reference,
                quantized_logits=flipped if precision != "BF16" else reference,
                qdm_by_step=[_qdm_metric(score) for score in scores],
                suffix_input_ids=[10, 11, 12, 13],
                target_token_ids=[11, 12, 13, 14],
                witness_summary={
                    "max_k_error": 0.0 if precision == "BF16" else 1.0,
                    "max_v_error": 0.0 if precision == "BF16" else 1.0,
                    "max_v_norm": 7.0,
                },
            )
        )

    summary = write_qdm_validation_artifacts(rows, tmp_path)
    written_rows = [
        json.loads(line)
        for line in (tmp_path / "per_token_qdm.jsonl").read_text().splitlines()
    ]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "saturation_report.json").exists()
    assert (tmp_path / "incremental_value.json").exists()
    assert (tmp_path / "paired_precision_qdm.jsonl").exists()
    assert (tmp_path / "precision_monotonicity.json").exists()
    assert (tmp_path / "paired_counterfactual_report.md").exists()
    assert (tmp_path / "layer_drift_trace.jsonl").exists()
    assert (tmp_path / "downstream_sensitivity.json").exists()
    assert (tmp_path / "sensitivity_report.md").exists()
    report_text = (tmp_path / "validation_report.md").read_text()
    assert "TV Saturation Diagnostics" in report_text
    assert "Margin-Controlled Incremental Value" in (
        report_text
    )
    assert {row["precision_composition"] for row in written_rows} == {
        "K2V2",
        "K4V2",
        "K8V4",
        "MIXED",
        "BF16",
    }
    assert all(row["teacher_forced"] for row in written_rows)
    assert all(row["risk_state"] != "UNASSIGNED" for row in written_rows)
    assert summary["bf16_witness"]["observed"] is True
    assert summary["bf16_witness"]["all_error_witness_zero"] is True
    assert (
        summary["saturation_report"]["by_precision"]["BF16"]["saturation_rate"]
        == pytest.approx(0.0)
    )
    assert summary["incremental_value"]["overall"]["count"] == 16
    assert (
        summary["precision_buckets"]["K2V2"]["correlations"]["qdm_vs_kl_pearson"]
        is not None
    )
    comparison = summary["precision_buckets"]["K2V2"]["metric_comparison"]
    for metric in ("max_tv_bound", "p95_tv_bound", "max_attention_error"):
        assert comparison[metric]["spearman_vs_kl"] is not None
        assert comparison[metric]["auroc_top1_flip"] is not None
        assert comparison[metric]["pr_auc_top1_flip"] is not None
    assert (
        len(
            summary["precision_buckets"]["K2V2"]["quantile_analysis"][
                "max_attention_error"
            ]["buckets"]
        )
        == 4
    )
    assert (
        summary["precision_buckets"]["K2V2"]["qdm_plus_margin"]["auroc_top1_flip"]
        is not None
    )
    assert summary["precision_buckets"]["K2V2"]["diagnostic_enrichment"][
        "qdm_high_small_margin_flip_rate"
    ] == pytest.approx(1.0)
    assert (tmp_path / "validation_report.md").exists()
    assert summary["validation"]["status"] == "INCONCLUSIVE"


def test_layer_drift_trace_projects_margin_gradient_and_cross_layer_behavior():
    layers, steps, hidden, vocab = 3, 2, 4, 6
    reference_attention = [torch.zeros(1, steps, hidden) for _ in range(layers)]
    quantized_attention = [
        tensor + (index + 1) * 0.1
        for index, tensor in enumerate(reference_attention)
    ]
    reference_hidden = [
        torch.ones(1, steps, hidden) * (index + 1) for index in range(layers)
    ]
    quantized_hidden = [
        tensor + (index + 1) * 0.05
        for index, tensor in enumerate(reference_hidden)
    ]
    gradients = [torch.ones(steps, hidden) * (index + 1) for index in range(layers)]
    reference_logits = torch.zeros(steps, vocab)
    reference_logits[:, 0] = 2.0
    reference_logits[:, 1] = 1.0
    quantized_logits = reference_logits.clone()
    quantized_logits[:, 0] -= 0.25
    traces = compute_layer_drift_trace(
        reference_attention,
        quantized_attention,
        reference_hidden,
        quantized_hidden,
        gradients,
        reference_logits,
        quantized_logits,
    )
    assert len(traces) == steps
    assert len(traces[0]["layers"]) == layers
    assert traces[0]["summary"]["physical_norm_only"] > 0.0
    assert traces[0]["summary"]["sensitivity_weighted_error"] > 0.0
    assert traces[0]["summary"]["hidden_amplified_transition_count"] == 2
    assert traces[0]["summary"]["hidden_attenuated_transition_count"] == 0
    assert traces[0]["summary"]["directional_cancellation_ratio"] == pytest.approx(0.0)
    assert traces[0]["summary"]["margin_abs_delta"] == pytest.approx(0.25)


def test_downstream_sensitivity_analysis_requires_complete_paired_oracle():
    reference = torch.tensor([[4.0, 1.0, 0.0], [3.0, 2.0, 0.0]])
    rows = []
    for precision in ("BF16", "K8V4", "K4V2", "K2V2", "MIXED"):
        rows.extend(
            make_teacher_forced_rows(
                sample="sensitivity",
                prefix_token_ids=[1, 2, 3],
                quantized_prefix_token_ids=[1, 2, 3],
                precision_composition=precision,
                reference_logits=reference,
                quantized_logits=reference,
                qdm_by_step=[
                    {
                        **_qdm_metric(0.0),
                        "exact_max_score_error": 0.0,
                        "exact_attention_TV": 0.0,
                        "exact_attention_output_error": 0.0,
                        "exact_oracle": True,
                    }
                    for _ in range(2)
                ],
                suffix_input_ids=[4, 5],
                target_token_ids=[5, 6],
                context_length=1024,
                requested_context_length=1024,
                row_metadata={
                    "precision_plan_source": "test",
                    "prefix_alignment_hash": "prefix",
                    "suffix_alignment_hash": "suffix",
                    "target_alignment_hash": "target",
                },
                witness_summary={
                    "precision_ids": [3]
                    if precision == "BF16"
                    else [0, 1, 2, 3]
                    if precision == "MIXED"
                    else [2],
                    "max_k_error": 0.0,
                    "max_v_error": 0.0,
                },
            )
        )
    for row in rows:
        row.update(
            {
                "downstream_sensitivity_available": True,
                "physical_norm_only": 0.0,
                "sensitivity_weighted_error": 0.0,
                "sensitivity_signed_error": 0.0,
                "hidden_delta_norm_sum": 0.0,
                "hidden_delta_norm_max": 0.0,
                "hidden_delta_norm_final": 0.0,
                "logit_delta_l2": 0.0,
                "margin_abs_delta": 0.0,
                "_layer_drift_trace": {
                    "step": row["step"],
                    "layers": [],
                    "summary": {
                        "physical_norm_only": 0.0,
                        "sensitivity_weighted_error": 0.0,
                    },
                },
            }
        )
    analysis = build_downstream_sensitivity_analysis(rows)
    assert analysis["summary"]["available"] is True
    assert analysis["summary"]["paired_token_count"] == 2
    transition = analysis["summary"]["sensitivity_vs_physical"]["overall"][
        "K8V4_to_K4V2"
    ]
    assert "vs_delta_logit_delta_l2" in transition
    assert "vs_delta_margin_abs" in transition
    assert isinstance(
        analysis["summary"]["sensitivity_vs_physical"]["by_layer"], dict
    )
    rows[0].pop("sensitivity_weighted_error")
    failed = build_downstream_sensitivity_analysis(rows)
    assert failed["summary"]["available"] is False
    assert failed["summary"]["integrity"]["fail_closed"] is True


def test_paired_precision_analysis_is_same_token_and_fail_closed():
    reference = torch.tensor(
        [[6.0, 1.0, 0.0], [5.0, 1.0, 0.0]], dtype=torch.float32
    )
    precision_scores = {
        "BF16": [0.0, 0.0],
        "K8V4": [1.0, 1.0],
        "K4V2": [2.0, 2.0],
        "K2V2": [3.0, 3.0],
        "MIXED": [1.5, 1.5],
    }
    precision_ids = {
        "BF16": [3],
        "K8V4": [2],
        "K4V2": [1],
        "K2V2": [0],
        "MIXED": [0, 1, 2, 3],
    }
    rows = []
    for precision, scores in precision_scores.items():
        qdm_rows = []
        for score in scores:
            qdm = _qdm_metric(score)
            qdm.update(
                {
                    "exact_max_score_error": score,
                    "exact_attention_TV": score,
                    "exact_attention_output_error": score,
                    "exact_oracle": True,
                }
            )
            qdm_rows.append(qdm)
        rows.extend(
            make_teacher_forced_rows(
                sample="paired",
                prefix_token_ids=[1, 2, 3],
                quantized_prefix_token_ids=[1, 2, 3],
                precision_composition=precision,
                reference_logits=reference,
                quantized_logits=reference,
                qdm_by_step=qdm_rows,
                suffix_input_ids=[10, 11],
                target_token_ids=[11, 12],
                context_length=1024,
                requested_context_length=1024,
                row_metadata={
                    "precision_plan_source": "test",
                    "prefix_alignment_hash": "prefix",
                    "suffix_alignment_hash": "suffix",
                    "target_alignment_hash": "target",
                },
                witness_summary={
                    "precision_ids": precision_ids[precision],
                    "max_k_error": scores[0],
                    "max_v_error": scores[0],
                },
            )
        )

    analysis = build_paired_precision_analysis(rows)
    assert (
        analysis["summary"]["available"] is True
    ), analysis["summary"]["integrity"]["invalid_groups"]
    assert analysis["summary"]["paired_token_count"] == 2
    monotonic = analysis["summary"]["monotonicity"]["overall"]["metrics"]
    assert monotonic["exact_attention_output_error"]["monotonic_token_fraction"] == 1.0
    delta = analysis["records"][0]["deltas"]["K8V4_to_K4V2"]
    assert delta["physical"]["exact_attention_output_error"] == pytest.approx(1.0)
    assert analysis["records"][0]["precision"]["MIXED"][
        "actual_block_precision_composition"
    ]["observed_precision_names"] == ["K2V2", "K4V2", "K8V4", "BF16"]

    rows[0]["prefix_alignment_hash"] = "mismatch"
    failed = build_paired_precision_analysis(rows)
    assert failed["summary"]["available"] is False
    assert failed["summary"]["integrity"]["fail_closed"] is True


def test_bf16_physical_qdm_drift_is_fail_closed(tmp_path):
    reference = torch.tensor([[4.0, 1.0, 0.0], [3.0, 2.0, 0.0]])
    qdm = [_qdm_metric(0.0), _qdm_metric(0.0)]
    rows = make_teacher_forced_rows(
        sample="bf16",
        prefix_token_ids=[1, 2, 3],
        quantized_prefix_token_ids=[1, 2, 3],
        precision_composition="BF16",
        reference_logits=reference,
        quantized_logits=reference,
        qdm_by_step=qdm,
        suffix_input_ids=[4, 5],
        target_token_ids=[5, 6],
        witness_summary={"max_k_error": 0.0, "max_v_error": 0.0},
    )
    summary = write_qdm_validation_artifacts(rows, tmp_path)
    summary["bf16_witness"]["physical_drift_max"] = 1.0
    summary["bf16_witness"]["all_physical_drift_zero"] = False
    validation = assess_qdm_validation(summary)
    assert validation["status"] == "FAILED"
