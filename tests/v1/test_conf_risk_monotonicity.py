# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.scoutrank_transfer.conf_risk_monotonicity import (
    MonotonicityValidationError,
    analyze_records,
    run_analysis,
)


PRECISIONS = ("BF16", "K8V4", "K4V2", "K2V2")
BYTES = {"BF16": 1000, "K8V4": 700, "K4V2": 500, "K2V2": 300}


def _rows(*, reverse: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    risks = {"BF16": 0.10, "K8V4": 0.20, "K4V2": 0.30, "K2V2": 0.40}
    for sample in ("prompt-a", "prompt-b"):
        for context in (1024, 2048):
            for step in range(8):
                hashes = {
                    "prefix_alignment_hash": f"prefix-{sample}-{context}",
                    "suffix_alignment_hash": f"suffix-{sample}-{context}",
                    "target_alignment_hash": f"target-{sample}-{context}",
                    "prefix_token_id_hash": f"prefix-token-{sample}-{context}",
                    "suffix_input_token_id_hash": f"suffix-token-{sample}-{context}",
                    "target_token_id_hash": f"target-token-{sample}-{context}",
                }
                for precision in PRECISIONS:
                    risk = risks[precision] + step * 0.001
                    if reverse and precision == "K2V2":
                        risk = 0.05 + step * 0.001
                    rows.append(
                        {
                            "protocol": "conf_makv_v1",
                            "scorer_version": "conf_makv_v1",
                            "semantics": "uncalibrated_precision_risk",
                            "sample": sample,
                            "requested_context_length": context,
                            "context_length": context,
                            "step": step,
                            "precision_composition": precision,
                            "prefix_length": 10,
                            "teacher_forced": True,
                            "free_running_ground_truth": False,
                            "prefix_aligned": True,
                            "suffix_aligned": True,
                            "target_aligned": True,
                            "risk_input": "current_decode_path_logits",
                            "risk_scorer_api": "compute_precision_risk_signal",
                            "kv_bytes_source": (
                                "production_serializer_actual_serialized_bytes"
                            ),
                            "kv_serializer": "LMCache.MaKV.encode_makv_object",
                            "kv_serialized_bytes": BYTES[precision],
                            "risk": risk,
                            "full_conf_risk": risk,
                            "margin_only_risk": risk + 0.01,
                            **hashes,
                        }
                    )
    return rows


def test_monotonicity_reports_paired_and_stratified_evidence():
    summary, records = analyze_records(_rows())

    assert summary["status"] == "MONOTONIC_SIGNAL_SUPPORTED"
    assert len(records) == 32
    assert summary["aligned_token_count"] == 32
    assert summary["prompt_count"] == 2
    assert summary["context_lengths"] == [1024, 2048]
    assert summary["overall"]["storage"]["K2V2"]["serialized_bytes"]["mean"] == 300.0
    for transition in (
        "BF16_to_K8V4",
        "K8V4_to_K4V2",
        "K4V2_to_K2V2",
    ):
        lift = summary["overall"]["scorers"]["full_conf"]["paired_risk_lifts"][
            transition
        ]
        assert lift["median"] > 0.0
        assert lift["positive_fraction"] == 1.0
    assert (
        summary["overall"]["scorers"]["full_conf"]["monotonicity"]["monotonic_fraction"]
        == 1.0
    )
    assert (
        summary["overall"]["scorers"]["full_conf"]["compression_ratio_vs_risk"][
            "pooled_over_precision_rows"
        ]
        > 0.0
    )
    assert summary["stability"]["full_conf"]["context"]["BF16_to_K8V4"][
        "majority_direction"
    ]


def test_monotonicity_is_inconclusive_when_precision_order_reverses():
    summary, _ = analyze_records(_rows(reverse=True))

    assert summary["status"] == "INCONCLUSIVE"
    assert (
        summary["overall"]["scorers"]["full_conf"]["paired_risk_lifts"]["K4V2_to_K2V2"][
            "median"
        ]
        < 0.0
    )


def test_monotonicity_fails_closed_on_missing_precision_or_alignment():
    rows = _rows()
    rows = [row for row in rows if row["precision_composition"] != "K2V2"]
    with pytest.raises(MonotonicityValidationError, match="missing precisions"):
        analyze_records(rows)

    rows = _rows()
    rows[0]["target_alignment_hash"] = "mismatched"
    with pytest.raises(MonotonicityValidationError, match="target_alignment_hash"):
        analyze_records(rows)


def test_run_analysis_writes_requested_artifacts(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "per_token_conf_makv.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8"
    )
    (input_dir / "summary.json").write_text("{}", encoding="utf-8")

    summary = run_analysis(
        input_dir,
        output_dir,
        require_real_qwen3=False,
    )

    assert summary["status"] == "MONOTONIC_SIGNAL_SUPPORTED"
    assert {
        "risk_by_precision.jsonl",
        "risk_monotonicity_summary.json",
        "risk_monotonicity_report.md",
    } == {path.name for path in output_dir.iterdir()}
