# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from lmcache.v1.storage_backend.makv.conf_makv import (
    CONF_MAKV_WEIGHTS,
    compute_confidence_metrics,
    make_conf_teacher_forced_rows,
    write_conf_makv_artifacts,
)


def _rows():
    reference = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [3.0, 2.9, 0.0, 0.0],
            [1.0, 2.0, 0.0, -1.0],
            [0.4, 0.3, 0.2, 0.0],
        ]
    )
    shifts = {
        "BF16": torch.zeros_like(reference),
        "K8V4": torch.tensor(
            [[0.0, 0.1, 0.0, 0.0], [0.0, -0.1, 0.0, 0.0], [0.0] * 4, [0.0] * 4]
        ),
        "K4V2": torch.tensor(
            [
                [-0.1, 0.1, 0.0, 0.0],
                [0.0, -0.2, 0.0, 0.0],
                [0.2, -0.2, 0.0, 0.0],
                [0.1, -0.1, 0.0, 0.0],
            ]
        ),
        "K2V2": torch.tensor(
            [
                [-0.4, 0.2, 0.1, 0.0],
                [0.2, -0.3, 0.0, 0.0],
                [1.2, -0.8, 0.0, 0.0],
                [0.2, -0.2, 0.0, 0.0],
            ]
        ),
        "MIXED": torch.tensor(
            [
                [-0.2, 0.1, 0.0, 0.0],
                [0.1, -0.1, 0.0, 0.0],
                [0.4, -0.2, 0.0, 0.0],
                [0.1, -0.1, 0.0, 0.0],
            ]
        ),
    }
    output = []
    for sample, context_length in (("sample-0", 1024), ("sample-1", 2048)):
        for precision, shift in shifts.items():
            output.extend(
                make_conf_teacher_forced_rows(
                    sample=sample,
                    prefix_token_ids=list(range(8)),
                    precision_composition=precision,
                    reference_logits=reference,
                    current_logits=reference + shift,
                    current_prefix_token_ids=list(range(8)),
                    suffix_input_ids=[8, 9, 10, 11],
                    target_token_ids=[9, 10, 11, 12],
                    current_suffix_input_ids=[8, 9, 10, 11],
                    current_target_token_ids=[9, 10, 11, 12],
                    context_length=context_length,
                    requested_context_length=context_length,
                )
            )
    return output


def test_conf_formula_uses_current_logits_and_frozen_weights():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    result = compute_confidence_metrics(logits)[0]
    probability = torch.softmax(logits, dim=-1)
    entropy = -(probability * probability.log()).sum().item()
    h_norm = entropy / torch.log(torch.tensor(4.0)).item()
    p1 = probability.max().item()
    margin_confidence = torch.sigmoid(torch.tensor(1.0)).item()
    confidence = (
        CONF_MAKV_WEIGHTS["entropy"] * (1.0 - h_norm)
        + CONF_MAKV_WEIGHTS["margin"] * margin_confidence
        + CONF_MAKV_WEIGHTS["p1"] * p1
    )
    assert result["p1"] == pytest.approx(p1)
    assert result["margin"] == pytest.approx(1.0)
    assert result["H_norm"] == pytest.approx(h_norm)
    assert result["full_conf_risk"] == pytest.approx(1.0 - confidence)
    assert result["risk"] == pytest.approx(result["full_conf_risk"])
    assert result["vocab_size"] == 4
    with pytest.raises(ValueError, match="non-finite"):
        compute_confidence_metrics(torch.tensor([[float("nan"), 0.0]]))


def test_conf_rows_are_teacher_forced_and_bf16_control_is_zero():
    rows = _rows()
    assert len(rows) == 40
    assert all(row["teacher_forced"] for row in rows)
    assert all(not row["free_running_ground_truth"] for row in rows)
    assert all(row["risk_input"] == "current_decode_path_logits" for row in rows)
    assert all(
        row["risk_scorer_api"] == "compute_precision_risk_signal" for row in rows
    )
    assert all("qdm" not in key.lower() for row in rows for key in row)
    assert all("witness" not in key.lower() for row in rows for key in row)
    bf16 = [row for row in rows if row["precision_composition"] == "BF16"]
    assert all(row["kl_bf16_quantized"] == pytest.approx(0.0, abs=1e-7) for row in bf16)
    assert all(row["js_divergence"] == pytest.approx(0.0, abs=1e-7) for row in bf16)
    assert not any(row["top1_flip"] for row in bf16)

    with pytest.raises(ValueError, match="prefixes"):
        make_conf_teacher_forced_rows(
            sample="bad",
            prefix_token_ids=[1, 2],
            current_prefix_token_ids=[1, 3],
            precision_composition="K8V4",
            reference_logits=torch.ones(1, 3),
            current_logits=torch.ones(1, 3),
        )
    with pytest.raises(ValueError, match="alignment hashes"):
        make_conf_teacher_forced_rows(
            sample="missing-streams",
            prefix_token_ids=[1, 2],
            precision_composition="K8V4",
            reference_logits=torch.ones(1, 3),
            current_logits=torch.ones(1, 3),
        )


def test_conf_artifacts_include_blocks_summary_and_status(tmp_path):
    summary = write_conf_makv_artifacts(
        _rows(),
        tmp_path,
        run_info={"mode": "conf_makv", "environment_status": "synthetic"},
    )
    expected = {
        "per_token_conf_makv.jsonl",
        "block_conf_makv.jsonl",
        "summary.json",
        "validation_report.md",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert summary["status"] in {
        "CONF_MAKV_VALIDATED",
        "CONF_MAKV_LITE_VALIDATED",
        "INCONCLUSIVE",
        "REJECTED",
    }
    assert summary["primary_block_metric"] == "risk_p90"
    assert summary["qdm_or_witness_features_used"] is False
    assert summary["bf16_control"]["ground_truth_drift_zero"] is True
    assert summary["paired_compression_tolerance"]["transitions"]
    assert summary["run"]["mode"] == "conf_makv"

    with (tmp_path / "block_conf_makv.jsonl").open(encoding="utf-8") as handle:
        blocks = [json.loads(line) for line in handle]
    assert blocks
    assert all(block["block_size"] == 32 for block in blocks)
    assert all("risk_p90" in block for block in blocks)
