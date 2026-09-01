# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.scoutrank_transfer.conf_controller import (
    PRECISIONS,
    build_pareto_frontier,
    evaluate_policy,
    load_aligned_oracle,
    max_protection,
    precision_from_risk,
)


ALIGNMENT = {
    "prefix_alignment_hash": "prefix",
    "suffix_alignment_hash": "suffix",
    "target_alignment_hash": "target",
}


def _artifacts(tmp_path):
    token_rows = []
    block_rows = []
    for precision in (*PRECISIONS, "MIXED"):
        for step in range(32):
            drift = {
                "K2V2": 0.5,
                "K4V2": 0.2,
                "K8V4": 0.01,
                "BF16": 0.0,
                "MIXED": 0.1,
            }[precision]
            token_rows.append(
                {
                    **ALIGNMENT,
                    "context_length": 1024,
                    "free_running_ground_truth": False,
                    "js_divergence": drift / 2,
                    "kl_divergence": drift,
                    "precision_composition": precision,
                    "protocol": "conf_makv_v1",
                    "requested_context_length": 1024,
                    "sample": "synthetic-0",
                    "step": step,
                    "target_token_id": 9,
                    "teacher_forced": True,
                    "top1_flip": precision == "K2V2" and step == 0,
                }
            )
        block_rows.append(
            {
                **ALIGNMENT,
                "block_id": 0,
                "block_size": 32,
                "context_length": 1024,
                "free_running_ground_truth": False,
                "full_conf_risk_p90": {
                    "K2V2": 0.8,
                    "K4V2": 0.6,
                    "K8V4": 0.4,
                    "BF16": 0.2,
                    "MIXED": 0.5,
                }[precision],
                "margin_only_risk_p90": 0.4,
                "margin_p1_risk_p90": 0.4,
                "precision_composition": precision,
                "protocol": "conf_makv_v1_block",
                "requested_context_length": 1024,
                "risk_max": 0.4,
                "risk_mean": 0.2,
                "risk_p90": 0.4,
                "sample": "synthetic-0",
                "step_end": 31,
                "step_start": 0,
                "teacher_forced": True,
                "token_count": 32,
            }
        )
    token_path = tmp_path / "per_token_conf_makv.jsonl"
    block_path = tmp_path / "block_conf_makv.jsonl"
    token_path.write_text(
        "".join(json.dumps(row) + "\n" for row in token_rows), encoding="utf-8"
    )
    block_path.write_text(
        "".join(json.dumps(row) + "\n" for row in block_rows), encoding="utf-8"
    )
    return token_path, block_path


def test_controller_enforces_precision_order_and_protection_floor():
    assert precision_from_risk(0.1, (0.2, 0.4, 0.6)) == "K2V2"
    assert precision_from_risk(0.4, (0.2, 0.4, 0.6)) == "K8V4"
    assert precision_from_risk(0.8, (0.2, 0.4, 0.6)) == "BF16"
    assert max_protection("K2V2", "K8V4") == "K8V4"
    assert max_protection("BF16", "K2V2") == "BF16"
    with pytest.raises(ValueError, match="T1 < T2 < T3"):
        precision_from_risk(0.3, (0.4, 0.2, 0.6))


def test_aligned_replay_uses_homogeneous_oracle_and_zeroes_bf16_noise(tmp_path):
    token_path, block_path = _artifacts(tmp_path)
    oracles = load_aligned_oracle(token_path, block_path)
    assert len(oracles) == 1
    oracle = oracles[0]
    assignment = {oracle.key: "K2V2"}
    result = evaluate_policy(
        name="fixed_K2V2",
        oracles=oracles,
        assignment=assignment,
        costs={"K2V2": 10, "K4V2": 20, "K8V4": 30, "BF16": 40},
    )
    assert result["estimated_kv_bytes"] == 10
    assert result["block_counts"] == {
        "K2V2": 1,
        "K4V2": 0,
        "K8V4": 0,
        "BF16": 0,
    }
    assert result["mean_kl"] == pytest.approx(0.5)
    assert result["top1_flip_rate"] == pytest.approx(1 / 32)


def test_replay_fails_closed_on_token_alignment_mismatch(tmp_path):
    token_path, block_path = _artifacts(tmp_path)
    rows = [json.loads(line) for line in token_path.read_text().splitlines()]
    rows[1]["prefix_alignment_hash"] = "mismatch"
    token_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="alignment mismatch"):
        load_aligned_oracle(token_path, block_path)


def test_frontier_excludes_unavailable_records():
    records = [
        {
            "status": "unavailable",
            "strategy": "scoutrank_only",
        },
        {
            "status": "success",
            "strategy": "fixed_K2V2",
            "estimated_kv_bytes": 10,
            "mean_kl": 1.0,
            "p95_kl": 1.0,
            "mean_js": 0.5,
            "top1_flip_rate": 0.5,
        },
        {
            "status": "success",
            "strategy": "fixed_BF16",
            "estimated_kv_bytes": 40,
            "mean_kl": 0.0,
            "p95_kl": 0.0,
            "mean_js": 0.0,
            "top1_flip_rate": 0.0,
        },
    ]
    assert {row["strategy"] for row in build_pareto_frontier(records)} == {
        "fixed_K2V2",
        "fixed_BF16",
    }
