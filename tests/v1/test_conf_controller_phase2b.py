# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json

import pytest

from experiments.scoutrank_transfer.conf_controller import (
    load_aligned_oracle,
    load_aligned_scout_rank_plan,
)
from experiments.scoutrank_transfer.phase2b_controller import (
    _best_oracle_under_budget,
    solve_oracle_frontier,
)

ALIGNMENT = {
    "prefix_alignment_hash": "prefix",
    "suffix_alignment_hash": "suffix",
    "target_alignment_hash": "target",
}


def _artifacts(tmp_path):
    token_rows = []
    block_rows = []
    for precision in ("K2V2", "K4V2", "K8V4", "BF16", "MIXED"):
        drift = {
            "K2V2": 0.5,
            "K4V2": 0.2,
            "K8V4": 0.01,
            "BF16": 0.0,
            "MIXED": 0.1,
        }[precision]
        for step in range(32):
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
                "full_conf_risk_p90": 0.4,
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


def test_oracle_frontier_uses_measured_kl_without_monotone_repair(tmp_path):
    token_path, block_path = _artifacts(tmp_path)
    oracles = load_aligned_oracle(token_path, block_path)
    frontier = solve_oracle_frontier(
        oracles,
        {"K2V2": 10, "K4V2": 20, "K8V4": 30, "BF16": 40},
    )
    assert len(frontier) == 4
    selected = _best_oracle_under_budget(frontier, 30)
    assert selected is not None
    assert selected["block_counts"] == {
        "K2V2": 0,
        "K4V2": 0,
        "K8V4": 1,
        "BF16": 0,
    }
    assert selected["mean_kl"] == pytest.approx(0.01)


def test_flat_scout_rank_plan_requires_matching_hashes(tmp_path):
    token_path, block_path = _artifacts(tmp_path)
    oracles = load_aligned_oracle(token_path, block_path)
    oracle = oracles[0]
    plan_path = Path(tmp_path) / "aligned_plan.jsonl"
    plan_path.write_text(
        '{"sample_id":"synthetic-0","requested_context_length":1024,'
        '"context_length":1024,"block_id":0,"precision":"K4V2",'
        '"prefix_alignment_hash":"prefix","suffix_alignment_hash":"suffix",'
        '"target_alignment_hash":"target"}\n',
        encoding="utf-8",
    )
    plan, status = load_aligned_scout_rank_plan(plan_path, oracles)
    assert status["status"] == "available"
    assert plan == {oracle.key: "K4V2"}

    plan_path.write_text(
        plan_path.read_text(encoding="utf-8").replace("target", "wrong"),
        encoding="utf-8",
    )
    plan, status = load_aligned_scout_rank_plan(plan_path, oracles)
    assert plan is None
    assert status["status"] == "unavailable"
    assert "target_alignment_hash" in status["reason"]
