# SPDX-License-Identifier: Apache-2.0

"""CPU validation of the independent MaKV precision-upgrade experiment."""

# Standard
import json
from pathlib import Path

# Third Party

# First Party
from benchmarks.makv_precision_upgrade import run_experiment, spearman


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_precision_upgrade_cpu_artifacts_and_invariants(tmp_path: Path) -> None:
    output = tmp_path / "precision-upgrade"
    result = run_experiment(
        output_dir=output,
        backend="synthetic",
        max_prompts=2,
        context_lengths=(16,),
        horizons=(4, 8),
        residual_dtypes=("float16", "float32"),
        upgrade_rates=(0.25, 0.5),
        window_tokens=16,
        random_repeats=4,
        candidate_token_limit=8,
        candidate_batch_size=2,
    )

    assert result["status"] == "success"
    expected_files = {
        "residual_fidelity.json",
        "upgrade_benefit.jsonl",
        "risk_selection.jsonl",
        "upgrade_frontier.json",
        "system_cost.json",
        "precision_upgrade_report.md",
    }
    assert {path.name for path in output.iterdir()} == expected_files

    fidelity = _json(output / "residual_fidelity.json")
    assert fidelity["status"] == "success"
    assert {row["residual_dtype"] for row in fidelity["rows"]} == {
        "float16",
        "float32",
    }
    assert all(row["future_only"] for row in fidelity["rows"])
    assert all(row["same_step_logits_excluded"] for row in fidelity["rows"])

    benefits = _jsonl(output / "upgrade_benefit.jsonl")
    assert benefits
    assert all(
        row["token_index"] == row["absolute_token_index"]
        and row["future_only"]
        and row["same_step_logits_excluded"]
        and row["teacher_forced"]
        and row["prefix_token_hash"]
        and row["suffix_input_token_hash"]
        and row["target_token_hash"]
        for row in benefits
    )
    assert {row["horizon"] for row in benefits} == {4, 8}
    assert {row["residual_dtype"] for row in benefits} == {
        "float16",
        "float32",
    }

    selections = _jsonl(output / "risk_selection.jsonl")
    selection_methods = {
        row["method"] for row in selections if row["row_type"] == "selection"
    }
    assert selection_methods == {
        "CONF_UPGRADE",
        "MARGIN_ONLY",
        "RANDOM_UPGRADE",
        "ORACLE_UPGRADE",
    }
    baselines = {row["method"] for row in selections if row["row_type"] == "baseline"}
    assert baselines == {"BF16", "DIRECT_HIGH", "LOW", "RESIDUAL_HIGH"}

    frontier = _json(output / "upgrade_frontier.json")
    grouped = {
        (
            row["residual_dtype"],
            row["horizon"],
            row["requested_upgrade_rate"],
            row["method"],
        ): row
        for row in frontier["frontier"]
    }
    for identity, row in grouped.items():
        if identity[3] != "RANDOM_UPGRADE":
            random_row = grouped[
                (identity[0], identity[1], identity[2], "RANDOM_UPGRADE")
            ]
            assert random_row["upgrade_rate"] == row["upgrade_rate"]
    assert all(
        row["method"]
        in {"CONF_UPGRADE", "MARGIN_ONLY", "RANDOM_UPGRADE", "ORACLE_UPGRADE"}
        for row in frontier["frontier"]
    )
    assert all(
        "spearman_risk_benefit" in row
        for row in frontier["frontier"]
        if row["method"] in {"CONF_UPGRADE", "MARGIN_ONLY"}
    )

    costs = _json(output / "system_cost.json")
    assert all(row["quantized_blob_bytes"] > 0 for row in costs["rows"])
    assert all(row["residual_bytes"] > 0 for row in costs["rows"])
    assert all(
        row["total_remote_bytes"] >= row["quantized_blob_bytes"]
        for row in costs["rows"]
    )
    assert all(row["risk_handler_latency_ms"] is not None for row in costs["rows"])
    invariants = costs["invariants"]
    assert invariants["status"] == "passed"
    assert invariants["canonical_hash_unchanged"]
    assert invariants["active_window_upgraded"]
    assert invariants["active_view_differs_from_canonical"]
    assert invariants["window_expired"]
    assert invariants["restored_hash_matches_before"]
    assert invariants["private_metadata_never_public"]
    assert invariants["residual_none_fail_closed"]
    assert invariants["manager_quantize_calls"] > 0


def test_tie_aware_spearman_is_deterministic() -> None:
    assert spearman((1.0, 1.0, 2.0), (1.0, 2.0, 3.0)) is not None
    assert spearman((1.0, 1.0), (2.0, 2.0)) is None
