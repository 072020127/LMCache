# SPDX-License-Identifier: Apache-2.0

import math

from benchmarks.makv_precision_risk_benchmark import run_precision_risk_benchmark


def test_precision_risk_benchmark_reports_disabled_and_enabled_conditions():
    result = run_precision_risk_benchmark(
        device="cpu",
        tokens=6,
        vocab_size=8,
        warmup=2,
        seed=23,
    )

    assert result["scope"] == "observer_only_pre_generated_logits"
    assert result["disabled"]["scorer_enabled"] is False
    assert result["disabled"]["valid_signals"] == 0
    assert result["enabled"]["scorer_enabled"] is True
    assert result["enabled"]["valid_signals"] == 6
    for condition in (result["disabled"], result["enabled"]):
        assert condition["tokens"] == 6
        assert condition["tpot_ms_per_token"] >= 0.0
        assert condition["tokens_per_s"] > 0.0
        assert math.isfinite(condition["observer_iteration_mean_ms"])
        assert math.isfinite(condition["observer_iteration_p95_ms"])
    assert result["enabled"]["scorer_latency_mean_ms"] > 0.0
    assert result["enabled"]["scorer_latency_p95_ms"] >= 0.0
