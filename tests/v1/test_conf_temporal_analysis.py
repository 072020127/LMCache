# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.scoutrank_transfer.conf_temporal_analysis import (
    HIGH_STATE,
    NORMAL_STATE,
    TemporalStream,
    _assess,
    _mask_metrics,
    run_temporal_trigger,
)


def _stream() -> TemporalStream:
    rows = []
    for step, (normal_kl, high_kl) in enumerate(
        ((1.0, 0.5), (2.0, 1.0), (3.0, 4.0), (4.0, 2.0))
    ):
        rows.append(
            {
                "sample": "synthetic",
                "requested_context_length": 1024,
                "context_length": 1024,
                "step": step,
                "normal_kl": normal_kl,
                "high_kl": high_kl,
                "normal_js": normal_kl / 2.0,
                "high_js": high_kl / 2.0,
                "normal_top1_flip": step in (0, 2),
                "high_top1_flip": step == 2,
            }
        )
    return TemporalStream(key=("synthetic", 1024, 1024), rows=tuple(rows))


def test_ema_trigger_holds_and_exits_after_protection_window():
    trace = run_temporal_trigger(
        [0.1, 0.95, 0.95, 0.1, 0.1],
        trigger_type="ema",
        threshold=0.8,
        exit_ratio=0.5,
        ema_alpha=1.0,
        protection_window=2,
    )

    assert trace.states == (
        NORMAL_STATE,
        HIGH_STATE,
        HIGH_STATE,
        NORMAL_STATE,
        NORMAL_STATE,
    )
    assert [event["event_type"] for event in trace.events] == ["ENTER", "EXIT"]
    assert trace.events[0]["hold_until_index"] == 3


def test_k_of_n_requires_k_high_risks():
    trace = run_temporal_trigger(
        [0.9, 0.1, 0.9, 0.1, 0.1, 0.1],
        trigger_type="k_of_n",
        threshold=0.8,
        exit_ratio=0.5,
        k=2,
        n=3,
        protection_window=2,
    )

    assert trace.states[0:3] == (NORMAL_STATE, NORMAL_STATE, HIGH_STATE)
    assert trace.states[3] == HIGH_STATE
    assert trace.states[4] == NORMAL_STATE


def test_window_metrics_measure_signed_and_avoidable_benefit():
    stream = _stream()
    metrics = _mask_metrics(
        stream,
        (True, True, False, False),
        method="conf_trigger",
        horizon=2,
        selection_mode="single_trigger_future_window",
    )

    assert metrics["normal_kl_damage"] == pytest.approx(10.0)
    assert metrics["policy_kl_damage"] == pytest.approx(8.5)
    assert metrics["recovered_kl_damage"] == pytest.approx(1.5)
    assert metrics["avoidable_kl_damage"] == pytest.approx(3.5)
    assert metrics["recovered_avoidable_kl"] == pytest.approx(1.5)
    assert metrics["benefit_recall_kl"] == pytest.approx(1.5 / 3.5)
    assert metrics["high_precision_duty_cycle"] == pytest.approx(0.5)
    assert metrics["top1_flip_reduction"] == 1


def test_assessment_matches_random_at_nearest_duty_not_event_id():
    base = {
        "status": "success",
        "trigger_type": "ema",
        "horizon": 32,
        "high_precision_duty_cycle": 0.5,
    }
    assessment = _assess(
        [
            {
                **base,
                "method": "conf_trigger",
                "config_id": "conf",
                "recovered_kl_damage": 5.0,
            },
            {
                **base,
                "method": "random_trigger",
                "reference_config_id": "different-config",
                "recovered_kl_damage": 4.0,
            },
            {
                **base,
                "method": "margin_trigger",
                "config_id": "margin",
                "recovered_kl_damage": 6.0,
            },
        ],
        conf_config_count=1,
    )

    assert assessment["conf_vs_random_win_fraction"] == pytest.approx(1.0)
    assert assessment["conf_vs_margin_win_fraction"] == pytest.approx(0.0)
    assert assessment["mean_random_duty_gap"] == pytest.approx(0.0)
    assert assessment["status"] == "INCONCLUSIVE"
