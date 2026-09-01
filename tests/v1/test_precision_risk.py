# SPDX-License-Identifier: Apache-2.0

import inspect
import math

import pytest
import torch
import lmcache.v1.storage_backend.makv.precision_risk as precision_risk

from lmcache.v1.storage_backend.makv.precision_risk import (
    CONF_MAKV_WEIGHTS,
    CONF_RISK_SEMANTICS,
    CONF_SCORER_VERSION,
    PrecisionRiskSignal,
    compute_precision_risk_signal,
)


def test_frozen_conf_formula_and_signal_contract():
    assert CONF_SCORER_VERSION == "conf_makv_v1"
    assert CONF_RISK_SEMANTICS == "uncalibrated_precision_risk"
    assert dict(CONF_MAKV_WEIGHTS) == {
        "entropy": 0.4,
        "margin": 0.3,
        "p1": 0.3,
    }
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    signal = compute_precision_risk_signal(logits, step=7)
    probability = torch.softmax(logits, dim=-1)
    entropy = -(probability * probability.log()).sum().item()
    entropy_norm = entropy / math.log(logits.numel())
    p1 = probability.max().item()
    margin = 1.0
    margin_confidence = torch.sigmoid(torch.tensor(margin)).item()
    confidence = (
        CONF_MAKV_WEIGHTS["entropy"] * (1.0 - entropy_norm)
        + CONF_MAKV_WEIGHTS["margin"] * margin_confidence
        + CONF_MAKV_WEIGHTS["p1"] * p1
    )

    assert isinstance(signal, PrecisionRiskSignal)
    assert signal.step == 7
    assert signal.scorer_version == CONF_SCORER_VERSION
    assert signal.semantics == CONF_RISK_SEMANTICS
    assert signal.valid is True
    assert signal.confidence == pytest.approx(confidence)
    assert signal.risk == pytest.approx(1.0 - confidence)
    assert signal.margin_risk == pytest.approx(1.0 - margin_confidence)
    assert signal.margin == pytest.approx(margin)
    assert signal.entropy_norm == pytest.approx(entropy_norm)
    assert signal.top1_probability == pytest.approx(p1)

    record = signal.to_dict()
    assert set(record) == {
        "step",
        "risk",
        "scorer_version",
        "semantics",
        "valid",
    }
    assert record["semantics"] == "uncalibrated_precision_risk"
    assert record["valid"] is True
    diagnostic_record = signal.to_dict(include_diagnostics=True)
    assert {
        "confidence",
        "margin_risk",
        "margin",
        "entropy_norm",
        "top1_probability",
    } <= diagnostic_record.keys()
    assert "vocab_size" not in diagnostic_record


def test_signal_consumes_only_one_current_logits_row():
    parameters = inspect.signature(compute_precision_risk_signal).parameters
    assert tuple(parameters) == ("logits", "step")
    with pytest.raises(TypeError):
        compute_precision_risk_signal(
            torch.tensor([1.0, 0.0]), reference_logits=torch.tensor([1.0, 0.0])
        )
    with pytest.raises(ValueError, match="one decode step"):
        compute_precision_risk_signal(torch.ones(2, 3))
    with pytest.raises(ValueError, match="shape"):
        compute_precision_risk_signal(torch.ones(1))


def test_production_signal_has_no_precision_or_observer_dependencies():
    parameters = inspect.signature(compute_precision_risk_signal).parameters
    assert "reference_logits" not in parameters
    assert "qdm" not in parameters
    assert "witness" not in parameters
    assert "precision" not in parameters
    assert not any(
        name in precision_risk.__dict__
        for name in ("QDMMetadata", "ScoutRankScorer", "quantize_canonical_kv")
    )


def test_signal_is_finite_bounded_and_deterministic():
    for logits in (
        torch.tensor([-1000.0, 0.0, 1000.0]),
        torch.randn(32),
        torch.zeros(8),
    ):
        first = compute_precision_risk_signal(logits)
        second = compute_precision_risk_signal(logits)
        assert first == second
        assert math.isfinite(first.risk)
        assert 0.0 <= first.risk <= 1.0
        assert math.isfinite(first.confidence)
        assert 0.0 <= first.confidence <= 1.0
        assert math.isfinite(first.margin_risk)
        assert 0.0 <= first.margin_risk <= 1.0
        assert 0.0 <= first.entropy_norm <= 1.0
        assert 0.0 <= first.top1_probability <= 1.0

    with pytest.raises(ValueError, match="non-finite"):
        compute_precision_risk_signal(torch.tensor([float("nan"), 0.0]))
    with pytest.raises(ValueError, match="non-finite"):
        compute_precision_risk_signal(torch.tensor([float("inf"), 0.0]))


def test_validation_compatibility_delegates_to_the_same_signal_formula():
    from lmcache.v1.storage_backend.makv.conf_makv import compute_confidence_metrics

    logits = torch.tensor([[3.0, 1.0, -2.0]])
    signal = compute_precision_risk_signal(logits[0])
    legacy = compute_confidence_metrics(logits)[0]
    assert legacy["risk"] == pytest.approx(signal.risk)
    assert legacy["full_conf_risk"] == pytest.approx(signal.risk)
    assert legacy["margin_only_risk"] == pytest.approx(signal.margin_risk)
    assert legacy["scorer_version"] == signal.scorer_version
    assert legacy["semantics"] == signal.semantics
    assert legacy["valid"] is True
