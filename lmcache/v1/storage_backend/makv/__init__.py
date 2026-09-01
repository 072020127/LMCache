# SPDX-License-Identifier: Apache-2.0

"""MaKV remote serde support and optional output-side risk signal."""

from .precision_risk import PrecisionRiskSignal, compute_precision_risk_signal

__all__ = ["PrecisionRiskSignal", "compute_precision_risk_signal"]
