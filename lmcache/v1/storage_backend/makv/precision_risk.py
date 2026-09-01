# SPDX-License-Identifier: Apache-2.0

"""Production-facing CONF-MaKV precision-risk signal.

The public boundary is deliberately small: current decode logits in, one
``PrecisionRiskSignal`` out.  This module has no QDM, ScoutRank, quantizer,
cache, threshold, or precision-controller dependency.  Remote policy code is
responsible for deciding what to do with the uncalibrated signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any
import math

import torch
import torch.nn.functional as F


CONF_SCORER_VERSION = "conf_makv_v1"
CONF_RISK_SEMANTICS = "uncalibrated_precision_risk"
CONF_MAKV_WEIGHTS = MappingProxyType(
    {
        "entropy": 0.4,
        "margin": 0.3,
        "p1": 0.3,
    }
)


@dataclass(frozen=True, slots=True)
class PrecisionRiskSignal:
    """One output-side signal for one current decode step.

    The score is intentionally not a probability. Diagnostics remain available
    on the object, while default transport omits them unless requested.
    """

    step: int
    risk: float
    scorer_version: str
    semantics: str
    valid: bool
    confidence: float
    margin_risk: float
    margin: float
    entropy_norm: float
    top1_probability: float
    vocab_size: int = field(repr=False, compare=False)
    # Optional KV position metadata. It is absent from the frozen minimal
    # signal unless the caller can map the output step to a prompt position.
    token_index: int | None = field(default=None, compare=False)
    window_tokens: int | None = field(default=None, compare=False)

    def as_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        """Return the minimal transport record, optionally with diagnostics."""
        record: dict[str, Any] = {
            "step": self.step,
            "risk": self.risk,
            "scorer_version": self.scorer_version,
            "semantics": self.semantics,
            "valid": self.valid,
        }
        if include_diagnostics:
            record.update(
                {
                    "confidence": self.confidence,
                    "margin_risk": self.margin_risk,
                    "margin": self.margin,
                    "entropy_norm": self.entropy_norm,
                    "top1_probability": self.top1_probability,
                }
            )
        if self.token_index is not None:
            record["token_index"] = self.token_index
        if self.window_tokens is not None:
            record["window_tokens"] = self.window_tokens
        return record

    def to_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        """Alias for ``as_dict`` with minimal transport as the default."""
        return self.as_dict(include_diagnostics=include_diagnostics)

    def for_kv_token(
        self, token_index: int, *, window_tokens: int | None = None
    ) -> "PrecisionRiskSignal":
        """Attach an absolute prompt/KV position without changing the scorer."""
        if (
            isinstance(token_index, bool)
            or not isinstance(token_index, (int, float))
            or (isinstance(token_index, float) and not token_index.is_integer())
            or token_index < 0
        ):
            raise ValueError("token_index must be non-negative")
        if window_tokens is not None:
            if (
                isinstance(window_tokens, bool)
                or not isinstance(window_tokens, (int, float))
                or (
                    isinstance(window_tokens, float)
                    and not window_tokens.is_integer()
                )
                or window_tokens <= 0
            ):
                raise ValueError("window_tokens must be positive")
        return replace(
            self,
            token_index=int(token_index),
            window_tokens=(
                None if window_tokens is None else int(window_tokens)
            ),
        )


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(logits)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] < 2:
        raise ValueError("logits must have shape [vocab] or [1, vocab]")
    tensor = tensor.detach().float()
    if not torch.isfinite(tensor).all():
        raise ValueError("logits contains non-finite values")
    return tensor


def _score_row(
    row: torch.Tensor,
    *,
    step: int,
    token_index: int | None = None,
    window_tokens: int | None = None,
) -> PrecisionRiskSignal:
    vocab_size = int(row.shape[-1])
    log_probability = F.log_softmax(row, dim=-1)
    probability = log_probability.exp()
    top_values = torch.topk(row, k=2, dim=-1).values
    margin = top_values[0] - top_values[1]
    top1_probability = probability.max(dim=-1).values
    entropy = -(probability * log_probability).sum(dim=-1)
    entropy_norm = (entropy / math.log(vocab_size)).clamp(0.0, 1.0)
    margin_confidence = torch.sigmoid(margin)

    confidence = (
        CONF_MAKV_WEIGHTS["entropy"] * (1.0 - entropy_norm)
        + CONF_MAKV_WEIGHTS["margin"] * margin_confidence
        + CONF_MAKV_WEIGHTS["p1"] * top1_probability
    )
    risk = 1.0 - confidence
    margin_risk = 1.0 - margin_confidence

    values = (
        float(risk.item()),
        float(confidence.item()),
        float(margin_risk.item()),
        float(margin.item()),
        float(entropy_norm.item()),
        float(top1_probability.item()),
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("CONF-MaKV produced a non-finite signal")
    bounded_values = values[:3] + values[4:]
    if not all(0.0 <= value <= 1.0 for value in bounded_values):
        raise RuntimeError("CONF-MaKV produced an out-of-range risk signal")

    return PrecisionRiskSignal(
        step=int(step),
        risk=values[0],
        scorer_version=CONF_SCORER_VERSION,
        semantics=CONF_RISK_SEMANTICS,
        valid=True,
        confidence=values[1],
        margin_risk=values[2],
        margin=values[3],
        entropy_norm=values[4],
        top1_probability=values[5],
        vocab_size=vocab_size,
        token_index=token_index,
        window_tokens=window_tokens,
    )


def compute_precision_risk_signal(
    logits: torch.Tensor,
    *,
    step: int = 0,
) -> PrecisionRiskSignal:
    """Compute the frozen signal from one current decode logits row.

    ``logits`` must be the logits produced by the currently active compressed
    decode path.  No reference logits, KV witness, precision plan, or
    controller state is accepted by this API.
    """
    rows = _normalize_logits(logits)
    if rows.shape[0] != 1:
        raise ValueError("production CONF-MaKV observer expects one decode step")
    if int(step) < 0:
        raise ValueError("step must be non-negative")
    return _score_row(rows[0], step=int(step))


def _compute_precision_risk_signals(
    logits: torch.Tensor,
) -> tuple[PrecisionRiskSignal, ...]:
    """Batch helper reserved for offline validation compatibility."""
    rows = _normalize_logits(logits)
    return tuple(_score_row(row, step=index) for index, row in enumerate(rows))


__all__ = [
    "CONF_RISK_SEMANTICS",
    "CONF_SCORER_VERSION",
    "PrecisionRiskSignal",
    "compute_precision_risk_signal",
]
