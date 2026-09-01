# SPDX-License-Identifier: Apache-2.0

"""CONF-MaKV v1 output-side compression-tolerance scorer.

This module is intentionally independent of QDM metadata and KV witnesses.
The production-facing score consumes only logits from the current decode path.
All correlation, quantile, and paired-precision analysis is validation-only;
the score is not an error probability and has no calibrated threshold.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import torch
import torch.nn.functional as F

from .precision_risk import (
    CONF_MAKV_WEIGHTS,
    CONF_RISK_SEMANTICS,
    CONF_SCORER_VERSION,
    PrecisionRiskSignal,
    compute_precision_risk_signal,
)

CONF_MAKV_VERSION = CONF_SCORER_VERSION
CONF_MAKV_BLOCK_SIZE = 32
CONF_FEATURES = ("margin_only_risk", "margin_p1_risk", "full_conf_risk")
PRECISION_COMPOSITIONS = ("K2V2", "K4V2", "K8V4", "BF16", "MIXED")
PAIRED_TRANSITIONS = (
    ("K8V4_to_K4V2", "K8V4", "K4V2"),
    ("K4V2_to_K2V2", "K4V2", "K2V2"),
)
ALIGNMENT_FIELDS = (
    "prefix_alignment_hash",
    "suffix_alignment_hash",
    "target_alignment_hash",
)


def _as_rows(logits: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(logits)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] < 2:
        raise ValueError(f"{name} must have shape [steps, vocab] or [vocab]")
    tensor = tensor.detach().float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite logits")
    return tensor


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps([int(token) for token in token_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compute_confidence_metrics(logits: torch.Tensor) -> list[dict[str, Any]]:
    """Return legacy validation rows from the production signal.

    This is a validation compatibility helper, not the production observer
    surface.  It preserves the historical ablation fields used by the
    offline artifacts while delegating the formula to ``precision_risk``.
    """
    # Keep validation provenance on the public production scorer boundary.
    # This makes it explicit that every precision row is scored from its own
    # current-path logits, rather than from a reference or a legacy helper.
    signals = tuple(
        compute_precision_risk_signal(row, step=index)
        for index, row in enumerate(logits)
    )
    output = []
    for signal in signals:
        legacy = signal.as_dict(include_diagnostics=True)
        legacy.pop("step", None)
        entropy = signal.entropy_norm * math.log(signal.vocab_size)
        legacy.update(
            {
                "p1": signal.top1_probability,
                "margin": signal.margin,
                "H_norm": signal.entropy_norm,
                "entropy_norm": signal.entropy_norm,
                "entropy": entropy,
                "confidence": signal.confidence,
                "risk": signal.risk,
                "margin_risk": signal.margin_risk,
                "margin_only_risk": signal.margin_risk,
                "margin_p1_risk": 0.5
                * (signal.margin_risk + (1.0 - signal.top1_probability)),
                "full_conf_risk": signal.risk,
                "vocab_size": signal.vocab_size,
                "scorer_version": signal.scorer_version,
                "semantics": signal.semantics,
                "risk_scorer_api": "compute_precision_risk_signal",
            }
        )
        output.append(legacy)
    return output


def _teacher_forced_ground_truth(
    reference_logits: torch.Tensor,
    current_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    """Compute validation-only drift labels from aligned logits."""
    reference = _as_rows(reference_logits, "reference_logits")
    current = _as_rows(current_logits, "current_logits")
    if reference.shape != current.shape:
        raise ValueError("reference and current logits must have equal shapes")
    reference_log_probability = F.log_softmax(reference, dim=-1)
    current_log_probability = F.log_softmax(current, dim=-1)
    reference_probability = reference_log_probability.exp()
    current_probability = current_log_probability.exp()
    kl = (
        reference_probability
        * (reference_log_probability - current_log_probability)
    ).sum(dim=-1)
    mixture = 0.5 * (reference_probability + current_probability)
    mixture_log_probability = torch.log(
        mixture.clamp_min(torch.finfo(torch.float32).tiny)
    )
    js = 0.5 * (
        (
            reference_probability
            * (reference_log_probability - mixture_log_probability)
        ).sum(dim=-1)
        + (
            current_probability * (current_log_probability - mixture_log_probability)
        ).sum(dim=-1)
    )
    reference_top = torch.topk(reference, k=2, dim=-1)
    current_top = torch.topk(current, k=2, dim=-1)
    reference_top1 = reference_top.indices[:, 0]
    current_top1 = current_top.indices[:, 0]
    reference_margin = reference_top.values[:, 0] - reference_top.values[:, 1]
    current_margin = current_top.values[:, 0] - current_top.values[:, 1]
    return [
        {
            "kl_bf16_quantized": float(kl[index].item()),
            "kl_divergence": float(kl[index].item()),
            "js_divergence": float(js[index].item()),
            "top1_flip": bool(reference_top1[index] != current_top1[index]),
            "reference_top1_token": int(reference_top1[index].item()),
            "current_top1_token": int(current_top1[index].item()),
            "reference_top1_top2_margin": float(reference_margin[index].item()),
            "current_top1_top2_margin": float(current_margin[index].item()),
        }
        for index in range(reference.shape[0])
    ]


def assert_conf_teacher_forced_alignment(
    reference_prefix_token_ids: Sequence[int],
    current_prefix_token_ids: Sequence[int],
    *,
    reference_suffix_input_ids: Sequence[int] | None = None,
    current_suffix_input_ids: Sequence[int] | None = None,
    reference_target_token_ids: Sequence[int] | None = None,
    current_target_token_ids: Sequence[int] | None = None,
) -> dict[str, str | None]:
    """Fail closed unless all teacher-forced token streams are identical."""
    prefix = [int(value) for value in reference_prefix_token_ids]
    current_prefix = [int(value) for value in current_prefix_token_ids]
    if prefix != current_prefix:
        raise ValueError("CONF-MaKV teacher-forced prefixes are not identical")

    def aligned_hash(
        reference: Sequence[int] | None,
        current: Sequence[int] | None,
        name: str,
    ) -> str | None:
        if (reference is None) != (current is None):
            raise ValueError(f"both CONF-MaKV {name} sequences are required")
        if reference is None:
            return None
        left = [int(value) for value in reference]
        right = [int(value) for value in current or []]
        if left != right:
            raise ValueError(f"CONF-MaKV {name} sequences are not identical")
        return _token_hash(left)

    return {
        "prefix_token_id_hash": _token_hash(prefix),
        "suffix_input_token_id_hash": aligned_hash(
            reference_suffix_input_ids, current_suffix_input_ids, "suffix"
        ),
        "target_token_id_hash": aligned_hash(
            reference_target_token_ids, current_target_token_ids, "target"
        ),
    }


def make_conf_teacher_forced_rows(
    *,
    sample: str,
    prefix_token_ids: Sequence[int],
    precision_composition: str,
    reference_logits: torch.Tensor,
    current_logits: torch.Tensor,
    top_k: int = 50,
    current_prefix_token_ids: Sequence[int] | None = None,
    suffix_input_ids: Sequence[int] | None = None,
    target_token_ids: Sequence[int] | None = None,
    current_suffix_input_ids: Sequence[int] | None = None,
    current_target_token_ids: Sequence[int] | None = None,
    context_length: int | None = None,
    requested_context_length: int | None = None,
    row_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build aligned per-token CONF rows without QDM/witness inputs."""
    del top_k  # CONF-MaKV uses full-vocabulary entropy, not top-K entropy.
    if precision_composition not in PRECISION_COMPOSITIONS:
        raise ValueError(f"unknown precision composition: {precision_composition}")
    reference = _as_rows(reference_logits, "reference_logits")
    current = _as_rows(current_logits, "current_logits")
    if reference.shape != current.shape:
        raise ValueError("reference and current logits must have equal shapes")
    if suffix_input_ids is not None and len(suffix_input_ids) != reference.shape[0]:
        raise ValueError("suffix_input_ids length does not match decode steps")
    if target_token_ids is not None and len(target_token_ids) != reference.shape[0]:
        raise ValueError("target_token_ids length does not match decode steps")
    prefix_ids = [int(value) for value in prefix_token_ids]
    current_prefix_ids = (
        prefix_ids
        if current_prefix_token_ids is None
        else [int(value) for value in current_prefix_token_ids]
    )
    hashes = assert_conf_teacher_forced_alignment(
        prefix_ids,
        current_prefix_ids,
        reference_suffix_input_ids=suffix_input_ids,
        current_suffix_input_ids=(
            suffix_input_ids
            if current_suffix_input_ids is None and suffix_input_ids is not None
            else current_suffix_input_ids
        ),
        reference_target_token_ids=target_token_ids,
        current_target_token_ids=(
            target_token_ids
            if current_target_token_ids is None and target_token_ids is not None
            else current_target_token_ids
        ),
    )
    if any(hashes[field] is None for field in hashes):
        raise ValueError(
            "CONF-MaKV requires explicit prefix, suffix, and target alignment hashes"
        )
    score_rows = compute_confidence_metrics(current)
    ground_truth = _teacher_forced_ground_truth(reference, current)
    rows = []
    for step, (score, truth) in enumerate(zip(score_rows, ground_truth, strict=True)):
        row: dict[str, Any] = {
            "protocol": CONF_MAKV_VERSION,
            "sample": str(sample),
            "step": step,
            "precision_composition": precision_composition,
            "prefix_length": len(prefix_ids),
            "teacher_forced": True,
            "free_running_ground_truth": False,
            "prefix_aligned": True,
            "suffix_aligned": True,
            "target_aligned": True,
            "prefix_token_id_hash": hashes["prefix_token_id_hash"],
            "suffix_input_token_id_hash": hashes["suffix_input_token_id_hash"],
            "target_token_id_hash": hashes["target_token_id_hash"],
            "prefix_alignment_hash": hashes["prefix_token_id_hash"],
            "suffix_alignment_hash": hashes["suffix_input_token_id_hash"],
            "target_alignment_hash": hashes["target_token_id_hash"],
            "risk_input": "current_decode_path_logits",
            "risk_calibration": "uncalibrated_score",
            **score,
            "top1_margin": score["margin"],
            "top1_top2_margin": score["margin"],
            **truth,
        }
        if context_length is not None:
            row["context_length"] = int(context_length)
        if requested_context_length is not None:
            row["requested_context_length"] = int(requested_context_length)
        if suffix_input_ids is not None:
            row["suffix_input_token_id"] = int(suffix_input_ids[step])
        if target_token_ids is not None:
            row["target_token_id"] = int(target_token_ids[step])
        if row_metadata is not None:
            protected = set(row)
            conflicting = protected.intersection(row_metadata)
            if conflicting:
                raise ValueError(
                    "row_metadata cannot overwrite protected CONF-MaKV fields: "
                    + ", ".join(sorted(conflicting))
                )
            row.update(dict(row_metadata))
        rows.append(row)
    return rows


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample"),
        row.get("context_length"),
        row.get("requested_context_length"),
        row.get("precision_composition"),
    )


def aggregate_conf_blocks(
    rows: Iterable[Mapping[str, Any]],
    *,
    block_size: int = CONF_MAKV_BLOCK_SIZE,
) -> list[dict[str, Any]]:
    """Aggregate token scores into 32-token diagnostic blocks."""
    if block_size <= 0:
        raise ValueError("CONF-MaKV block_size must be positive")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        if not row.get("teacher_forced") or not all(
            row.get(field, False)
            for field in ("prefix_aligned", "suffix_aligned", "target_aligned")
        ):
            raise ValueError("CONF-MaKV block aggregation requires aligned rows")
        if any(row.get(field) is None for field in ALIGNMENT_FIELDS):
            raise ValueError("CONF-MaKV block aggregation requires alignment hashes")
        groups[_group_key(row) + (int(row["step"]) // block_size,)].append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        group.sort(key=lambda row: int(row["step"]))
        alignment = {
            field: group[0].get(field)
            for field in ALIGNMENT_FIELDS
        }
        for field in ALIGNMENT_FIELDS:
            if any(row.get(field) != alignment[field] for row in group):
                raise ValueError(f"CONF-MaKV block has inconsistent {field}")
        block_id = int(key[-1])
        output.append(
            {
                "protocol": f"{CONF_MAKV_VERSION}_block",
                "sample": key[0],
                "context_length": key[1],
                "requested_context_length": key[2],
                "precision_composition": key[3],
                "block_id": block_id,
                "block_size": int(block_size),
                "step_start": int(group[0]["step"]),
                "step_end": int(group[-1]["step"]),
                "token_count": len(group),
                "risk_p90": _percentile(
                    [float(row["full_conf_risk"]) for row in group], 0.90
                ),
                "risk_max": max(float(row["full_conf_risk"]) for row in group),
                "risk_mean": sum(float(row["full_conf_risk"]) for row in group)
                / len(group),
                "margin_only_risk_p90": _percentile(
                    [float(row["margin_only_risk"]) for row in group], 0.90
                ),
                "margin_p1_risk_p90": _percentile(
                    [float(row["margin_p1_risk"]) for row in group], 0.90
                ),
                "full_conf_risk_p90": _percentile(
                    [float(row["full_conf_risk"]) for row in group], 0.90
                ),
                **alignment,
                "teacher_forced": True,
                "free_running_ground_truth": False,
            }
        )
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator.item()) == 0.0:
        return None
    return float((x * y).sum().div(denominator).item())


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(
        enumerate(float(value) for value in values),
        key=lambda item: item[1],
    )
    result = [0.0] * len(indexed)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = 0.5 * (index + 1 + end)
        for position in range(index, end):
            result[indexed[position][0]] = rank
        index = end
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right)) if len(left) >= 2 else None


def _roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [
        float(score)
        for score, label in zip(scores, labels, strict=True)
        if label
    ]
    negatives = [
        float(score)
        for score, label in zip(scores, labels, strict=True)
        if not label
    ]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _pr_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positive_count = sum(bool(label) for label in labels)
    if positive_count == 0 or positive_count == len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    true_positives = 0
    previous_recall = 0.0
    area = 0.0
    for rank, index in enumerate(order, 1):
        true_positives += int(labels[index])
        recall = true_positives / positive_count
        precision = true_positives / rank
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _feature_stats(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    scores = [float(row[feature]) for row in rows]
    kl = [float(row["kl_bf16_quantized"]) for row in rows]
    js = [float(row["js_divergence"]) for row in rows]
    labels = [bool(row["top1_flip"]) for row in rows]
    threshold = _percentile(scores, 0.90)
    high = [
        row
        for row in rows
        if threshold is not None and float(row[feature]) >= threshold
    ]
    all_flip_rate = sum(labels) / len(labels) if labels else None
    high_flip_rate = (
        sum(bool(row["top1_flip"]) for row in high) / len(high) if high else None
    )
    return {
        "count": len(rows),
        "spearman_vs_kl": _spearman(scores, kl),
        "spearman_vs_js": _spearman(scores, js),
        "auroc_top1_flip": _roc_auc(scores, labels),
        "pr_auc_top1_flip": _pr_auc(scores, labels),
        "top1_flip_rate": all_flip_rate,
        "mean_kl": sum(kl) / len(kl) if kl else None,
        "mean_js": sum(js) / len(js) if js else None,
        "high_risk_quantile": 0.90,
        "high_risk_threshold": threshold,
        "high_risk_count": len(high),
        "high_risk_kl_mean": (
            sum(float(row["kl_bf16_quantized"]) for row in high) / len(high)
            if high
            else None
        ),
        "high_risk_js_mean": (
            sum(float(row["js_divergence"]) for row in high) / len(high)
            if high
            else None
        ),
        "high_risk_flip_rate": high_flip_rate,
        "high_risk_flip_enrichment": (
            high_flip_rate / all_flip_rate
            if high_flip_rate is not None and all_flip_rate not in (None, 0.0)
            else None
        ),
    }


def _group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "risk_features": {
            feature: _feature_stats(rows, feature) for feature in CONF_FEATURES
        },
    }


def _paired_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample"),
        row.get("context_length"),
        row.get("requested_context_length"),
        int(row["step"]),
    )


def _validate_paired_group(group: Sequence[Mapping[str, Any]]) -> None:
    if not group:
        raise ValueError("empty CONF-MaKV paired group")
    for field in ALIGNMENT_FIELDS:
        values = {row.get(field) for row in group}
        if len(values) != 1 or None in values:
            raise ValueError(f"CONF-MaKV paired group has inconsistent {field}")


def _paired_records(
    rows: Sequence[Mapping[str, Any]],
    high_precision: str,
    low_precision: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = _paired_key(row)
        precision = str(row["precision_composition"])
        if precision in grouped[key]:
            raise ValueError(f"duplicate CONF-MaKV paired token: {key}/{precision}")
        grouped[key][precision] = row
    records = []
    missing = 0
    for key, by_precision in grouped.items():
        if high_precision not in by_precision or low_precision not in by_precision:
            missing += 1
            continue
        pair = [by_precision[high_precision], by_precision[low_precision]]
        _validate_paired_group(pair)
        high = by_precision[high_precision]
        low = by_precision[low_precision]
        record: dict[str, Any] = {
            "sample": key[0],
            "context_length": key[1],
            "requested_context_length": key[2],
            "step": key[3],
            "precision_high": high_precision,
            "precision_low": low_precision,
            "delta_kl": float(low["kl_bf16_quantized"])
            - float(high["kl_bf16_quantized"]),
            "delta_js": float(low["js_divergence"]) - float(high["js_divergence"]),
            "delta_top1_flip": int(bool(low["top1_flip"]))
            - int(bool(high["top1_flip"])),
            "prefix_alignment_hash": high["prefix_alignment_hash"],
            "suffix_alignment_hash": high["suffix_alignment_hash"],
            "target_alignment_hash": high["target_alignment_hash"],
        }
        for feature in CONF_FEATURES:
            record[f"{feature}_high"] = float(high[feature])
            record[f"{feature}_low"] = float(low[feature])
            record[f"{feature}_delta"] = float(low[feature]) - float(high[feature])
        records.append(record)
    return records, {
        "candidate_token_count": len(grouped),
        "missing_pair_count": missing,
    }


def _paired_relation_stats(
    records: Sequence[Mapping[str, Any]], feature: str, score_view: str = "high"
) -> dict[str, Any]:
    if score_view not in ("high", "low", "delta"):
        raise ValueError(f"unknown paired score view: {score_view}")
    score = [float(row[f"{feature}_{score_view}"]) for row in records]
    delta_kl = [float(row["delta_kl"]) for row in records]
    delta_js = [float(row["delta_js"]) for row in records]
    delta_flip = [float(row["delta_top1_flip"]) for row in records]
    return {
        "count": len(records),
        "spearman_vs_delta_kl": _spearman(score, delta_kl),
        "spearman_vs_delta_js": _spearman(score, delta_js),
        "spearman_vs_delta_top1_flip": _spearman(score, delta_flip),
        "pearson_vs_delta_kl": _pearson(score, delta_kl),
        "pearson_vs_delta_js": _pearson(score, delta_js),
        "mean_delta_kl": sum(delta_kl) / len(delta_kl) if delta_kl else None,
        "mean_delta_js": sum(delta_js) / len(delta_js) if delta_js else None,
        "positive_delta_kl_fraction": (
            sum(value > 0.0 for value in delta_kl) / len(delta_kl)
            if delta_kl
            else None
        ),
    }


def _group_paired_records(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        output[str(record.get(field))].append(record)
    return dict(output)


def _paired_transition_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score_views = ("high", "low", "delta")
    return {
        "count": len(records),
        "overall": {
            feature: _paired_relation_stats(records, feature)
            for feature in CONF_FEATURES
        },
        "overall_by_score_view": {
            feature: {
                view: _paired_relation_stats(records, feature, view)
                for view in score_views
            }
            for feature in CONF_FEATURES
        },
        "by_prompt": {
            prompt: {
                feature: _paired_relation_stats(group, feature)
                for feature in CONF_FEATURES
            }
            for prompt, group in _group_paired_records(records, "sample").items()
        },
        "by_prompt_by_score_view": {
            prompt: {
                feature: {
                    view: _paired_relation_stats(group, feature, view)
                    for view in score_views
                }
                for feature in CONF_FEATURES
            }
            for prompt, group in _group_paired_records(records, "sample").items()
        },
        "by_context": {
            context: {
                feature: _paired_relation_stats(group, feature)
                for feature in CONF_FEATURES
            }
            for context, group in _group_paired_records(
                records, "context_length"
            ).items()
        },
        "by_context_by_score_view": {
            context: {
                feature: {
                    view: _paired_relation_stats(group, feature, view)
                    for view in score_views
                }
                for feature in CONF_FEATURES
            }
            for context, group in _group_paired_records(
                records, "context_length"
            ).items()
        },
    }


def _status_from_evidence(
    *,
    normalized: Sequence[Mapping[str, Any]],
    precision_buckets: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Any],
    bf16_control: Mapping[str, Any],
    prompt_count: int,
    context_count: int,
) -> tuple[str, dict[str, Any]]:
    quantized = [
        row for row in normalized if row.get("precision_composition") != "BF16"
    ]
    required_precisions = ("K8V4", "K4V2", "K2V2", "MIXED")
    supported_precisions = [
        precision
        for precision in required_precisions
        if int(precision_buckets[precision]["count"]) >= 4
    ]
    transition_support = {
        name: int(value.get("count", 0))
        for name, value in paired.get("transitions", {}).items()
    }
    supported_transitions = [
        name for name, count in transition_support.items() if count >= 4
    ]
    evidence: dict[str, Any] = {
        "quantized_row_count": len(quantized),
        "supported_precision_count": len(supported_precisions),
        "supported_precisions": supported_precisions,
        "required_precisions": list(required_precisions),
        "supported_transitions": supported_transitions,
        "prompt_count": int(prompt_count),
        "context_count": int(context_count),
        "bf16_control_valid": bool(
            bf16_control.get("ground_truth_drift_zero")
            and bf16_control.get("count", 0) >= 4
        ),
        "minimum_rows_per_precision": 4,
        "minimum_paired_rows_per_transition": 4,
    }
    if not evidence["bf16_control_valid"]:
        evidence["reason"] = "BF16 control failed; validation is fail-closed"
        return "INCONCLUSIVE", evidence
    if (
        len(quantized) < 16
        or len(supported_precisions) < len(required_precisions)
        or len(supported_transitions) < 2
        or prompt_count < 2
        or context_count < 2
    ):
        evidence["reason"] = (
            "insufficient multi-prompt, multi-context precision/pair support"
        )
        return "INCONCLUSIVE", evidence

    comparison_wins = {feature: 0 for feature in CONF_FEATURES}
    comparison_total = 0
    positive_groups = {feature: 0 for feature in CONF_FEATURES}
    for precision in supported_precisions:
        feature_stats = precision_buckets[precision]["risk_features"]
        for metric in (
            "spearman_vs_kl",
            "spearman_vs_js",
            "auroc_top1_flip",
            "pr_auc_top1_flip",
        ):
            values = {
                feature: feature_stats[feature].get(metric)
                for feature in CONF_FEATURES
            }
            finite = {
                feature: value
                for feature, value in values.items()
                if value is not None
            }
            if not finite:
                continue
            comparison_total += 1
            best = max(finite.values())
            for feature, value in finite.items():
                if value >= best:
                    comparison_wins[feature] += 1
            positive_groups.update(
                {
                    feature: positive_groups[feature]
                    + int(
                        any(
                            value is not None and value > 0.0
                            for value in (
                                feature_stats[feature].get("spearman_vs_kl"),
                                feature_stats[feature].get("spearman_vs_js"),
                            )
                        )
                    )
                    for feature in CONF_FEATURES
                }
            )

    paired_positive = {feature: True for feature in CONF_FEATURES}
    for transition in supported_transitions:
        overall = paired["transitions"][transition]["overall"]
        for feature in CONF_FEATURES:
            relation = overall[feature]
            paired_positive[feature] = paired_positive[feature] and any(
                relation.get(metric) is not None and relation[metric] > 0.0
                for metric in ("spearman_vs_delta_kl", "spearman_vs_delta_js")
            )
    evidence.update(
        {
            "comparison_total": comparison_total,
            "comparison_wins": comparison_wins,
            "positive_precision_groups": positive_groups,
            "paired_positive": paired_positive,
        }
    )
    if (
        comparison_total > 0
        and paired_positive["full_conf_risk"]
        and comparison_wins["full_conf_risk"]
        >= max(1, comparison_wins["margin_only_risk"])
        and comparison_wins["full_conf_risk"]
        >= max(1, comparison_wins["margin_p1_risk"])
    ):
        evidence["reason"] = "full CONF score has paired and within-precision evidence"
        return "CONF_MAKV_VALIDATED", evidence
    if any(paired_positive.values()) and any(
        value > 0 for value in positive_groups.values()
    ):
        evidence["reason"] = "a lightweight confidence ablation has partial evidence"
        return "CONF_MAKV_LITE_VALIDATED", evidence
    evidence["reason"] = "supported data shows no stable confidence signal"
    return "REJECTED", evidence


def summarize_conf_makv_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    block_size: int = CONF_MAKV_BLOCK_SIZE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return validation summary and block rows for aligned CONF records."""
    normalized = [dict(row) for row in rows]
    for row in normalized:
        if row.get("protocol") != CONF_MAKV_VERSION:
            raise ValueError("CONF-MaKV summary received a non-CONF row")
        if not row.get("teacher_forced") or not all(
            row.get(field, False)
            for field in ("prefix_aligned", "suffix_aligned", "target_aligned")
        ):
            raise ValueError(
                "CONF-MaKV summary requires strict teacher-forced alignment"
            )
        if any(row.get(field) is None for field in ALIGNMENT_FIELDS):
            raise ValueError("CONF-MaKV summary requires all alignment hashes")
        if row.get("risk_input") != "current_decode_path_logits":
            raise ValueError("CONF-MaKV risk input must be current-path logits")
    groups: dict[str, list[dict[str, Any]]] = {
        precision: [] for precision in PRECISION_COMPOSITIONS
    }
    for row in normalized:
        precision = str(row.get("precision_composition"))
        if precision not in groups:
            raise ValueError(f"unknown precision composition: {precision}")
        groups[precision].append(row)
    contexts = sorted(
        {
            row.get("context_length")
            for row in normalized
            if row.get("context_length") is not None
        }
    )
    prompts = sorted({str(row.get("sample")) for row in normalized})
    precision_buckets = {
        precision: _group_stats(group) for precision, group in groups.items()
    }
    context_buckets = {
        str(context): _group_stats(
            [row for row in normalized if row.get("context_length") == context]
        )
        for context in contexts
    }
    prompt_buckets = {
        prompt: _group_stats(
            [row for row in normalized if str(row.get("sample")) == prompt]
        )
        for prompt in prompts
    }
    paired_transitions: dict[str, Any] = {}
    paired_integrity: dict[str, Any] = {}
    for name, high, low in PAIRED_TRANSITIONS:
        records, integrity = _paired_records(normalized, high, low)
        paired_transitions[name] = _paired_transition_summary(records)
        paired_integrity[name] = integrity
    paired = {
        "transitions": paired_transitions,
        "integrity": paired_integrity,
        "same_token_key": [
            "sample",
            "context_length",
            "requested_context_length",
            "step",
        ],
    }
    bf16_rows = groups["BF16"]
    bf16_kl_zero = all(
        abs(float(row["kl_bf16_quantized"])) <= 1.0e-7 for row in bf16_rows
    )
    bf16_js_zero = all(
        abs(float(row["js_divergence"])) <= 1.0e-7 for row in bf16_rows
    )
    bf16_no_flip = not any(bool(row["top1_flip"]) for row in bf16_rows)
    summary: dict[str, Any] = {
        "conf_makv_version": CONF_MAKV_VERSION,
        "block_size": int(block_size),
        "weights": dict(CONF_MAKV_WEIGHTS),
        "primary_block_metric": "risk_p90",
        "score_semantics": CONF_RISK_SEMANTICS,
        "risk_input": "current_decode_path_logits_only",
        "qdm_or_witness_features_used": False,
        "overall": _group_stats(normalized),
        "quantized_overall": _group_stats(
            [row for row in normalized if row.get("precision_composition") != "BF16"]
        ),
        "precision_buckets": precision_buckets,
        "context_buckets": context_buckets,
        "prompt_buckets": prompt_buckets,
        "paired_compression_tolerance": paired,
        "bf16_control": {
            "count": len(bf16_rows),
            "kl_zero": bf16_kl_zero,
            "js_zero": bf16_js_zero,
            "top1_flip_zero": bf16_no_flip,
            "ground_truth_drift_zero": bf16_kl_zero and bf16_js_zero and bf16_no_flip,
            "risk_is_not_expected_to_be_zero": True,
        },
    }
    status, evidence = _status_from_evidence(
        normalized=normalized,
        precision_buckets=precision_buckets,
        paired=paired,
        bf16_control=summary["bf16_control"],
        prompt_count=len(prompts),
        context_count=len(contexts),
    )
    summary["status"] = status
    summary["validation"] = {"status": status, "evidence": evidence}
    return summary, aggregate_conf_blocks(normalized, block_size=block_size)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validation_report(summary: Mapping[str, Any]) -> str:
    def metric(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    validation = summary.get("validation", {})
    status = validation.get("status", "INCONCLUSIVE")
    evidence = validation.get("evidence", {})
    control = summary.get("bf16_control", {})
    precision_buckets = summary.get("precision_buckets", {})
    paired = summary.get("paired_compression_tolerance", {})
    lines = [
        "# CONF-MaKV v1 Validation",
        "",
        f"Status: **{status}**",
        "",
        "CONF-MaKV is an uncalibrated output-confidence score, not an error "
        "probability. Every quantized risk row uses only its current decode "
        "path logits; BF16 is retained as a control/reference label path and "
        "never feeds a quantized production risk score. CONF-MaKV is not a "
        "block importance score or precision controller.",
        "",
        "## Frozen Formula",
        "",
        "`confidence = 0.4*(1-H_norm) + 0.3*sigmoid(margin) + 0.3*p1`",
        "",
        f"Primary block metric: `{summary.get('primary_block_metric', 'risk_p90')}`.",
        "Empirical quantiles are analysis-only and are not production thresholds.",
        "",
        "## Evidence",
        "",
        f"- Quantized token rows: `{evidence.get('quantized_row_count', 0)}`",
        f"- Supported precision groups: `{evidence.get('supported_precisions', [])}`",
        "- Supported paired transitions: "
        f"`{evidence.get('supported_transitions', [])}`",
        f"- Decision: {evidence.get('reason', 'insufficient evidence')}",
        "",
        "## Within-Precision Metrics",
        "",
        "Scores are evaluated on the current precision path. BF16 is a control, "
        "not a quantized degradation bucket.",
        "",
        "| precision | scorer | Spearman KL | Spearman JS | AUROC flip | "
        "PR-AUC flip | flip enrichment |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for precision in ("K8V4", "K4V2", "K2V2", "MIXED"):
        bucket = precision_buckets.get(precision, {})
        for feature in CONF_FEATURES:
            stats = bucket.get("risk_features", {}).get(feature, {})
            lines.append(
                "| "
                f"{precision} | {feature} | {metric(stats.get('spearman_vs_kl'))} | "
                f"{metric(stats.get('spearman_vs_js'))} | "
                f"{metric(stats.get('auroc_top1_flip'))} | "
                f"{metric(stats.get('pr_auc_top1_flip'))} | "
                f"{metric(stats.get('high_risk_flip_enrichment'))} |"
            )
    lines.extend(
        [
            "",
            "## Paired Compression Tolerance",
            "",
            "The primary paired view uses the higher-precision current-path score "
            "to predict low-minus-high degradation for the same token.",
            "",
            "| transition | scorer | Spearman delta KL | Spearman delta JS | "
            "positive delta KL |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for transition, data in paired.get("transitions", {}).items():
        for feature in CONF_FEATURES:
            stats = data.get("overall", {}).get(feature, {})
            lines.append(
                "| "
                f"{transition} | {feature} | "
                f"{metric(stats.get('spearman_vs_delta_kl'))} | "
                f"{metric(stats.get('spearman_vs_delta_js'))} | "
                f"{metric(stats.get('positive_delta_kl_fraction'))} |"
            )
    lines.extend(
        [
            "",
            "## Context Strata",
            "",
            "| transition | context | scorer | Spearman delta KL | Spearman delta JS |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for transition, data in paired.get("transitions", {}).items():
        for context, features in sorted(data.get("by_context", {}).items()):
            for feature in CONF_FEATURES:
                stats = features.get(feature, {})
                lines.append(
                    "| "
                    f"{transition} | {context} | {feature} | "
                    f"{metric(stats.get('spearman_vs_delta_kl'))} | "
                    f"{metric(stats.get('spearman_vs_delta_js'))} |"
                )
    lines.extend(
        [
            "",
            "## Prompt Strata",
            "",
            "| transition | prompt | scorer | Spearman delta KL | Spearman delta JS |",
            "|---|---|---|---:|---:|",
        ]
    )
    for transition, data in paired.get("transitions", {}).items():
        for prompt, features in sorted(data.get("by_prompt", {}).items()):
            for feature in CONF_FEATURES:
                stats = features.get(feature, {})
                lines.append(
                    "| "
                    f"{transition} | {prompt} | {feature} | "
                    f"{metric(stats.get('spearman_vs_delta_kl'))} | "
                    f"{metric(stats.get('spearman_vs_delta_js'))} |"
                )

    for transition, data in paired.get("transitions", {}).items():
        prompt_groups = data.get("by_prompt", {})
        context_groups = data.get("by_context", {})
        full_prompt_better = sum(
            data_group.get("full_conf_risk", {}).get("spearman_vs_delta_kl")
            is not None
            and data_group.get("margin_only_risk", {}).get("spearman_vs_delta_kl")
            is not None
            and data_group["full_conf_risk"]["spearman_vs_delta_kl"]
            > data_group["margin_only_risk"]["spearman_vs_delta_kl"]
            for data_group in prompt_groups.values()
        )
        full_context_better = sum(
            data_group.get("full_conf_risk", {}).get("spearman_vs_delta_kl")
            is not None
            and data_group.get("margin_only_risk", {}).get("spearman_vs_delta_kl")
            is not None
            and data_group["full_conf_risk"]["spearman_vs_delta_kl"]
            > data_group["margin_only_risk"]["spearman_vs_delta_kl"]
            for data_group in context_groups.values()
        )
        lines.extend(
            [
                "",
                f"`{transition}` full-vs-margin-only delta-KL Spearman: "
                f"full higher in {full_prompt_better}/{len(prompt_groups)} prompts "
                f"and {full_context_better}/{len(context_groups)} contexts.",
            ]
        )

    lines.extend(
        [
            "",
            "## BF16 Control",
            "",
            f"- KL zero: `{control.get('kl_zero', False)}`",
            f"- JS zero: `{control.get('js_zero', False)}`",
            f"- top1 flip zero: `{control.get('top1_flip_zero', False)}`",
            "",
            "## Interpretation",
            "",
            "All comparisons are teacher-forced and require identical prefix, suffix, "
            "target, and alignment hashes. No QDM witness, quantizer feature, "
            "training, backward pass, precision controller, or CUDA attention change "
            "is used by this scorer.",
            "",
        ]
    )
    return "\n".join(lines)


def write_conf_makv_artifacts(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    block_size: int = CONF_MAKV_BLOCK_SIZE,
    run_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the standalone CONF-MaKV validation artifact set."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized = [dict(row) for row in rows]
    summary, block_rows = summarize_conf_makv_rows(normalized, block_size=block_size)
    summary["run"] = dict(run_info or {})
    summary["artifacts"] = {
        "per_token": "per_token_conf_makv.jsonl",
        "blocks": "block_conf_makv.jsonl",
        "summary": "summary.json",
        "report": "validation_report.md",
    }
    _write_jsonl(output / "per_token_conf_makv.jsonl", normalized)
    _write_jsonl(output / "block_conf_makv.jsonl", block_rows)
    _write_json(output / "summary.json", summary)
    report = output / "validation_report.md"
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(_validation_report(summary), encoding="utf-8")
    temporary.replace(report)
    return summary


__all__ = [
    "ALIGNMENT_FIELDS",
    "CONF_FEATURES",
    "CONF_MAKV_BLOCK_SIZE",
    "CONF_MAKV_VERSION",
    "CONF_MAKV_WEIGHTS",
    "CONF_RISK_SEMANTICS",
    "CONF_SCORER_VERSION",
    "PrecisionRiskSignal",
    "PRECISION_COMPOSITIONS",
    "aggregate_conf_blocks",
    "assert_conf_teacher_forced_alignment",
    "compute_confidence_metrics",
    "compute_precision_risk_signal",
    "make_conf_teacher_forced_rows",
    "summarize_conf_makv_rows",
    "write_conf_makv_artifacts",
]
