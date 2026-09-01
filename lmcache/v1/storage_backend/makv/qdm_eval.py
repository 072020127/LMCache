# SPDX-License-Identifier: Apache-2.0

"""Phase 1.5, teacher-forced QDM validity evaluation.

This module contains the model-independent part of the validation protocol.
It compares logits for identical token prefixes, aggregates block witnesses
without materializing attention weights, and writes the per-token and summary
artifacts used by the offline study.

The model runner lives under ``experiments/scoutrank_transfer`` so that the
production MaKV package does not acquire a Transformers dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from lmcache.v1.storage_backend.makv.qdm import (
    QDM_BLOCK_SIZE,
    QDMMetadata,
    PRECISION_ID_BF16,
    PRECISION_ID_K2V2,
    PRECISION_ID_K4V2,
    PRECISION_ID_K8V4,
    PRECISION_ID_MIXED,
)

PRECISION_COMPOSITIONS = ("K2V2", "K4V2", "K8V4", "MIXED", "BF16")
VALIDATION_PROTOCOL = "qdm_phase1_5_teacher_forced_v1"
KV_COMPRESSION_SCOPE = "teacher_forced_prefix_only_suffix_bf16"
QDM_VALIDATION_ANALYSIS_VERSION = "qdm_real_qwen3_8b_empirical_v1"
QDM_METRIC_NAMES = ("max_tv_bound", "p95_tv_bound", "max_attention_error")
QDM_DIAGNOSTIC_METRIC_NAMES = (
    "mean_attention_error",
    "p90_attention_error",
    "p95_attention_error",
    "top_k_layer_mean",
)
QDM_ANALYSIS_METRIC_NAMES = QDM_METRIC_NAMES + QDM_DIAGNOSTIC_METRIC_NAMES
EXACT_DRIFT_METRIC_NAMES = (
    "exact_max_score_error",
    "exact_attention_TV",
    "exact_attention_output_error",
)
BOUND_TIGHTNESS_METRIC_NAMES = (
    "score_cauchy_ratio",
    "tv_block_max_ratio",
    "tv_transform_ratio",
    "tv_bound_exact_ratio",
    "output_bound_exact_ratio",
)
QDM_TOP_K_LAYER_COUNT = 4
# Evidence gate for the real-model study only. This is not a production
# threshold and is never used by QDM runtime classification.
QDM_MINIMUM_VALIDATION_PROMPTS = 4
# Offline-oracle integrity tolerances only. These values do not alter the
# production witness, clamp, or any runtime risk threshold.
EXACT_ORACLE_PROBABILITY_TOLERANCE = 1.0e-4
EXACT_ORACLE_BF16_ZERO_TOLERANCE = 1.0e-4
PAIRED_PRECISION_ORDER = ("BF16", "K8V4", "K4V2", "K2V2")
PAIRED_REQUIRED_PRECISIONS = PAIRED_PRECISION_ORDER + ("MIXED",)
PAIRED_MONOTONIC_METRICS = (
    "exact_attention_output_error",
    "qdm_attention_error",
    "kl_precision",
    "js_precision",
)
PAIRED_EXACT_METRICS = (
    "exact_max_score_error",
    "exact_attention_TV",
    "exact_attention_output_error",
)
PAIRED_ALIGNMENT_FIELDS = (
    "prefix_alignment_hash",
    "suffix_alignment_hash",
    "target_alignment_hash",
)
PAIRED_DELTA_TRANSITIONS = (
    ("K8V4_to_K4V2", "K8V4", "K4V2"),
    ("K4V2_to_K2V2", "K4V2", "K2V2"),
)
PAIRED_MIXED_TRANSITIONS = (
    ("MIXED_vs_BF16", "BF16", "MIXED"),
    ("MIXED_vs_K2V2", "K2V2", "MIXED"),
)
PAIRED_ANALYSIS_VERSION = "qdm_paired_precision_counterfactual_v1"
# These are evidence-support constants for offline stratification, not risk
# thresholds and not parameters used by QDM runtime classification.
PAIRED_MIN_GROUP_ROWS = 16
PAIRED_MIN_PROMPT_SUPPORT = 3
PAIRED_NUMERICAL_TOLERANCE = 1.0e-7
SENSITIVITY_ANALYSIS_VERSION = "qdm_downstream_sensitivity_oracle_v1"
SENSITIVITY_REQUIRED_FIELDS = (
    "physical_norm_only",
    "sensitivity_weighted_error",
    "sensitivity_signed_error",
    "hidden_delta_norm_sum",
    "hidden_delta_norm_max",
    "hidden_delta_norm_final",
    "logit_delta_l2",
    "margin_abs_delta",
)
SENSITIVITY_FEATURES = (
    "physical_norm_only",
    "sensitivity_weighted_error",
    "sensitivity_signed_error_abs",
    "hidden_delta_norm_sum",
    "hidden_delta_norm_max",
    "hidden_delta_norm_final",
)
SENSITIVITY_TRANSITIONS = (
    ("BF16_to_K8V4", "BF16", "K8V4"),
    ("K8V4_to_K4V2", "K8V4", "K4V2"),
    ("K4V2_to_K2V2", "K4V2", "K2V2"),
)
SENSITIVITY_MIXED_TRANSITIONS = (
    ("MIXED_vs_BF16", "BF16", "MIXED"),
    ("MIXED_vs_K2V2", "K2V2", "MIXED"),
)
SENSITIVITY_OUTCOMES = (
    "vs_delta_KL",
    "vs_delta_JS",
    "vs_delta_top1_flip",
    "vs_delta_logit_delta_l2",
    "vs_delta_margin_abs",
)
SENSITIVITY_MIN_GROUP_ROWS = 16
SENSITIVITY_NUMERICAL_TOLERANCE = 1.0e-7
PRECISION_ID_NAMES = {
    PRECISION_ID_K2V2: "K2V2",
    PRECISION_ID_K4V2: "K4V2",
    PRECISION_ID_K8V4: "K8V4",
    PRECISION_ID_BF16: "BF16",
    PRECISION_ID_MIXED: "MIXED",
}


def _as_rows(logits: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(logits)
    if tensor.ndim != 2 or tensor.shape[-1] < 2:
        raise ValueError(f"{name} must have shape [steps, vocab]")
    return tensor.float()


def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
    values = torch.topk(logits, k=2, dim=-1).values
    return values[:, 0] - values[:, 1]


def _topk_entropy(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    values = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1).values
    log_prob = values - torch.logsumexp(values, dim=-1, keepdim=True)
    probability = log_prob.exp()
    return -(probability * log_prob).sum(dim=-1)


def teacher_forced_logit_metrics(
    reference_logits: torch.Tensor,
    quantized_logits: torch.Tensor,
    *,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """Compute per-step BF16/reference versus quantized decision metrics.

    Both tensors must be produced with the same teacher-forced input IDs. The
    function intentionally does not accept generated token sequences, which
    keeps free-running trajectory divergence out of the ground truth.
    """
    reference = _as_rows(reference_logits, "reference_logits")
    quantized = _as_rows(quantized_logits, "quantized_logits")
    if reference.shape != quantized.shape:
        raise ValueError("reference and quantized logits must have equal shapes")

    reference_log_prob = F.log_softmax(reference, dim=-1)
    quantized_log_prob = F.log_softmax(quantized, dim=-1)
    reference_probability = reference_log_prob.exp()
    kl = (reference_probability * (reference_log_prob - quantized_log_prob)).sum(dim=-1)

    quantized_probability = quantized_log_prob.exp()
    mixture = 0.5 * (reference_probability + quantized_probability)
    mixture_log_prob = torch.log(mixture.clamp_min(torch.finfo(torch.float32).tiny))
    js = 0.5 * (
        (reference_probability * (reference_log_prob - mixture_log_prob)).sum(dim=-1)
        + (quantized_probability * (quantized_log_prob - mixture_log_prob)).sum(dim=-1)
    )
    reference_margin = _top2_margin(reference)
    quantized_margin = _top2_margin(quantized)
    entropy = _topk_entropy(reference, top_k)
    reference_top1 = reference.argmax(dim=-1)
    quantized_top1 = quantized.argmax(dim=-1)

    result = []
    for step in range(reference.shape[0]):
        result.append(
            {
                "kl_bf16_quantized": float(kl[step].item()),
                "kl_divergence": float(kl[step].item()),
                "js_divergence": float(js[step].item()),
                "top1_flip": bool(
                    reference_top1[step].item() != quantized_top1[step].item()
                ),
                # The reference margin is the model-fragility signal. Keep
                # the quantized margin as a diagnostic for interpretation.
                "top1_top2_margin": float(reference_margin[step].item()),
                "top1_margin": float(reference_margin[step].item()),
                "reference_top1_top2_margin": float(reference_margin[step].item()),
                "quantized_top1_top2_margin": float(quantized_margin[step].item()),
                "topK_entropy": float(entropy[step].item()),
                "reference_top1_token": int(reference_top1[step].item()),
                "quantized_top1_token": int(quantized_top1[step].item()),
            }
        )
    return result


@dataclass(frozen=True)
class QDMBlockObservation:
    """One query-head observation from a streaming attention reference pass.

    ``block_probability`` and ``visible_v_norm`` cover every attention-visible
    block, including newly appended BF16 tokens. The production witness is
    copied into the corresponding prefix blocks by ``aggregate_qdm_step``.
    """

    layer: int
    step: int
    query_head: int
    kv_head: int
    query_norm: float
    block_probability: torch.Tensor
    visible_v_norm: torch.Tensor
    visible_block_start: int = 0
    query_vector: torch.Tensor | None = None
    visible_key_end: int | None = None

    def __post_init__(self) -> None:
        probability = torch.as_tensor(self.block_probability)
        value_norm = torch.as_tensor(self.visible_v_norm)
        if probability.ndim != 1 or value_norm.ndim != 1:
            raise ValueError("QDM block observations must be one-dimensional")
        if probability.shape != value_norm.shape:
            raise ValueError(
                "QDM probabilities and visible value norms must have equal lengths"
            )
        if self.layer < 0 or self.step < 0 or self.query_head < 0 or self.kv_head < 0:
            raise ValueError("QDM observation indices must be non-negative")
        if not math.isfinite(float(self.query_norm)) or self.query_norm < 0:
            raise ValueError("QDM query norm must be finite and non-negative")
        if self.query_vector is not None:
            query = torch.as_tensor(self.query_vector)
            if query.ndim != 1:
                raise ValueError("QDM exact query vector must be one-dimensional")
        if self.visible_key_end is not None and self.visible_key_end <= 0:
            raise ValueError("QDM visible key end must be positive")


QDM_ATTENTION_IMPLEMENTATION = "qdm_streaming_reference_v1"


@dataclass(frozen=True)
class _MaskRow:
    values: torch.Tensor
    is_boolean: bool


def _mask_row(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    query_index: int,
    start: int,
    end: int,
) -> _MaskRow | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        row = attention_mask[
            min(batch, attention_mask.shape[0] - 1),
            0,
            min(query_index, attention_mask.shape[2] - 1),
            start:end,
        ]
    elif attention_mask.ndim == 3:
        row = attention_mask[
            min(batch, attention_mask.shape[0] - 1),
            min(query_index, attention_mask.shape[1] - 1),
            start:end,
        ]
    elif attention_mask.ndim == 2:
        row = attention_mask[min(batch, attention_mask.shape[0] - 1), start:end]
    else:
        raise ValueError(
            f"unsupported attention mask shape: {tuple(attention_mask.shape)}"
        )
    if row.numel() == 1 and end - start != 1:
        row = row.expand(end - start)
    if row.numel() != end - start:
        raise ValueError("attention mask key dimension does not match K/V")
    return _MaskRow(values=row, is_boolean=row.dtype == torch.bool)


class StreamingQDMCollector:
    """Keep scalar QDM observations and optional exact-oracle reference traces."""

    def __init__(self, *, capture_exact: bool = False) -> None:
        self._observations: dict[tuple[int, int, int], QDMBlockObservation] = {}
        self.capture_exact = bool(capture_exact)
        self._exact_kv: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def record_kv(
        self,
        *,
        layer: int,
        kv_head: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Retain one reference K/V trace for the offline exact oracle only."""
        if not self.capture_exact:
            return
        if key.ndim != 2 or value.shape != key.shape:
            raise ValueError("exact QDM K/V traces must have shape [tokens, dim]")
        trace_key = (int(layer), int(kv_head))
        existing = self._exact_kv.get(trace_key)
        if existing is not None:
            if existing[0].shape != key.shape or not torch.equal(existing[0], key):
                raise RuntimeError(f"inconsistent exact QDM K trace: {trace_key}")
            if not torch.equal(existing[1], value):
                raise RuntimeError(f"inconsistent exact QDM V trace: {trace_key}")
            return
        self._exact_kv[trace_key] = (key.detach(), value.detach())

    def record(
        self,
        *,
        layer: int,
        step: int,
        query_head: int,
        kv_head: int,
        query: torch.Tensor,
        block_probability: Sequence[torch.Tensor],
        visible_v_norm: Sequence[torch.Tensor],
        visible_key_end: int | None = None,
    ) -> None:
        probability = torch.stack([value.detach().cpu() for value in block_probability])
        value_norm = torch.stack([value.detach().cpu() for value in visible_v_norm])
        key = (int(layer), int(step), int(query_head))
        if key in self._observations:
            raise RuntimeError(f"duplicate QDM attention observation: {key}")
        self._observations[key] = QDMBlockObservation(
            layer=int(layer),
            step=int(step),
            query_head=int(query_head),
            kv_head=int(kv_head),
            query_norm=float(torch.linalg.vector_norm(query.float()).item()),
            block_probability=probability,
            visible_v_norm=value_norm,
            query_vector=(query.detach().clone() if self.capture_exact else None),
            visible_key_end=(
                int(visible_key_end) if visible_key_end is not None else None
            ),
        )

    def for_step(self, step: int) -> list[QDMBlockObservation]:
        result = [
            observation
            for (
                layer,
                current_step,
                query_head,
            ), observation in self._observations.items()
            if current_step == step
        ]
        return sorted(result, key=lambda item: (item.layer, item.query_head))

    def exact_kv(
        self, layer: int, kv_head: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.capture_exact:
            raise RuntimeError("exact QDM capture is disabled")
        try:
            return self._exact_kv[(int(layer), int(kv_head))]
        except KeyError as error:
            raise KeyError(
                f"missing exact QDM K/V trace for layer={layer}, kv_head={kv_head}"
            ) from error


_ACTIVE_COLLECTOR: ContextVar[StreamingQDMCollector | None] = ContextVar(
    "makv_qdm_active_collector", default=None
)


def qdm_streaming_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    """Reference attention with scalar block observation and no score matrix."""
    if dropout:
        raise ValueError("QDM streaming reference requires dropout=0")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("QDM streaming attention expects [batch, heads, seq, dim]")
    if key.shape != value.shape:
        raise ValueError("QDM key/value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("QDM query and KV batch/head dimensions are incompatible")
    if key.shape[1] <= 0 or query.shape[1] % key.shape[1] != 0:
        raise ValueError("QDM requires an integer query-to-KV head grouping")

    collector = _ACTIVE_COLLECTOR.get()
    output = torch.empty_like(query)
    query_groups = query.shape[1] // key.shape[1]
    key_length = key.shape[2]
    query_length = query.shape[2]
    if collector is not None and collector.capture_exact:
        if query.shape[0] != 1:
            raise ValueError("exact QDM capture currently requires batch size one")
        layer_index = int(getattr(module, "layer_idx", 0))
        for kv_head in range(key.shape[1]):
            collector.record_kv(
                layer=layer_index,
                kv_head=kv_head,
                key=key[0, kv_head],
                value=value[0, kv_head],
            )
    for batch in range(query.shape[0]):
        # Keep the existing mask semantics while reusing each query row for
        # every KV head. The score temporary below is one query block only.
        mask_rows = [
            _mask_row(
                attention_mask,
                batch=batch,
                query_index=query_index,
                start=0,
                end=key_length,
            )
            for query_index in range(query_length)
        ]
        visible_key_ends: list[int | None] = [None] * query_length
        if collector is not None and collector.capture_exact:
            for query_index, row in enumerate(mask_rows):
                if row is None:
                    visible_key_ends[query_index] = key_length
                    continue
                if row.is_boolean:
                    row_valid = row.values.to(torch.bool)
                else:
                    row_valid = torch.isfinite(row.values)
                    if torch.is_floating_point(row.values):
                        row_valid = row_valid & row.values.ne(
                            torch.finfo(row.values.dtype).min
                        )
                valid_positions = torch.nonzero(row_valid, as_tuple=False)
                if valid_positions.numel():
                    candidate_end = int(valid_positions[-1].item()) + 1
                    prefix_valid = bool(row_valid[:candidate_end].all().item())
                    suffix_masked = not bool(row_valid[candidate_end:].any().item())
                    if prefix_valid and suffix_masked:
                        visible_key_ends[query_index] = candidate_end
        for kv_head in range(key.shape[1]):
            group_start = kv_head * query_groups
            group_end = group_start + query_groups
            query_states = query[batch, group_start:group_end].float()
            key_states = key[batch, kv_head].float()
            value_states = value[batch, kv_head].float()
            running_max = torch.full(
                (query_groups, query_length),
                -math.inf,
                dtype=torch.float32,
                device=query.device,
            )
            running_denominator = torch.zeros(
                query_groups, query_length, dtype=torch.float32, device=query.device
            )
            running_numerator = torch.zeros(
                (query_groups, query_length, key.shape[-1]),
                dtype=torch.float32,
                device=query.device,
            )
            seen_valid = torch.zeros(
                query_groups, query_length, dtype=torch.bool, device=query.device
            )
            block_masses: list[torch.Tensor] = []
            block_norms: list[torch.Tensor] = []
            for start in range(0, key_length, QDM_BLOCK_SIZE):
                end = min(start + QDM_BLOCK_SIZE, key_length)
                # This is [query_groups, query_length, block_size], never the
                # full attention score matrix over the visible KV range.
                scores = torch.einsum(
                    "gqd,bd->gqb", query_states, key_states[start:end]
                )
                scores = scores * float(scaling)
                block_masks = [
                    None if row is None else row.values[start:end]
                    for row in mask_rows
                ]
                if all(row is None for row in block_masks):
                    valid = torch.ones_like(scores, dtype=torch.bool)
                    masked_scores = scores
                else:
                    if any(row is None for row in block_masks):
                        raise ValueError("QDM attention mask rows must be consistent")
                    mask_values = torch.stack(
                        [row for row in block_masks if row is not None]
                    )
                    is_boolean = all(
                        row is not None and mask_rows[index].is_boolean
                        for index, row in enumerate(block_masks)
                    )
                    if is_boolean:
                        valid = mask_values.to(torch.bool).unsqueeze(0).expand(
                            query_groups, -1, -1
                        )
                        masked_scores = scores.masked_fill(~valid, -math.inf)
                    else:
                        mask_values = mask_values.to(scores.dtype)
                        masked_scores = scores + mask_values.unsqueeze(0)
                        valid = torch.isfinite(masked_scores)
                        if torch.is_floating_point(mask_values):
                            valid = valid & mask_values.ne(
                                torch.finfo(mask_values.dtype).min
                            )

                has_valid = valid.any(dim=-1)
                local_max = masked_scores.masked_fill(~valid, -math.inf).amax(dim=-1)
                new_max = torch.maximum(running_max, local_max)
                rescale = torch.where(
                    seen_valid,
                    torch.exp(running_max - new_max),
                    torch.zeros_like(running_max),
                )
                if block_masses:
                    block_masses = [mass * rescale for mass in block_masses]
                safe_max = torch.where(
                    torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
                )
                local_exp = torch.where(
                    valid,
                    torch.exp(masked_scores - safe_max.unsqueeze(-1)),
                    torch.zeros_like(masked_scores),
                )
                running_numerator = (
                    running_numerator * rescale.unsqueeze(-1)
                    + local_exp.matmul(value_states[start:end])
                )
                running_denominator = (
                    running_denominator * rescale + local_exp.sum(dim=-1)
                )
                block_masses.append(local_exp.sum(dim=-1))
                token_norms = torch.linalg.vector_norm(
                    value_states[start:end], dim=-1
                )
                block_norms.append(
                    torch.where(
                        valid,
                        token_norms.view(1, 1, -1),
                        torch.zeros_like(token_norms).view(1, 1, -1),
                    ).amax(dim=-1)
                )
                running_max = new_max
                seen_valid = seen_valid | has_valid

            if not bool(seen_valid.all().item()) or not bool(
                (running_denominator > 0.0).all().item()
            ):
                raise RuntimeError("QDM attention query has no visible KV token")
            output[batch, group_start:group_end] = (
                running_numerator / running_denominator.unsqueeze(-1)
            ).to(output.dtype)
            if collector is not None:
                probabilities = torch.stack(block_masses, dim=-1) / (
                    running_denominator.unsqueeze(-1)
                )
                norms = torch.stack(block_norms, dim=-1)
                for local_head in range(query_groups):
                    query_head = group_start + local_head
                    for query_index in range(query_length):
                        collector.record(
                            layer=int(getattr(module, "layer_idx", 0)),
                            step=query_index,
                            query_head=query_head,
                            kv_head=kv_head,
                            query=query[batch, query_head, query_index],
                            block_probability=probabilities[
                                local_head, query_index
                            ],
                            visible_v_norm=norms[local_head, query_index],
                            visible_key_end=visible_key_ends[query_index],
                        )
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def qdm_reference_attention(
    model: Any,
    collector: StreamingQDMCollector | None = None,
) -> Iterator[None]:
    """Temporarily select the Python streaming attention implementation."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(
        QDM_ATTENTION_IMPLEMENTATION,
        qdm_streaming_attention,
    )
    configs = []
    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
    ):
        if candidate is not None and all(candidate is not item for item in configs):
            configs.append(candidate)
    previous = [getattr(config, "_attn_implementation", None) for config in configs]
    for config in configs:
        config._attn_implementation = QDM_ATTENTION_IMPLEMENTATION
    token = _ACTIVE_COLLECTOR.set(collector)
    try:
        yield
    finally:
        _ACTIVE_COLLECTOR.reset(token)
        for config, value in zip(configs, previous, strict=True):
            config._attn_implementation = value


def _expanded_witness(
    metadata: QDMMetadata,
    observation: QDMBlockObservation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = torch.as_tensor(observation.block_probability, dtype=torch.float32)
    visible_v_norm = torch.as_tensor(
        observation.visible_v_norm, dtype=torch.float32, device=probability.device
    )
    if probability.numel() == 0:
        raise ValueError("QDM observations must contain at least one visible block")
    if not 0 <= observation.layer < metadata.num_layers:
        raise ValueError("QDM observation layer is outside witness metadata")
    if not 0 <= observation.kv_head < metadata.num_kv_heads:
        raise ValueError("QDM observation KV head is outside witness metadata")

    block_count = probability.numel()
    k_error = torch.zeros(block_count, dtype=torch.float32, device=probability.device)
    v_error = torch.zeros_like(k_error)
    witness_start = metadata.block_start - observation.visible_block_start
    witness_end = witness_start + metadata.num_blocks
    overlap_start = max(0, witness_start)
    overlap_end = min(block_count, witness_end)
    if overlap_start < overlap_end:
        local_start = overlap_start - witness_start
        local_end = local_start + overlap_end - overlap_start
        k_error[overlap_start:overlap_end] = metadata.k_error[
            observation.layer, local_start:local_end, observation.kv_head
        ].to(device=probability.device)
        v_error[overlap_start:overlap_end] = metadata.v_error[
            observation.layer, local_start:local_end, observation.kv_head
        ].to(device=probability.device)
    return probability, k_error, v_error, visible_v_norm


def _estimate_observation(
    metadata: QDMMetadata,
    observation: QDMBlockObservation,
    *,
    head_dim: int,
) -> dict[str, float]:
    probability, k_error, v_error, visible_v_norm = _expanded_witness(
        metadata, observation
    )
    visible_v_norm_max = visible_v_norm.max()
    if not bool(torch.count_nonzero(k_error).item()) and not bool(
        torch.count_nonzero(v_error).item()
    ):
        # A zero production witness has mathematically zero drift. Avoid
        # turning block-probability summation roundoff into a BF16 false
        # positive while retaining the visible-range norm for auditability.
        return {
            "k_tv_bound": 0.0,
            "v_error": 0.0,
            "attention_error_bound": 0.0,
            "v_norm_max": float(visible_v_norm_max.item()),
            "raw_A": 1.0,
            "raw_tv_bound": 0.0,
            "log_A": 0.0,
            "saturated": False,
        }
    # This is the same scalar certificate as QDMScalarAccumulator, expressed
    # as one tensor reduction so the offline evaluator does not spend millions
    # of Python calls on unchanged production math.
    query_norm = torch.tensor(
        float(observation.query_norm), dtype=torch.float32, device=probability.device
    )
    c_block = query_norm * k_error / math.sqrt(head_dim)
    a = torch.sum(probability * torch.exp(c_block))
    v_error_sum = torch.sum(probability * v_error)
    if not bool(torch.isfinite(a).item()):
        tv = torch.ones_like(a)
    else:
        tv = torch.clamp((a.square() - 1.0) / 2.0, min=0.0, max=1.0)
    attention_error = 2.0 * tv * visible_v_norm_max + v_error_sum
    estimate = {
        "k_tv_bound": float(tv.item()),
        "v_error": float(v_error_sum.item()),
        "attention_error_bound": float(attention_error.item()),
        "v_norm_max": float(visible_v_norm_max.item()),
    }
    # These are reference-only diagnostics from the same scalar accumulator.
    # They are intentionally not a replacement certificate for the clamped TV.
    raw_a = float(a.item())
    if math.isfinite(raw_a) and raw_a > 0.0:
        raw_tv_bound = (raw_a * raw_a - 1.0) / 2.0
        log_a = math.log(raw_a)
        saturated = raw_tv_bound >= 1.0
    else:
        raw_tv_bound = None
        log_a = None
        saturated = True
    return {
        "k_tv_bound": estimate["k_tv_bound"],
        "v_error": estimate["v_error"],
        "attention_error_bound": estimate["attention_error_bound"],
        "v_norm_max": estimate["v_norm_max"],
        "raw_A": raw_a,
        "raw_tv_bound": raw_tv_bound,
        "log_A": log_a,
        "saturated": saturated,
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
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


def _layer_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [steps, hidden]")
    return tensor.float()


def compute_layer_drift_trace(
    reference_attention: Sequence[torch.Tensor],
    quantized_attention: Sequence[torch.Tensor],
    reference_hidden: Sequence[torch.Tensor],
    quantized_hidden: Sequence[torch.Tensor],
    margin_gradients: Sequence[torch.Tensor],
    reference_logits: torch.Tensor,
    quantized_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    """Compute offline per-layer drift and margin-gradient projections.

    The activation and gradient tensors are transient inputs to this function.
    Returned records contain scalar norms, cosines, and dot products rather
    than hidden-size vectors, so the JSONL artifact remains bounded. The
    gradient is the BF16 top1-vs-top2 margin gradient with respect to the
    post-``o_proj`` attention output of each decoder layer.
    """
    sequences = (
        reference_attention,
        quantized_attention,
        reference_hidden,
        quantized_hidden,
        margin_gradients,
    )
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("layer drift requires a non-empty layer sequence")
    layer_count = len(reference_attention)
    if any(len(sequence) != layer_count for sequence in sequences):
        raise ValueError("layer drift sequences must have equal layer counts")

    reference_logit_rows = _as_rows(reference_logits, "reference_logits")
    quantized_logit_rows = _as_rows(quantized_logits, "quantized_logits")
    if reference_logit_rows.shape != quantized_logit_rows.shape:
        raise ValueError("layer drift logits must have equal shapes")
    steps = int(reference_logit_rows.shape[0])
    if steps <= 0:
        raise ValueError("layer drift requires at least one decode step")

    references = []
    quantized = []
    h_references = []
    h_quantized = []
    gradients = []
    for layer in range(layer_count):
        ref_attn = _layer_tensor(reference_attention[layer], "reference_attention")
        quant_attn = _layer_tensor(quantized_attention[layer], "quantized_attention")
        ref_hidden = _layer_tensor(reference_hidden[layer], "reference_hidden")
        quant_hidden = _layer_tensor(quantized_hidden[layer], "quantized_hidden")
        gradient = _layer_tensor(margin_gradients[layer], "margin_gradient")
        if ref_attn.shape[0] != steps or any(
            tensor.shape != ref_attn.shape for tensor in (quant_attn, gradient)
        ):
            raise ValueError("attention and margin-gradient shapes do not match")
        if ref_hidden.shape[0] != steps or quant_hidden.shape != ref_hidden.shape:
            raise ValueError("hidden-state shapes do not match logit steps")
        references.append(ref_attn)
        quantized.append(quant_attn)
        h_references.append(ref_hidden)
        h_quantized.append(quant_hidden)
        gradients.append(gradient)

    reference_top2 = torch.topk(reference_logit_rows.detach(), k=2, dim=-1).indices
    reference_margin = (
        reference_logit_rows.gather(1, reference_top2[:, :1]).squeeze(1)
        - reference_logit_rows.gather(1, reference_top2[:, 1:2]).squeeze(1)
    )
    quantized_top2 = torch.topk(quantized_logit_rows.detach(), k=2, dim=-1).indices
    quantized_margin = (
        quantized_logit_rows.gather(1, quantized_top2[:, :1]).squeeze(1)
        - quantized_logit_rows.gather(1, quantized_top2[:, 1:2]).squeeze(1)
    )

    per_layer: list[dict[str, torch.Tensor]] = []
    for layer in range(layer_count):
        delta_attention = quantized[layer] - references[layer]
        delta_hidden = h_quantized[layer] - h_references[layer]
        signed_directional = (gradients[layer] * delta_attention).sum(dim=-1)
        per_layer.append(
            {
                "attention_output_bf16_norm": torch.linalg.vector_norm(
                    references[layer], dim=-1
                ),
                "attention_output_quantized_norm": torch.linalg.vector_norm(
                    quantized[layer], dim=-1
                ),
                "delta_attn_norm": torch.linalg.vector_norm(delta_attention, dim=-1),
                "hidden_state_bf16_norm": torch.linalg.vector_norm(
                    h_references[layer], dim=-1
                ),
                "hidden_state_quantized_norm": torch.linalg.vector_norm(
                    h_quantized[layer], dim=-1
                ),
                "delta_hidden_norm": torch.linalg.vector_norm(delta_hidden, dim=-1),
                "hidden_cosine": F.cosine_similarity(
                    h_references[layer], h_quantized[layer], dim=-1, eps=1.0e-12
                ),
                "gradient_norm": torch.linalg.vector_norm(gradients[layer], dim=-1),
                "signed_directional_error": signed_directional,
                "directional_error": signed_directional.abs(),
            }
        )

    output: list[dict[str, Any]] = []
    logit_delta = quantized_logit_rows - reference_logit_rows
    for step in range(steps):
        layer_rows: list[dict[str, Any]] = []
        delta_attn_norms = []
        delta_hidden_norms = []
        signed_values = []
        directional_values = []
        for layer in range(layer_count):
            values = per_layer[layer]
            attn_norm = float(values["delta_attn_norm"][step].item())
            hidden_norm = float(values["delta_hidden_norm"][step].item())
            signed = float(values["signed_directional_error"][step].item())
            directional = float(values["directional_error"][step].item())
            delta_attn_norms.append(attn_norm)
            delta_hidden_norms.append(hidden_norm)
            signed_values.append(signed)
            directional_values.append(directional)
            layer_rows.append(
                {
                    "layer": layer,
                    **{
                        key: float(value[step].item())
                        for key, value in values.items()
                    },
                    "hidden_to_attention_ratio": (
                        hidden_norm / attn_norm if attn_norm > 0.0 else None
                    ),
                }
            )

        transition_count = max(0, layer_count - 1)
        amplified = sum(
            right > left + SENSITIVITY_NUMERICAL_TOLERANCE
            for left, right in zip(
                delta_hidden_norms, delta_hidden_norms[1:], strict=False
            )
        )
        attenuated = sum(
            right + SENSITIVITY_NUMERICAL_TOLERANCE < left
            for left, right in zip(
                delta_hidden_norms, delta_hidden_norms[1:], strict=False
            )
        )
        directional_total = sum(directional_values)
        signed_total = sum(signed_values)
        cancellation_ratio = (
            1.0 - min(1.0, abs(signed_total) / directional_total)
            if directional_total > 0.0
            else 0.0
        )
        hidden_peak = max(delta_hidden_norms, default=0.0)
        hidden_final = delta_hidden_norms[-1] if delta_hidden_norms else 0.0
        output.append(
            {
                "step": step,
                "layers": layer_rows,
                "summary": {
                    "physical_norm_only": sum(delta_attn_norms),
                    "physical_norm_max": max(delta_attn_norms, default=0.0),
                    "sensitivity_weighted_error": directional_total,
                    "sensitivity_signed_error": signed_total,
                    "sensitivity_signed_error_abs": abs(signed_total),
                    "directional_cancellation_ratio": cancellation_ratio,
                    "hidden_delta_norm_sum": sum(delta_hidden_norms),
                    "hidden_delta_norm_max": hidden_peak,
                    "hidden_delta_norm_final": hidden_final,
                    "hidden_amplified_transition_count": int(amplified),
                    "hidden_attenuated_transition_count": int(attenuated),
                    "hidden_transition_count": transition_count,
                    "hidden_amplified_transition_fraction": (
                        amplified / transition_count if transition_count else 0.0
                    ),
                    "hidden_attenuated_transition_fraction": (
                        attenuated / transition_count if transition_count else 0.0
                    ),
                    "hidden_peak_to_final_ratio": (
                        hidden_final / hidden_peak if hidden_peak > 0.0 else None
                    ),
                    "logit_delta_l2": float(
                        torch.linalg.vector_norm(logit_delta[step]).item()
                    ),
                    "reference_margin": float(reference_margin[step].item()),
                    "quantized_margin": float(quantized_margin[step].item()),
                    "margin_delta": float(
                        (quantized_margin[step] - reference_margin[step]).item()
                    ),
                    "margin_abs_delta": float(
                        (quantized_margin[step] - reference_margin[step])
                        .abs()
                        .item()
                    ),
                },
            }
        )
    return output


def aggregate_qdm_step(
    metadata: QDMMetadata,
    observations: Iterable[QDMBlockObservation],
    *,
    head_dim: int,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Aggregate one decode step over all observed layers and query heads.

    The value norm passed to the scalar accumulator is the maximum norm over
    all visible blocks. This is deliberately independent of the precision
    bucket containing the witness and is the key correctness condition for
    ``2 * TV_K * v_norm_max + V_error``.
    """
    items = list(observations)
    if not items:
        raise ValueError("at least one QDM block observation is required")
    if expected_step is not None and any(item.step != expected_step for item in items):
        raise ValueError("QDM observations contain more than the requested step")
    estimates = [
        (
            item,
            _estimate_observation(metadata, item, head_dim=head_dim),
        )
        for item in items
    ]
    tv_values = [estimate["k_tv_bound"] for _, estimate in estimates]
    v_values = [estimate["v_error"] for _, estimate in estimates]
    attention_values = [estimate["attention_error_bound"] for _, estimate in estimates]
    layer_attention: dict[int, list[float]] = {}
    for item, estimate in estimates:
        layer_attention.setdefault(int(item.layer), []).append(
            float(estimate["attention_error_bound"])
        )
    layer_means = {
        layer: sum(values) / len(values)
        for layer, values in layer_attention.items()
    }
    top_layer_means = sorted(layer_means.values(), reverse=True)[
        :QDM_TOP_K_LAYER_COUNT
    ]

    def saturation_counts(key_function: Any) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for item, estimate in estimates:
            key = str(key_function(item))
            bucket = counts.setdefault(
                key,
                {"saturated_count": 0, "observation_count": 0},
            )
            bucket["observation_count"] += 1
            bucket["saturated_count"] += int(bool(estimate["saturated"]))
        return counts

    worst_item, worst_estimate = max(
        estimates,
        key=lambda pair: (
            pair[1]["attention_error_bound"],
            -pair[0].layer,
            -pair[0].kv_head,
            -pair[0].query_head,
        ),
    )
    formula_errors = [
        abs(
            estimate["attention_error_bound"]
            - (
                2.0 * estimate["k_tv_bound"] * estimate["v_norm_max"]
                + estimate["v_error"]
            )
        )
        for _, estimate in estimates
    ]
    return {
        "max_tv_bound": max(tv_values),
        "p95_tv_bound": _percentile(tv_values, 0.95),
        "max_v_error": max(v_values),
        "max_attention_error": max(attention_values),
        "mean_attention_error": sum(attention_values) / len(attention_values),
        "p90_attention_error": _percentile(attention_values, 0.90),
        "p95_attention_error": _percentile(attention_values, 0.95),
        "top_k_layer_mean": (
            sum(top_layer_means) / len(top_layer_means)
            if top_layer_means
            else 0.0
        ),
        "top_k_layer_count": min(QDM_TOP_K_LAYER_COUNT, len(top_layer_means)),
        "worst_layer": int(worst_item.layer),
        "worst_kv_head": int(worst_item.kv_head),
        "worst_layer_identity": (
            f"layer={int(worst_item.layer)},kv_head={int(worst_item.kv_head)}"
        ),
        "v_norm_max": max(estimate["v_norm_max"] for _, estimate in estimates),
        "attention_error_formula_max_abs_error": max(formula_errors),
        "qdm_score": max(attention_values),
        "qdm_observation_count": len(estimates),
        "saturated_observation_count": sum(
            int(bool(estimate["saturated"])) for _, estimate in estimates
        ),
        "saturation_rate": sum(
            int(bool(estimate["saturated"])) for _, estimate in estimates
        )
        / len(estimates),
        "raw_A": worst_estimate["raw_A"],
        "raw_tv_bound": worst_estimate["raw_tv_bound"],
        "log_A": worst_estimate["log_A"],
        "saturated": bool(worst_estimate["saturated"]),
        "saturation_by_layer": saturation_counts(lambda item: item.layer),
        "saturation_by_kv_head": saturation_counts(lambda item: item.kv_head),
        "saturation_by_layer_head": saturation_counts(
            lambda item: f"layer={int(item.layer)},kv_head={int(item.kv_head)}"
        ),
        # Retain the worst decomposition for audit/debugging.
        "worst_k_tv_bound": worst_estimate["k_tv_bound"],
        "worst_v_error": worst_estimate["v_error"],
        "worst_v_norm_max": worst_estimate["v_norm_max"],
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    return float(numerator / denominator)


def compute_exact_attention_drift(
    metadata: QDMMetadata,
    collector: StreamingQDMCollector,
    source: torch.Tensor,
    restored: torch.Tensor,
    *,
    prefix_length: int,
    head_dim: int,
    block_size: int = QDM_BLOCK_SIZE,
) -> list[dict[str, Any]]:
    """Compute a query-conditioned exact oracle for one restored KV cache.

    This is deliberately reference-only. It compares BF16 source K/V and the
    actual production-restored K/V while using the BF16 query. K/V work is
    streamed per layer/KV head group; no model forward or production metadata
    is changed by this oracle.
    """
    if not collector.capture_exact:
        raise ValueError("exact QDM computation requires capture_exact=True")
    if source.ndim != 5 or restored.shape != source.shape or source.shape[1] != 2:
        raise ValueError("exact QDM source/restored tensors must have [L,2,T,H,D]")
    if prefix_length != int(source.shape[2]) or prefix_length <= 0:
        raise ValueError("exact QDM prefix length must match the source KV block")
    if head_dim != int(source.shape[-1]) or block_size <= 0:
        raise ValueError("exact QDM geometry does not match the source KV block")
    if metadata.num_layers != int(source.shape[0]):
        raise ValueError("exact QDM metadata layer count does not match source KV")
    if metadata.num_kv_heads != int(source.shape[3]):
        raise ValueError("exact QDM metadata head count does not match source KV")

    observations_by_group: dict[tuple[int, int], list[QDMBlockObservation]] = {}
    all_observations = [
        observation
        for step in range(
            max(
                (observation.step for observation in collector._observations.values()),
                default=-1,
            )
            + 1
        )
        for observation in collector.for_step(step)
    ]
    if not all_observations:
        raise ValueError("exact QDM computation requires reference observations")
    for observation in all_observations:
        if observation.query_vector is None:
            raise ValueError("exact QDM observation is missing the query vector")
        if observation.visible_key_end is None:
            raise ValueError(
                "exact QDM requires a contiguous visible-key mask per observation"
            )
        observations_by_group.setdefault(
            (observation.layer, observation.kv_head), []
        ).append(observation)

    results: list[dict[str, Any]] = []
    scale = 1.0 / math.sqrt(float(head_dim))
    for (layer, kv_head), group in sorted(observations_by_group.items()):
        reference_key_trace, reference_value_trace = collector.exact_kv(
            layer, kv_head
        )
        if (
            reference_key_trace.ndim != 2
            or reference_value_trace.shape != reference_key_trace.shape
        ):
            raise ValueError("exact QDM reference K/V trace shape is invalid")
        device = reference_key_trace.device
        reference_key = reference_key_trace.float().clone()
        reference_value = reference_value_trace.float().clone()
        source_key = source[layer, 0, :, kv_head].to(device=device, dtype=torch.float32)
        source_value = source[layer, 1, :, kv_head].to(
            device=device, dtype=torch.float32
        )
        restored_key = restored[layer, 0, :, kv_head].to(
            device=device, dtype=torch.float32
        )
        restored_value = restored[layer, 1, :, kv_head].to(
            device=device, dtype=torch.float32
        )
        key_length = int(reference_key.shape[0])
        if key_length < prefix_length:
            raise ValueError("exact QDM reference trace is shorter than the prefix")
        reference_key[:prefix_length] = source_key
        reference_value[:prefix_length] = source_value
        quantized_key = reference_key.clone()
        quantized_value = reference_value.clone()
        quantized_key[:prefix_length] = restored_key
        quantized_value[:prefix_length] = restored_value

        residual_norm = torch.linalg.vector_norm(
            source_key - restored_key, dim=-1
        )
        block_count = (key_length + block_size - 1) // block_size
        block_k_error = torch.zeros(block_count, device=device, dtype=torch.float32)
        block_v_error = torch.zeros_like(block_k_error)
        metadata_blocks = min(block_count, metadata.num_blocks)
        block_k_error[:metadata_blocks] = metadata.k_error[
            layer, :metadata_blocks, kv_head
        ].to(device=device, dtype=torch.float32)
        block_v_error[:metadata_blocks] = metadata.v_error[
            layer, :metadata_blocks, kv_head
        ].to(device=device, dtype=torch.float32)
        block_v_full = torch.repeat_interleave(block_v_error, block_size)[:key_length]
        token_residual_full = torch.zeros(key_length, device=device)
        token_residual_full[:prefix_length] = residual_norm
        group_by_key = {
            (observation.step, observation.query_head): observation
            for observation in group
        }
        steps = sorted({observation.step for observation in group})
        query_heads = sorted({observation.query_head for observation in group})
        if not steps or not query_heads:
            raise ValueError("exact QDM observation group is empty")
        queries = torch.stack(
            [
                torch.as_tensor(
                    group_by_key[(step, query_head)].query_vector,
                    device=device,
                    dtype=torch.float32,
                )
                for step in steps
                for query_head in query_heads
            ]
        ).reshape(len(steps), len(query_heads), head_dim)
        visible_ends = torch.tensor(
            [
                int(group_by_key[(step, query_heads[0])].visible_key_end)
                for step in steps
            ],
            device=device,
            dtype=torch.long,
        )
        if bool((visible_ends > key_length).any().item()):
            raise ValueError("exact QDM visible-key range exceeds the K/V trace")
        positions = torch.arange(key_length, device=device)
        valid = positions.unsqueeze(0) < visible_ends.unsqueeze(1)
        valid_group = valid[:, None, :]
        reference_scores = torch.einsum(
            "sgd,kd->sgk", queries, reference_key
        ) * scale
        quantized_scores = torch.einsum(
            "sgd,kd->sgk", queries, quantized_key
        ) * scale
        reference_scores = reference_scores.masked_fill(~valid_group, -math.inf)
        quantized_scores = quantized_scores.masked_fill(~valid_group, -math.inf)
        reference_probability = torch.softmax(reference_scores, dim=-1)
        quantized_probability = torch.softmax(quantized_scores, dim=-1)
        score_delta = torch.where(
            valid_group,
            reference_scores - quantized_scores,
            torch.zeros_like(reference_scores),
        )
        exact_score_error = score_delta.abs()
        exact_score_max = exact_score_error[:, :, :prefix_length].amax(dim=-1)
        exact_score_l2 = torch.linalg.vector_norm(
            score_delta[:, :, :prefix_length], dim=-1
        )
        exact_score_mean_abs = exact_score_error[:, :, :prefix_length].sum(
            dim=-1
        ) / float(prefix_length)

        query_norm = torch.linalg.vector_norm(queries, dim=-1)
        token_cauchy = (
            query_norm[:, :, None]
            * token_residual_full[:prefix_length][None, None, :]
            * scale
        )
        token_cauchy_full = torch.zeros(
            len(steps), len(query_heads), key_length, device=device
        )
        token_cauchy_full[:, :, :prefix_length] = token_cauchy
        block_cauchy = query_norm[:, :, None] * block_k_error[None, None, :] * scale
        block_cauchy_full = torch.repeat_interleave(
            block_cauchy, block_size, dim=-1
        )[:, :, :key_length]
        block_v_error_full = block_v_full[None, None, :].expand(
            len(steps), len(query_heads), -1
        )
        token_a = (
            reference_probability * torch.exp(token_cauchy_full)
        ).sum(dim=-1)
        block_a = (
            reference_probability * torch.exp(block_cauchy_full)
        ).sum(dim=-1)
        token_raw_tv = (token_a.square() - 1.0) / 2.0
        block_raw_tv = (block_a.square() - 1.0) / 2.0
        production_tv = block_raw_tv.clamp(0.0, 1.0)
        visible_v_norm_max = torch.linalg.vector_norm(
            reference_value, dim=-1
        ).amax()
        production_v_error = (
            reference_probability * block_v_error_full
        ).sum(dim=-1)
        production_output_bound = (
            2.0 * production_tv * visible_v_norm_max + production_v_error
        )

        reference_output = torch.einsum(
            "sgk,kd->sgd", reference_probability, reference_value
        )
        score_only_output = torch.einsum(
            "sgk,kd->sgd", quantized_probability, reference_value
        )
        quantized_output = torch.einsum(
            "sgk,kd->sgd", quantized_probability, quantized_value
        )
        exact_tv = 0.5 * (
            reference_probability - quantized_probability
        ).abs().sum(dim=-1)
        exact_score_only_output_error = torch.linalg.vector_norm(
            reference_output - score_only_output, dim=-1
        )
        exact_value_output_error = torch.linalg.vector_norm(
            score_only_output - quantized_output, dim=-1
        )
        exact_output_error = torch.linalg.vector_norm(
            reference_output - quantized_output, dim=-1
        )

        padded_probability = F.pad(
            reference_probability,
            (0, block_count * block_size - key_length),
        )
        reference_block_probability = padded_probability.reshape(
            len(steps), len(query_heads), block_count, block_size
        ).sum(dim=-1)
        for step_index, step in enumerate(steps):
            for head_index, query_head in enumerate(query_heads):
                observation = group_by_key[(step, query_head)]
                observed_probability = torch.as_tensor(
                    observation.block_probability,
                    device=device,
                    dtype=torch.float32,
                )
                if observed_probability.shape != (
                    reference_block_probability.shape[-1],
                ):
                    raise ValueError(
                        "exact QDM reference block probability shape mismatch"
                    )
                probability_error = float(
                    (
                        reference_block_probability[step_index, head_index]
                        - observed_probability
                    )
                    .abs()
                    .max()
                    .item()
                )
                exact_score = float(exact_score_max[step_index, head_index].item())
                block_score = float(
                    block_cauchy_full[step_index, head_index].max().item()
                )
                token_tv = float(token_raw_tv[step_index, head_index].item())
                block_tv = float(block_raw_tv[step_index, head_index].item())
                exact_tv_value = float(exact_tv[step_index, head_index].item())
                exact_output = float(
                    exact_output_error[step_index, head_index].item()
                )
                production_bound = float(
                    production_output_bound[step_index, head_index].item()
                )
                results.append(
                    {
                        "step": int(step),
                        "layer": int(layer),
                        "query_head": int(query_head),
                        "kv_head": int(kv_head),
                        "exact_max_score_error": exact_score,
                        "exact_score_error_l2": float(
                            exact_score_l2[step_index, head_index].item()
                        ),
                        "exact_score_error_mean_abs": float(
                            exact_score_mean_abs[step_index, head_index].item()
                        ),
                        "exact_attention_TV": exact_tv_value,
                        "exact_attention_output_error": exact_output,
                        "exact_score_only_output_error": float(
                            exact_score_only_output_error[
                                step_index, head_index
                            ].item()
                        ),
                        "exact_value_output_error": float(
                            exact_value_output_error[step_index, head_index].item()
                        ),
                        "cauchy_score_error_bound": float(
                            token_cauchy[step_index, head_index].max().item()
                        ),
                        "block_cauchy_score_error_bound": block_score,
                        "token_cauchy_raw_tv_bound": token_tv,
                        "block_cauchy_raw_tv_bound": block_tv,
                        "production_raw_tv_bound": block_tv,
                        "production_tv_bound": float(
                            production_tv[step_index, head_index].item()
                        ),
                        "production_v_error_bound": float(
                            production_v_error[step_index, head_index].item()
                        ),
                        "production_attention_error_bound": production_bound,
                        "score_cauchy_ratio": _safe_ratio(block_score, exact_score),
                        "tv_block_max_ratio": _safe_ratio(block_tv, token_tv),
                        "tv_transform_ratio": _safe_ratio(token_tv, exact_tv_value),
                        "tv_bound_exact_ratio": _safe_ratio(
                            min(1.0, block_tv), exact_tv_value
                        ),
                        "output_bound_exact_ratio": _safe_ratio(
                            production_bound, exact_output
                        ),
                        "reference_block_probability_max_abs_error": probability_error,
                        "saturated": block_tv >= 1.0,
                    }
                )
    return sorted(
        results,
        key=lambda item: (item["step"], item["layer"], item["query_head"]),
    )


def aggregate_exact_attention_step(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact-oracle observations without changing production QDM."""
    items = [dict(item) for item in observations]
    if not items:
        raise ValueError("at least one exact QDM observation is required")

    def maximum(field: str) -> float:
        values = [
            float(item[field])
            for item in items
            if item.get(field) is not None and math.isfinite(float(item[field]))
        ]
        return max(values, default=0.0)

    def finite_maximum(field: str) -> float | None:
        values = [
            float(item[field])
            for item in items
            if item.get(field) is not None and math.isfinite(float(item[field]))
        ]
        return max(values) if values else None

    exact_worst = max(
        items,
        key=lambda item: (
            float(item.get("exact_attention_output_error", 0.0)),
            -int(item.get("layer", 0)),
            -int(item.get("kv_head", 0)),
            -int(item.get("query_head", 0)),
        ),
    )
    bound_worst = max(
        items,
        key=lambda item: (
            float(item.get("production_attention_error_bound", 0.0)),
            -int(item.get("layer", 0)),
            -int(item.get("kv_head", 0)),
            -int(item.get("query_head", 0)),
        ),
    )
    result: dict[str, Any] = {
        "exact_observation_count": len(items),
        "exact_max_score_error": maximum("exact_max_score_error"),
        "exact_score_error_l2": maximum("exact_score_error_l2"),
        "exact_score_error_mean_abs": maximum("exact_score_error_mean_abs"),
        "exact_attention_TV": maximum("exact_attention_TV"),
        "exact_attention_output_error": maximum(
            "exact_attention_output_error"
        ),
        "exact_score_only_output_error": maximum(
            "exact_score_only_output_error"
        ),
        "exact_value_output_error": maximum("exact_value_output_error"),
        "cauchy_score_error_bound": maximum("cauchy_score_error_bound"),
        "block_cauchy_score_error_bound": maximum(
            "block_cauchy_score_error_bound"
        ),
        "token_cauchy_raw_tv_bound": maximum("token_cauchy_raw_tv_bound"),
        "block_cauchy_raw_tv_bound": maximum("block_cauchy_raw_tv_bound"),
        "production_raw_tv_bound": maximum("production_raw_tv_bound"),
        "production_tv_bound": maximum("production_tv_bound"),
        "production_v_error_bound": maximum("production_v_error_bound"),
        "production_attention_error_bound": maximum(
            "production_attention_error_bound"
        ),
        "reference_block_probability_max_abs_error": maximum(
            "reference_block_probability_max_abs_error"
        ),
        "saturated": any(bool(item.get("saturated", False)) for item in items),
        "exact_worst_layer": int(exact_worst["layer"]),
        "exact_worst_kv_head": int(exact_worst["kv_head"]),
        "exact_worst_layer_identity": (
            f"layer={int(exact_worst['layer'])},"
            f"kv_head={int(exact_worst['kv_head'])}"
        ),
        "production_bound_worst_layer": int(bound_worst["layer"]),
        "production_bound_worst_kv_head": int(bound_worst["kv_head"]),
        "production_bound_worst_layer_identity": (
            f"layer={int(bound_worst['layer'])},"
            f"kv_head={int(bound_worst['kv_head'])}"
        ),
    }
    for field in BOUND_TIGHTNESS_METRIC_NAMES:
        result[field] = finite_maximum(field)
    return result


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps([int(token) for token in token_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_teacher_forced_prefix_alignment(
    reference_prefix_token_ids: Sequence[int],
    quantized_prefix_token_ids: Sequence[int],
) -> str:
    """Require the exact token prefix used by both teacher-forced runs."""
    reference = [int(token) for token in reference_prefix_token_ids]
    quantized = [int(token) for token in quantized_prefix_token_ids]
    if reference != quantized:
        raise ValueError("BF16 and quantized teacher-forced prefixes are not identical")
    return _token_hash(reference)


def assert_teacher_forced_sequence_alignment(
    reference_prefix_token_ids: Sequence[int],
    quantized_prefix_token_ids: Sequence[int],
    *,
    reference_suffix_input_ids: Sequence[int] | None = None,
    quantized_suffix_input_ids: Sequence[int] | None = None,
    reference_target_token_ids: Sequence[int] | None = None,
    quantized_target_token_ids: Sequence[int] | None = None,
) -> dict[str, str | None]:
    """Require identical teacher-forced tokens and return auditable hashes.

    The target IDs are not fed to the model, but checking them prevents an
    evaluation harness from comparing logits against a shifted or different
    ground-truth token stream.
    """
    hashes: dict[str, str | None] = {
        "prefix_token_id_hash": assert_teacher_forced_prefix_alignment(
            reference_prefix_token_ids, quantized_prefix_token_ids
        ),
        "suffix_input_token_id_hash": None,
        "target_token_id_hash": None,
    }
    if (reference_suffix_input_ids is None) != (quantized_suffix_input_ids is None):
        raise ValueError("both teacher-forced suffix sequences are required")
    if reference_suffix_input_ids is not None:
        if [int(value) for value in reference_suffix_input_ids] != [
            int(value) for value in quantized_suffix_input_ids or []
        ]:
            raise ValueError(
                "BF16 and quantized teacher-forced suffixes are not identical"
            )
        hashes["suffix_input_token_id_hash"] = _token_hash(reference_suffix_input_ids)

    if (reference_target_token_ids is None) != (quantized_target_token_ids is None):
        raise ValueError("both teacher-forced target sequences are required")
    if reference_target_token_ids is not None:
        if [int(value) for value in reference_target_token_ids] != [
            int(value) for value in quantized_target_token_ids or []
        ]:
            raise ValueError(
                "BF16 and quantized teacher-forced targets are not identical"
            )
        hashes["target_token_id_hash"] = _token_hash(reference_target_token_ids)
    return hashes


def make_teacher_forced_rows(
    *,
    sample: str,
    prefix_token_ids: Sequence[int],
    precision_composition: str,
    reference_logits: torch.Tensor,
    quantized_logits: torch.Tensor,
    qdm_by_step: Sequence[Mapping[str, Any]],
    top_k: int = 50,
    quantized_prefix_token_ids: Sequence[int] | None = None,
    suffix_input_ids: Sequence[int] | None = None,
    target_token_ids: Sequence[int] | None = None,
    quantized_suffix_input_ids: Sequence[int] | None = None,
    quantized_target_token_ids: Sequence[int] | None = None,
    context_length: int | None = None,
    requested_context_length: int | None = None,
    row_metadata: Mapping[str, Any] | None = None,
    witness_summary: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build traceable per-token rows for one aligned precision run."""
    if precision_composition not in PRECISION_COMPOSITIONS:
        raise ValueError(f"unknown precision composition: {precision_composition}")
    metrics = teacher_forced_logit_metrics(
        reference_logits, quantized_logits, top_k=top_k
    )
    if len(metrics) != len(qdm_by_step):
        raise ValueError("one QDM aggregate is required for each decode step")
    if suffix_input_ids is not None and len(suffix_input_ids) != len(metrics):
        raise ValueError("suffix_input_ids length does not match decode steps")
    if target_token_ids is not None and len(target_token_ids) != len(metrics):
        raise ValueError("target_token_ids length does not match decode steps")
    prefix_ids = [int(token) for token in prefix_token_ids]
    quantized_prefix_ids = (
        prefix_ids
        if quantized_prefix_token_ids is None
        else [int(token) for token in quantized_prefix_token_ids]
    )
    suffix_hashes = assert_teacher_forced_sequence_alignment(
        prefix_ids,
        quantized_prefix_ids,
        reference_suffix_input_ids=suffix_input_ids,
        quantized_suffix_input_ids=(
            suffix_input_ids
            if quantized_suffix_input_ids is None and suffix_input_ids is not None
            else quantized_suffix_input_ids
        ),
        reference_target_token_ids=target_token_ids,
        quantized_target_token_ids=(
            target_token_ids
            if quantized_target_token_ids is None and target_token_ids is not None
            else quantized_target_token_ids
        ),
    )
    rows = []
    for step, (logit_metric, qdm_metric) in enumerate(
        zip(metrics, qdm_by_step, strict=True)
    ):
        row: dict[str, Any] = {
            "protocol": VALIDATION_PROTOCOL,
            "sample": sample,
            "step": step,
            "precision_composition": precision_composition,
            "kv_compression_scope": KV_COMPRESSION_SCOPE,
            "prefix_length": len(prefix_ids),
            **suffix_hashes,
            "prefix_aligned": True,
            "suffix_aligned": True,
            "target_aligned": True,
            "teacher_forced": True,
            "free_running_ground_truth": False,
            **logit_metric,
            **dict(qdm_metric),
            "risk_state": "UNASSIGNED",
            "risk_state_calibration": "diagnostic_quantiles_not_calibrated",
        }
        if context_length is not None:
            row["context_length"] = int(context_length)
        if requested_context_length is not None:
            row["requested_context_length"] = int(requested_context_length)
        if row_metadata is not None:
            protected = set(row)
            conflicting = protected.intersection(row_metadata)
            if conflicting:
                raise ValueError(
                    "row_metadata cannot overwrite protected fields: "
                    + ", ".join(sorted(conflicting))
                )
            row.update(dict(row_metadata))
        if suffix_input_ids is not None:
            row["suffix_input_token_id"] = int(suffix_input_ids[step])
        if target_token_ids is not None:
            row["target_token_id"] = int(target_token_ids[step])
        if witness_summary is not None:
            if "qdm_version" in witness_summary:
                row["qdm_version"] = str(witness_summary["qdm_version"])
            if "quantizer_version" in witness_summary:
                row["quantizer_version"] = str(witness_summary["quantizer_version"])
            if "block_size" in witness_summary:
                row["qdm_block_size"] = int(witness_summary["block_size"])
            row["witness_max_k_error"] = float(witness_summary.get("max_k_error", 0.0))
            row["witness_max_v_error"] = float(witness_summary.get("max_v_error", 0.0))
            row["witness_max_v_norm"] = float(witness_summary.get("max_v_norm", 0.0))
            if "precision_ids" in witness_summary:
                row["witness_precision_ids"] = [
                    int(value) for value in witness_summary["precision_ids"]
                ]
            for field in ("precision_id_counts", "block_precision_id_counts"):
                if field in witness_summary:
                    row[f"witness_{field}"] = {
                        str(key): int(value)
                        for key, value in dict(witness_summary[field]).items()
                    }
        rows.append(row)
    return rows


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


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks without a scipy/sklearn dependency."""
    ordered = sorted(
        enumerate(float(value) for value in values),
        key=lambda item: (item[1], item[0]),
    )
    ranks = [0.0] * len(ordered)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right, strict=True)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    return _pearson(
        _average_ranks([pair[0] for pair in pairs]),
        _average_ranks([pair[1] for pair in pairs]),
    )


def _roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    """Compute AUROC from ranks; ties receive their average rank."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal lengths")
    pairs = [
        (float(score), 1 if bool(label) else 0)
        for score, label in zip(scores, labels, strict=True)
        if math.isfinite(float(score))
    ]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks([score for score, _ in pairs])
    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, pairs, strict=True) if label
    )
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _pr_auc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    """Compute threshold-grouped average precision (step PR-AUC)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal lengths")
    pairs = sorted(
        [
            (float(score), 1 if bool(label) else 0, index)
            for index, (score, label) in enumerate(zip(scores, labels, strict=True))
            if math.isfinite(float(score))
        ],
        key=lambda item: (-item[0], item[2]),
    )
    total_positive = sum(label for _, label, _ in pairs)
    if total_positive == 0 or not pairs:
        return None
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    position = 0
    while position < len(pairs):
        score = pairs[position][0]
        while position < len(pairs) and pairs[position][0] == score:
            true_positive += pairs[position][1]
            false_positive += 1 - pairs[position][1]
            position += 1
        recall = true_positive / total_positive
        precision = true_positive / (true_positive + false_positive)
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return float(area)


def _classification_metric(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    risk_direction: str,
) -> dict[str, Any]:
    values: list[float] = []
    kl: list[float] = []
    js: list[float] = []
    flip: list[float] = []
    for row in rows:
        value = _analysis_metric_value(row, metric)
        current_kl = float(row["kl_bf16_quantized"])
        current_js = float(row["js_divergence"])
        if not all(math.isfinite(item) for item in (value, current_kl, current_js)):
            continue
        values.append(value)
        kl.append(current_kl)
        js.append(current_js)
        flip.append(float(bool(row["top1_flip"])))
    if risk_direction == "higher_is_risk":
        risk_score = values
    elif risk_direction == "lower_is_risk":
        risk_score = [-value for value in values]
    else:
        raise ValueError(f"unknown risk direction: {risk_direction}")
    return {
        "count": len(values),
        "risk_direction": risk_direction,
        "spearman_vs_kl": _spearman(values, kl),
        "spearman_vs_js": _spearman(values, js),
        "spearman_vs_top1_flip": _spearman(values, flip),
        "auroc_top1_flip": _roc_auc(risk_score, flip),
        "pr_auc_top1_flip": _pr_auc(risk_score, flip),
    }


def _normalized_rank_scores(values: Sequence[float]) -> list[float]:
    ranks = _average_ranks(values)
    if len(ranks) <= 1:
        return [0.5 for _ in ranks]
    denominator = float(len(ranks) - 1)
    return [(rank - 1.0) / denominator for rank in ranks]


def _qdm_plus_margin_metric(
    rows: Sequence[Mapping[str, Any]], qdm_metric: str
) -> dict[str, Any]:
    qdm_scores = _normalized_rank_scores(
        [_analysis_metric_value(row, qdm_metric) for row in rows]
    )
    margin_scores = _normalized_rank_scores(
        [-float(row["top1_top2_margin"]) for row in rows]
    )
    combined = [
        qdm_score + margin_score
        for qdm_score, margin_score in zip(qdm_scores, margin_scores, strict=True)
    ]
    kl = [float(row["kl_bf16_quantized"]) for row in rows]
    js = [float(row["js_divergence"]) for row in rows]
    flip = [float(bool(row["top1_flip"])) for row in rows]
    return {
        "count": len(combined),
        "definition": (
            "normalized empirical rank(max_qdm_metric) + "
            "normalized empirical rank(-reference_top1_top2_margin)"
        ),
        "parameter_fitting": False,
        "spearman_vs_kl": _spearman(combined, kl),
        "spearman_vs_js": _spearman(combined, js),
        "spearman_vs_top1_flip": _spearman(combined, flip),
        "auroc_top1_flip": _roc_auc(combined, flip),
        "pr_auc_top1_flip": _pr_auc(combined, flip),
    }


def _quantile_analysis(
    rows: Sequence[Mapping[str, Any]], metric: str, *, bins: int = 4
) -> dict[str, Any]:
    if bins <= 0:
        raise ValueError("quantile bin count must be positive")
    values = [_analysis_metric_value(row, metric) for row in rows]
    if not values:
        return {"metric": metric, "quantile_boundaries": [], "buckets": []}
    ordered = sorted(range(len(rows)), key=lambda index: (values[index], index))
    grouped: list[list[Mapping[str, Any]]] = [[] for _ in range(bins)]
    for rank, index in enumerate(ordered):
        grouped[min(bins - 1, rank * bins // len(ordered))].append(rows[index])
    return {
        "metric": metric,
        "quantile_boundaries": [
            _percentile(values, index / bins) for index in range(bins + 1)
        ],
        "buckets": [
            {
                "quantile_range": [index / bins, (index + 1) / bins],
                "count": len(group),
                "value_min": min(
                    (_analysis_metric_value(row, metric) for row in group),
                    default=None,
                ),
                "value_max": max(
                    (_analysis_metric_value(row, metric) for row in group),
                    default=None,
                ),
                "top1_flip_rate": _rate(group),
                "kl_mean": (
                    sum(float(row["kl_bf16_quantized"]) for row in group) / len(group)
                    if group
                    else None
                ),
                "js_mean": (
                    sum(float(row["js_divergence"]) for row in group) / len(group)
                    if group
                    else None
                ),
            }
            for index, group in enumerate(grouped)
        ],
        "assignment": "stable_empirical_rank_quantiles_for_analysis_only",
    }


def _rate(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row["top1_flip"]) for row in rows) / len(rows)


def _analysis_metric_value(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None and metric in QDM_DIAGNOSTIC_METRIC_NAMES:
        # Keep older synthetic/reference rows analyzable. Real runs always
        # contain the non-production diagnostic aggregates.
        value = row.get("max_attention_error")
    if value is None:
        raise KeyError(f"missing QDM analysis metric: {metric}")
    return float(value)


def _assign_diagnostic_labels(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["precision_composition"]), []).append(row)
    for group in groups.values():
        qdm = [float(row["qdm_score"]) for row in group]
        margins = [float(row["top1_top2_margin"]) for row in group]
        qdm_threshold = _percentile(qdm, 0.75)
        margin_threshold = _percentile(margins, 0.25)
        for row, score, margin in zip(group, qdm, margins, strict=True):
            qdm_high = (
                score > 0.0 and qdm_threshold is not None and score >= qdm_threshold
            )
            margin_small = margin_threshold is not None and margin <= margin_threshold
            if qdm_high and margin_small:
                state = "KV_TOKEN_RISK"
            elif qdm_high:
                state = "KV_DRIFT_ROBUST"
            elif margin_small:
                state = "MODEL_FRAGILE"
            else:
                state = "SAFE"
            row["risk_state"] = state
            row["risk_state_calibration"] = "diagnostic_quantiles_not_calibrated"


def _diagnostic_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "means": {},
            "correlations": {
                "qdm_vs_kl_pearson": None,
                "qdm_vs_js_pearson": None,
                "qdm_vs_top1_flip_pearson": None,
                "qdm_vs_kl_spearman": None,
                "qdm_vs_js_spearman": None,
                "qdm_vs_top1_flip_spearman": None,
            },
            "metric_comparison": {
                metric: _classification_metric(
                    [],
                    metric,
                    risk_direction=(
                        "lower_is_risk" if metric == "top1_margin" else "higher_is_risk"
                    ),
                )
                for metric in (*QDM_ANALYSIS_METRIC_NAMES, "top1_margin")
            },
            "qdm_plus_margin": {},
            "qdm_plus_margin_by_metric": {},
            "quantile_analysis": {},
            "diagnostic_quantiles": {
                "qdm_high_quantile": 0.75,
                "margin_small_quantile": 0.25,
                "qdm_high_threshold": None,
                "margin_small_threshold": None,
            },
            "diagnostic_enrichment": {},
            "witness": {
                "max_k_error": 0.0,
                "max_v_error": 0.0,
                "max_v_norm": 0.0,
            },
        }

    qdm = [float(row["qdm_score"]) for row in rows]
    kl = [float(row["kl_bf16_quantized"]) for row in rows]
    js = [float(row["js_divergence"]) for row in rows]
    flip = [float(bool(row["top1_flip"])) for row in rows]
    margin = [float(row["top1_top2_margin"]) for row in rows]
    qdm_threshold = _percentile(qdm, 0.75)
    margin_threshold = _percentile(margin, 0.25)
    # This is a rank diagnostic, not a calibrated risk threshold. A zero QDM
    # distribution (the BF16 control) is not marked as high drift.
    qdm_high = [
        value > 0.0 and qdm_threshold is not None and value >= qdm_threshold
        for value in qdm
    ]
    margin_small = [
        margin_threshold is not None and value <= margin_threshold for value in margin
    ]
    for row, high, small in zip(rows, qdm_high, margin_small, strict=True):
        if high and small:
            state = "KV_TOKEN_RISK"
        elif high:
            state = "KV_DRIFT_ROBUST"
        elif small:
            state = "MODEL_FRAGILE"
        else:
            state = "SAFE"
        row["risk_state"] = state

    all_flip_rate = sum(flip) / len(flip)
    high_rows = [row for row, high in zip(rows, qdm_high, strict=True) if high]
    high_small_rows = [
        row
        for row, high, small in zip(rows, qdm_high, margin_small, strict=True)
        if high and small
    ]
    high_rate = _rate(high_rows)
    high_small_rate = _rate(high_small_rows)
    small_rows = [row for row, small in zip(rows, margin_small, strict=True) if small]

    def enrichment(rate: float | None) -> float | None:
        if rate is None or all_flip_rate == 0.0:
            return None
        return rate / all_flip_rate

    witness_keys = ("witness_max_k_error", "witness_max_v_error", "witness_max_v_norm")
    witness = {
        key.removeprefix("witness_"): max(float(row.get(key, 0.0)) for row in rows)
        for key in witness_keys
    }
    metric_comparison = {
        metric: _classification_metric(
            rows,
            metric,
            risk_direction=(
                "lower_is_risk" if metric == "top1_margin" else "higher_is_risk"
            ),
        )
        for metric in (*QDM_ANALYSIS_METRIC_NAMES, "top1_margin")
    }
    qdm_plus_margin_by_metric = {
        metric: _qdm_plus_margin_metric(rows, metric)
        for metric in QDM_ANALYSIS_METRIC_NAMES
    }

    def _relative_rate(
        numerator: float | None, denominator: float | None
    ) -> float | None:
        if numerator is None or denominator in (None, 0.0):
            return None
        return numerator / denominator

    return {
        "count": len(rows),
        "means": {
            "qdm_score": sum(qdm) / len(qdm),
            "max_tv_bound": sum(float(row["max_tv_bound"]) for row in rows) / len(rows),
            "p95_tv_bound": sum(float(row["p95_tv_bound"]) for row in rows) / len(rows),
            "max_v_error": sum(float(row["max_v_error"]) for row in rows) / len(rows),
            "max_attention_error": sum(
                float(row["max_attention_error"]) for row in rows
            )
            / len(rows),
            "mean_attention_error": sum(
                _analysis_metric_value(row, "mean_attention_error") for row in rows
            )
            / len(rows),
            "p90_attention_error": sum(
                _analysis_metric_value(row, "p90_attention_error") for row in rows
            )
            / len(rows),
            "p95_attention_error": sum(
                _analysis_metric_value(row, "p95_attention_error") for row in rows
            )
            / len(rows),
            "top_k_layer_mean": sum(
                _analysis_metric_value(row, "top_k_layer_mean") for row in rows
            )
            / len(rows),
            "saturation_rate": sum(
                float(row.get("saturation_rate", 0.0)) for row in rows
            )
            / len(rows),
            "kl_bf16_quantized": sum(kl) / len(kl),
            "js_divergence": sum(js) / len(js),
            "top1_flip_rate": sum(flip) / len(flip),
            "top1_top2_margin": sum(margin) / len(margin),
            "top1_margin": sum(margin) / len(margin),
            "topK_entropy": sum(float(row["topK_entropy"]) for row in rows) / len(rows),
        },
        "correlations": {
            "qdm_vs_kl_pearson": _pearson(qdm, kl),
            "qdm_vs_js_pearson": _pearson(qdm, js),
            "qdm_vs_top1_flip_pearson": _pearson(qdm, flip),
            "qdm_vs_kl_spearman": _spearman(qdm, kl),
            "qdm_vs_js_spearman": _spearman(qdm, js),
            "qdm_vs_top1_flip_spearman": _spearman(qdm, flip),
        },
        "metric_comparison": metric_comparison,
        "qdm_plus_margin": qdm_plus_margin_by_metric["max_attention_error"],
        "qdm_plus_margin_by_metric": qdm_plus_margin_by_metric,
        "quantile_analysis": {
            metric: _quantile_analysis(rows, metric)
            for metric in QDM_ANALYSIS_METRIC_NAMES
        },
        "diagnostic_quantiles": {
            "qdm_high_quantile": 0.75,
            "margin_small_quantile": 0.25,
            "qdm_high_threshold": qdm_threshold,
            "margin_small_threshold": margin_threshold,
        },
        "diagnostic_enrichment": {
            "all_flip_rate": all_flip_rate,
            "qdm_high_count": len(high_rows),
            "qdm_high_flip_rate": high_rate,
            "qdm_high_flip_enrichment": enrichment(high_rate),
            "qdm_high_small_margin_count": len(high_small_rows),
            "qdm_high_small_margin_flip_rate": high_small_rate,
            "qdm_high_small_margin_flip_enrichment": enrichment(high_small_rate),
            "small_margin_count": len(small_rows),
            "small_margin_flip_rate": _rate(small_rows),
            "small_margin_flip_enrichment": enrichment(_rate(small_rows)),
            "qdm_high_small_margin_vs_qdm_high_enrichment": _relative_rate(
                high_small_rate, high_rate
            ),
            "qdm_high_small_margin_vs_small_margin_enrichment": _relative_rate(
                high_small_rate, _rate(small_rows)
            ),
        },
        "witness": witness,
    }


def _distribution_stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "p90": _percentile(finite, 0.90),
        "p95": _percentile(finite, 0.95),
        "max": max(finite),
    }


def _saturation_observation_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostic_rows = [
        row
        for row in rows
        if all(
            field in row for field in ("raw_A", "raw_tv_bound", "log_A", "saturated")
        )
    ]
    observation_count = sum(
        int(row.get("qdm_observation_count", 0)) for row in diagnostic_rows
    )
    saturated_count = sum(
        int(row.get("saturated_observation_count", 0)) for row in diagnostic_rows
    )
    raw_a = [
        float(row["raw_A"])
        for row in diagnostic_rows
        if row.get("raw_A") is not None
    ]
    raw_tv = [
        float(row["raw_tv_bound"])
        for row in diagnostic_rows
        if row.get("raw_tv_bound") is not None
    ]
    log_a = [
        float(row["log_A"])
        for row in diagnostic_rows
        if row.get("log_A") is not None
    ]
    return {
        "row_count": len(rows),
        "diagnostic_row_count": len(diagnostic_rows),
        "diagnostic_coverage": len(diagnostic_rows) / len(rows) if rows else None,
        "observation_count": observation_count,
        "saturated_observation_count": saturated_count,
        "saturation_rate": (
            saturated_count / observation_count if observation_count else None
        ),
        "token_saturation_rate": (
            sum(bool(row.get("saturated", False)) for row in diagnostic_rows)
            / len(diagnostic_rows)
            if diagnostic_rows
            else None
        ),
        "raw_A": _distribution_stats(raw_a),
        "raw_tv_bound": _distribution_stats(raw_tv),
        "log_A": _distribution_stats(log_a),
    }


def _saturation_map_stats(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, int]] = {}
    for row in rows:
        mapping = row.get(field, {})
        if not isinstance(mapping, Mapping):
            continue
        for key, value in mapping.items():
            if not isinstance(value, Mapping):
                continue
            bucket = merged.setdefault(
                str(key), {"saturated_count": 0, "observation_count": 0}
            )
            bucket["saturated_count"] += int(value.get("saturated_count", 0))
            bucket["observation_count"] += int(value.get("observation_count", 0))
    return {
        key: {
            **counts,
            "saturation_rate": (
                counts["saturated_count"] / counts["observation_count"]
                if counts["observation_count"]
                else None
            ),
        }
        for key, counts in sorted(merged.items())
    }


def build_saturation_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build clamp-saturation diagnostics without changing the TV certificate."""
    normalized = [dict(row) for row in rows]
    quantized_rows = [
        row for row in normalized if row.get("precision_composition") != "BF16"
    ]
    precision_groups = {
        precision: [
            row
            for row in normalized
            if row.get("precision_composition") == precision
        ]
        for precision in PRECISION_COMPOSITIONS
    }
    context_values = sorted(
        {
            int(row["context_length"])
            for row in quantized_rows
            if row.get("context_length") is not None
        }
    )
    context_groups = {
        str(context): [
            row
            for row in quantized_rows
            if int(row["context_length"]) == context
        ]
        for context in context_values
    }
    by_precision = {
        precision: _saturation_observation_stats(group)
        for precision, group in precision_groups.items()
    }
    by_context = {
        context: _saturation_observation_stats(group)
        for context, group in context_groups.items()
    }
    return {
        "diagnostic_only": True,
        "certificate_unchanged": True,
        "definition": (
            "raw_A and raw_tv_bound are read from the same scalar accumulator "
            "before the production TV clamp; raw_tv_bound >= 1 marks saturation"
        ),
        "overall": _saturation_observation_stats(normalized),
        "quantized_overall": _saturation_observation_stats(quantized_rows),
        "by_precision": by_precision,
        "by_context": by_context,
        "saturation_rate": _saturation_observation_stats(quantized_rows)[
            "saturation_rate"
        ],
        "saturation_rate_by_precision": {
            key: value["saturation_rate"] for key, value in by_precision.items()
        },
        "saturation_rate_by_context": {
            key: value["saturation_rate"] for key, value in by_context.items()
        },
        "saturation_rate_by_layer": _saturation_map_stats(
            quantized_rows, "saturation_by_layer"
        ),
        "saturation_rate_by_kv_head": _saturation_map_stats(
            quantized_rows, "saturation_by_kv_head"
        ),
        "saturation_rate_by_layer_head": _saturation_map_stats(
            quantized_rows, "saturation_by_layer_head"
        ),
    }


def _enrichment_for_masks(
    rows: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Sequence[bool]],
) -> dict[str, Any]:
    all_rate = _rate(rows)

    def selected(name: str) -> list[Mapping[str, Any]]:
        return [row for row, keep in zip(rows, masks[name], strict=True) if keep]

    result: dict[str, Any] = {"all_flip_rate": all_rate}
    rates: dict[str, float | None] = {}
    for name in masks:
        selected_rows = selected(name)
        rate = _rate(selected_rows)
        rates[name] = rate
        result[name] = {
            "count": len(selected_rows),
            "flip_rate": rate,
            "enrichment": (
                rate / all_rate
                if rate is not None and all_rate not in (None, 0.0)
                else None
            ),
        }
    if "qdm_high" in rates and "small_margin" in rates:
        combined_mask = [
            high and small
            for high, small in zip(
                masks["qdm_high"], masks["small_margin"], strict=True
            )
        ]
        combined_rows = [
            row for row, keep in zip(rows, combined_mask, strict=True) if keep
        ]
        combined_rate = _rate(combined_rows)
        result["qdm_high_small_margin"] = {
            "count": len(combined_rows),
            "flip_rate": combined_rate,
            "enrichment": (
                combined_rate / all_rate
                if combined_rate is not None and all_rate not in (None, 0.0)
                else None
            ),
        }
    return result


def _margin_conditioned_qdm(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bins: int = 4,
) -> dict[str, Any]:
    if not rows:
        return {"metric": metric, "bins": [], "supported_bin_count": 0}
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["top1_top2_margin"]),
            index,
        ),
    )
    grouped: list[list[Mapping[str, Any]]] = [[] for _ in range(bins)]
    for rank, index in enumerate(ordered):
        grouped[min(bins - 1, rank * bins // len(ordered))].append(rows[index])
    result_bins = []
    for index, group in enumerate(grouped):
        qdm_values = [_analysis_metric_value(row, metric) for row in group]
        threshold = _percentile(qdm_values, 0.75)
        high_rows = [
            row
            for row, value in zip(group, qdm_values, strict=True)
            if value > 0.0 and threshold is not None and value >= threshold
        ]
        low_rows = [row for row in group if row not in high_rows]

        def mean(
            rows_for_mean: Sequence[Mapping[str, Any]], field: str
        ) -> float | None:
            if not rows_for_mean:
                return None
            return sum(float(row[field]) for row in rows_for_mean) / len(rows_for_mean)

        high_flip = _rate(high_rows)
        low_flip = _rate(low_rows)
        high_kl = mean(high_rows, "kl_bf16_quantized")
        low_kl = mean(low_rows, "kl_bf16_quantized")
        high_js = mean(high_rows, "js_divergence")
        low_js = mean(low_rows, "js_divergence")
        result_bins.append(
            {
                "margin_quantile_range": [index / bins, (index + 1) / bins],
                "count": len(group),
                "margin_min": min(
                    (float(row["top1_top2_margin"]) for row in group),
                    default=None,
                ),
                "margin_max": max(
                    (float(row["top1_top2_margin"]) for row in group),
                    default=None,
                ),
                "qdm_high_threshold": threshold,
                "qdm_high_count": len(high_rows),
                "qdm_low_count": len(low_rows),
                "qdm_high_flip_rate": high_flip,
                "qdm_low_flip_rate": low_flip,
                "qdm_high_minus_low_flip_rate": (
                    high_flip - low_flip
                    if high_flip is not None and low_flip is not None
                    else None
                ),
                "qdm_high_kl_mean": high_kl,
                "qdm_low_kl_mean": low_kl,
                "qdm_high_minus_low_kl": (
                    high_kl - low_kl
                    if high_kl is not None and low_kl is not None
                    else None
                ),
                "qdm_high_js_mean": high_js,
                "qdm_low_js_mean": low_js,
                "qdm_high_minus_low_js": (
                    high_js - low_js
                    if high_js is not None and low_js is not None
                    else None
                ),
            }
        )
    supported = [
        item
        for item in result_bins
        if item["qdm_high_count"] > 0 and item["qdm_low_count"] > 0
    ]
    return {
        "metric": metric,
        "bins": result_bins,
        "bin_count": bins,
        "supported_bin_count": len(supported),
        "positive_flip_support_count": sum(
            item["qdm_high_minus_low_flip_rate"] is not None
            and item["qdm_high_minus_low_flip_rate"] > 0.0
            for item in supported
        ),
        "positive_kl_support_count": sum(
            item["qdm_high_minus_low_kl"] is not None
            and item["qdm_high_minus_low_kl"] > 0.0
            for item in supported
        ),
        "positive_js_support_count": sum(
            item["qdm_high_minus_low_js"] is not None
            and item["qdm_high_minus_low_js"] > 0.0
            for item in supported
        ),
        "analysis_only": True,
    }


def _incremental_group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "by_qdm_metric": {}}
    margin_values = [float(row["top1_top2_margin"]) for row in rows]
    margin_threshold = _percentile(margin_values, 0.25)
    small_margin = [
        margin_threshold is not None and value <= margin_threshold
        for value in margin_values
    ]
    margin_only = _classification_metric(
        rows, "top1_margin", risk_direction="lower_is_risk"
    )
    by_metric: dict[str, Any] = {}
    for metric in QDM_ANALYSIS_METRIC_NAMES:
        values = [_analysis_metric_value(row, metric) for row in rows]
        qdm_threshold = _percentile(values, 0.75)
        qdm_high = [
            value > 0.0 and qdm_threshold is not None and value >= qdm_threshold
            for value in values
        ]
        masks = {
            "qdm_high": qdm_high,
            "small_margin": small_margin,
        }
        enrichment = _enrichment_for_masks(rows, masks)
        qdm_only = _classification_metric(
            rows, metric, risk_direction="higher_is_risk"
        )
        combined = _qdm_plus_margin_metric(rows, metric)
        qdm_enrichment = enrichment["qdm_high"]
        margin_enrichment = enrichment["small_margin"]
        combined_enrichment = enrichment["qdm_high_small_margin"]
        by_metric[metric] = {
            "qdm_threshold": qdm_threshold,
            "margin_small_threshold": margin_threshold,
            "qdm_only": qdm_only,
            "margin_only": margin_only,
            "margin_plus_qdm": combined,
            "top1_flip_enrichment": {
                "qdm_only": qdm_enrichment,
                "margin_only": margin_enrichment,
                "margin_plus_qdm": combined_enrichment,
            },
            "incremental_gain": {
                "auroc_over_best_single": (
                    combined["auroc_top1_flip"]
                    - max(
                        qdm_only["auroc_top1_flip"],
                        margin_only["auroc_top1_flip"],
                    )
                    if combined["auroc_top1_flip"] is not None
                    and qdm_only["auroc_top1_flip"] is not None
                    and margin_only["auroc_top1_flip"] is not None
                    else None
                ),
                "pr_auc_over_best_single": (
                    combined["pr_auc_top1_flip"]
                    - max(
                        qdm_only["pr_auc_top1_flip"],
                        margin_only["pr_auc_top1_flip"],
                    )
                    if combined["pr_auc_top1_flip"] is not None
                    and qdm_only["pr_auc_top1_flip"] is not None
                    and margin_only["pr_auc_top1_flip"] is not None
                    else None
                ),
                "enrichment_over_best_single": (
                    combined_enrichment["enrichment"]
                    - max(
                        qdm_enrichment["enrichment"],
                        margin_enrichment["enrichment"],
                    )
                    if combined_enrichment["enrichment"] is not None
                    and qdm_enrichment["enrichment"] is not None
                    and margin_enrichment["enrichment"] is not None
                    else None
                ),
            },
            "margin_conditioned": _margin_conditioned_qdm(rows, metric),
        }
    return {
        "count": len(rows),
        "margin_only": margin_only,
        "margin_small_threshold": margin_threshold,
        "by_qdm_metric": by_metric,
        "analysis_only": True,
    }


def build_incremental_value_report(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare QDM-only, margin-only, and combined diagnostics."""
    normalized = [dict(row) for row in rows]
    quantized_rows = [
        row for row in normalized if row.get("precision_composition") != "BF16"
    ]
    precision_groups = {
        precision: [
            row
            for row in normalized
            if row.get("precision_composition") == precision
        ]
        for precision in PRECISION_COMPOSITIONS
    }
    context_values = sorted(
        {
            int(row["context_length"])
            for row in quantized_rows
            if row.get("context_length") is not None
        }
    )
    context_groups = {
        str(context): [
            row
            for row in quantized_rows
            if int(row["context_length"]) == context
        ]
        for context in context_values
    }
    return {
        "diagnostic_only": True,
        "definition": (
            "Empirical rank quantiles are used only to form comparable "
            "QDM-high and margin-small analysis masks; no production threshold "
            "or fitted predictor is created."
        ),
        "overall": _incremental_group_stats(quantized_rows),
        "by_precision": {
            precision: _incremental_group_stats(group)
            for precision, group in precision_groups.items()
        },
        "by_context": {
            context: _incremental_group_stats(group)
            for context, group in context_groups.items()
        },
    }


def _exact_group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "available": False, "metrics": {}}
    if any(metric not in rows[0] for metric in EXACT_DRIFT_METRIC_NAMES):
        return {
            "count": len(rows),
            "available": False,
            "metrics": {},
            "reason": "exact oracle fields are absent",
        }
    metrics: dict[str, Any] = {}
    margin_values = [float(row["top1_top2_margin"]) for row in rows]
    margin_threshold = _percentile(margin_values, 0.25)
    margin_only = _classification_metric(
        rows, "top1_margin", risk_direction="lower_is_risk"
    )
    for metric in EXACT_DRIFT_METRIC_NAMES:
        values = [float(row[metric]) for row in rows]
        kl = [float(row["kl_bf16_quantized"]) for row in rows]
        js = [float(row["js_divergence"]) for row in rows]
        flip = [float(bool(row["top1_flip"])) for row in rows]
        exact_threshold = _percentile(values, 0.75)
        exact_high = [
            value > 0.0
            and exact_threshold is not None
            and value >= exact_threshold
            for value in values
        ]
        margin_small = [
            margin_threshold is not None and value <= margin_threshold
            for value in margin_values
        ]
        metric_comparison = _classification_metric(
            rows, metric, risk_direction="higher_is_risk"
        )
        enrichment = _enrichment_for_masks(
            rows,
            {
                "qdm_high": exact_high,
                "small_margin": margin_small,
            },
        )
        combined_enrichment = enrichment["qdm_high_small_margin"]
        exact_enrichment = enrichment["qdm_high"]
        margin_enrichment = enrichment["small_margin"]
        combined = _qdm_plus_margin_metric(rows, metric)
        metrics[metric] = {
            "mean": sum(values) / len(values),
            "distribution": _distribution_stats(values),
            "spearman_vs_kl": _spearman(values, kl),
            "spearman_vs_js": _spearman(values, js),
            "spearman_vs_top1_flip": _spearman(values, flip),
            "metric_comparison": metric_comparison,
            "margin_only": margin_only,
            "margin_plus_exact": combined,
            "top1_flip_enrichment": {
                "exact_only": exact_enrichment,
                "margin_only": margin_enrichment,
                "margin_plus_exact": combined_enrichment,
            },
            "incremental_gain": {
                "auroc_over_best_single": (
                    combined["auroc_top1_flip"]
                    - max(
                        metric_comparison["auroc_top1_flip"],
                        margin_only["auroc_top1_flip"],
                    )
                    if combined["auroc_top1_flip"] is not None
                    and metric_comparison["auroc_top1_flip"] is not None
                    and margin_only["auroc_top1_flip"] is not None
                    else None
                ),
                "pr_auc_over_best_single": (
                    combined["pr_auc_top1_flip"]
                    - max(
                        metric_comparison["pr_auc_top1_flip"],
                        margin_only["pr_auc_top1_flip"],
                    )
                    if combined["pr_auc_top1_flip"] is not None
                    and metric_comparison["pr_auc_top1_flip"] is not None
                    and margin_only["pr_auc_top1_flip"] is not None
                    else None
                ),
                "enrichment_over_best_single": (
                    combined_enrichment["enrichment"]
                    - max(
                        exact_enrichment["enrichment"],
                        margin_enrichment["enrichment"],
                    )
                    if combined_enrichment.get("enrichment") is not None
                    and exact_enrichment.get("enrichment") is not None
                    and margin_enrichment.get("enrichment") is not None
                    else None
                ),
            },
            "margin_conditioned": _margin_conditioned_qdm(rows, metric),
        }
    return {
        "count": len(rows),
        "available": True,
        "metrics": metrics,
        "exact_oracle": True,
        "scope": "BF16_query_with_prefix_source_vs_production_restored_KV",
    }


def _grouped_exact_stats(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    return {
        key: _exact_group_stats(group)
        for key, group in _group_rows_by_key(
            list(rows), lambda row: row.get(field)
        ).items()
    }


def build_exact_drift_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build offline exact-oracle drift diagnostics for validation only."""
    normalized = [dict(row) for row in rows]
    quantized_rows = [
        row for row in normalized if row.get("precision_composition") != "BF16"
    ]
    available = bool(normalized) and all(
        all(metric in row for metric in EXACT_DRIFT_METRIC_NAMES)
        for row in normalized
    )
    if not available:
        return {
            "diagnostic_only": True,
            "available": False,
            "definition": (
                "Exact BF16-query attention oracle fields were not present in "
                "the input rows."
            ),
        }
    return {
        "diagnostic_only": True,
        "available": True,
        "definition": (
            "Exact query-conditioned score/KV restore oracle. It is not a "
            "production certificate and uses no additional model forward."
        ),
        "overall": _exact_group_stats(normalized),
        "quantized_overall": _exact_group_stats(quantized_rows),
        "by_precision": {
            precision: _exact_group_stats(
                [
                    row
                    for row in normalized
                    if row.get("precision_composition") == precision
                ]
            )
            for precision in PRECISION_COMPOSITIONS
        },
        "by_context": _grouped_exact_stats(quantized_rows, "context_length"),
        "by_sample": _grouped_exact_stats(quantized_rows, "sample"),
        "by_exact_worst_layer": _grouped_exact_stats(
            quantized_rows, "exact_worst_layer"
        ),
        "by_exact_worst_kv_head": _grouped_exact_stats(
            quantized_rows, "exact_worst_kv_head"
        ),
        "by_exact_worst_layer_head": _grouped_exact_stats(
            quantized_rows, "exact_worst_layer_identity"
        ),
    }


def _tightness_group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows or any(field not in rows[0] for field in BOUND_TIGHTNESS_METRIC_NAMES):
        return {
            "count": len(rows),
            "available": False,
            "metrics": {},
        }
    metrics = {
        field: {
            "distribution": _distribution_stats(
                [
                    float(row[field])
                    for row in rows
                    if row.get(field) is not None
                ]
            ),
            "finite_count": sum(row.get(field) is not None for row in rows),
        }
        for field in BOUND_TIGHTNESS_METRIC_NAMES
    }
    return {
        "count": len(rows),
        "available": True,
        "metrics": metrics,
        "saturation_rate": sum(
            bool(row.get("saturated", False)) for row in rows
        )
        / len(rows),
        "reference_block_probability_max_abs_error": max(
            float(row.get("reference_block_probability_max_abs_error", 0.0))
            for row in rows
        ),
        "production_raw_tv_bound": _distribution_stats(
            [float(row["production_raw_tv_bound"]) for row in rows]
        ),
        "exact_attention_TV": _distribution_stats(
            [float(row["exact_attention_TV"]) for row in rows]
        ),
        "production_attention_error_bound": _distribution_stats(
            [float(row["production_attention_error_bound"]) for row in rows]
        ),
        "exact_attention_output_error": _distribution_stats(
            [float(row["exact_attention_output_error"]) for row in rows]
        ),
    }


def _grouped_tightness_stats(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    return {
        key: _tightness_group_stats(group)
        for key, group in _group_rows_by_key(
            list(rows), lambda row: row.get(field)
        ).items()
    }


def build_bound_tightness_report(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Decompose production-bound looseness using offline exact observations."""
    normalized = [dict(row) for row in rows]
    quantized_rows = [
        row for row in normalized if row.get("precision_composition") != "BF16"
    ]
    available = bool(normalized) and all(
        all(field in row for field in BOUND_TIGHTNESS_METRIC_NAMES)
        for row in normalized
    )
    if not available:
        return {
            "diagnostic_only": True,
            "available": False,
            "definition": (
                "Exact bound-tightness fields were not present in the input rows."
            ),
        }
    return {
        "diagnostic_only": True,
        "available": True,
        "definition": {
            "score_cauchy": (
                "production block witness score bound divided by exact max "
                "query-conditioned score error"
            ),
            "tv_block_max": (
                "block-max Cauchy raw TV divided by token-level Cauchy raw TV"
            ),
            "tv_transform": (
                "token-level Cauchy raw TV divided by exact attention TV"
            ),
            "tv_bound_exact": (
                "clamped production TV bound divided by exact attention TV"
            ),
            "output_bound_exact": (
                "production attention error bound divided by exact attention "
                "output error"
            ),
        },
        "overall": _tightness_group_stats(normalized),
        "quantized_overall": _tightness_group_stats(quantized_rows),
        "by_precision": {
            precision: _tightness_group_stats(
                [
                    row
                    for row in normalized
                    if row.get("precision_composition") == precision
                ]
            )
            for precision in PRECISION_COMPOSITIONS
        },
        "by_context": _grouped_tightness_stats(quantized_rows, "context_length"),
        "by_sample": _grouped_tightness_stats(quantized_rows, "sample"),
        "by_exact_worst_layer": _grouped_tightness_stats(
            quantized_rows, "exact_worst_layer"
        ),
        "by_exact_worst_kv_head": _grouped_tightness_stats(
            quantized_rows, "exact_worst_kv_head"
        ),
        "by_exact_worst_layer_head": _grouped_tightness_stats(
            quantized_rows, "exact_worst_layer_identity"
        ),
    }


def _group_rows_by_key(
    rows: Sequence[Mapping[str, Any]], key_function: Any
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = key_function(row)
        if key is None:
            continue
        groups.setdefault(str(key), []).append(dict(row))
    return groups


def _paired_token_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    sample = row.get("sample")
    context = row.get("context_length")
    requested_context = row.get("requested_context_length", context)
    step = row.get("step")
    if sample is None or context is None or requested_context is None or step is None:
        raise ValueError(
            "paired precision rows require sample, context_length, "
            "requested_context_length, and step"
        )
    return str(sample), int(context), int(requested_context), int(step)


def _paired_key_payload(key: tuple[str, int, int, int]) -> dict[str, Any]:
    return {
        "sample": key[0],
        "context_length": key[1],
        "requested_context_length": key[2],
        "step": key[3],
    }


def _normalise_precision_id_counts(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(int(key)): int(count) for key, count in value.items()}


def _mixed_block_precision_composition(row: Mapping[str, Any]) -> dict[str, Any]:
    observed_ids = sorted(
        int(value) for value in (row.get("witness_precision_ids") or [])
    )
    observed_names = [
        PRECISION_ID_NAMES.get(value, f"ID_{value}") for value in observed_ids
    ]
    block_counts = _normalise_precision_id_counts(
        row.get("witness_block_precision_id_counts")
    )
    counts_source = "production_qdm_metadata" if block_counts is not None else None
    count_unit = "layer_block_first_kv_head" if block_counts is not None else None

    # Existing real-model artifacts predate block count telemetry. The mixed
    # validation plan is fixed round-robin, so its block composition can be
    # reconstructed from geometry without reimplementing quantization.
    if block_counts is None and row.get("precision_plan_source") == (
        "qdm_validation_round_robin_v1"
    ):
        prefix_length = row.get("prefix_length")
        block_size = row.get("qdm_block_size")
        if prefix_length is not None and block_size:
            block_count = (int(prefix_length) + int(block_size) - 1) // int(block_size)
            cycle = (
                PRECISION_ID_K2V2,
                PRECISION_ID_K4V2,
                PRECISION_ID_K8V4,
                PRECISION_ID_BF16,
            )
            block_counts = {
                str(value): sum(
                    1
                    for index in range(block_count)
                    if cycle[index % len(cycle)] == value
                )
                for value in cycle
            }
            counts_source = "fixed_validation_plan_geometry"
            count_unit = "block"

    named_block_counts = None
    if block_counts is not None:
        named_block_counts = {
            PRECISION_ID_NAMES.get(int(key), f"ID_{key}"): int(value)
            for key, value in block_counts.items()
        }
    return {
        "precision_plan_source": row.get("precision_plan_source"),
        "observed_precision_ids": observed_ids,
        "observed_precision_names": observed_names,
        "block_precision_counts": named_block_counts,
        "block_precision_count_unit": count_unit,
        "block_precision_counts_source": counts_source,
    }


def _validate_paired_group(
    group: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    by_precision: dict[str, list[Mapping[str, Any]]] = {}
    for row in group:
        precision = str(row.get("precision_composition", ""))
        by_precision.setdefault(precision, []).append(row)
    missing = [
        precision
        for precision in PAIRED_REQUIRED_PRECISIONS
        if precision not in by_precision
    ]
    if missing:
        errors.append(f"missing precision rows: {','.join(missing)}")
    duplicates = [
        precision
        for precision, rows in by_precision.items()
        if len(rows) != 1
    ]
    if duplicates:
        errors.append(f"duplicate precision rows: {','.join(sorted(duplicates))}")
    if set(by_precision) - set(PAIRED_REQUIRED_PRECISIONS):
        errors.append("unknown precision composition in paired group")

    for row in group:
        if not row.get("teacher_forced", False):
            errors.append("teacher_forced is false")
        if row.get("free_running_ground_truth", False):
            errors.append("free-running ground truth is not allowed")
        if any(
            not row.get(field, False)
            for field in ("prefix_aligned", "suffix_aligned", "target_aligned")
        ):
            errors.append("token alignment flag is false")
        if row.get("exact_oracle") is not True:
            errors.append("exact oracle marker is missing")
        for field in PAIRED_ALIGNMENT_FIELDS:
            if not row.get(field):
                errors.append(f"missing {field}")
        for field in (
            *PAIRED_EXACT_METRICS,
            "max_attention_error",
            "kl_bf16_quantized",
            "js_divergence",
            "top1_top2_margin",
            "quantized_top1_top2_margin",
            "reference_top1_token",
            "reference_top1_top2_margin",
            "topK_entropy",
        ):
            if field not in row or row[field] is None:
                errors.append(f"missing paired metric: {field}")
            elif isinstance(row[field], (int, float)) and not math.isfinite(
                float(row[field])
            ):
                errors.append(f"non-finite paired metric: {field}")

    for field in PAIRED_ALIGNMENT_FIELDS:
        values = {row.get(field) for row in group}
        if len(values) != 1:
            errors.append(f"{field} differs across precision rows")
    for field in (
        "prefix_token_id_hash",
        "suffix_input_token_id_hash",
        "target_token_id_hash",
    ):
        values = {row.get(field) for row in group if field in row}
        if values and len(values) != 1:
            errors.append(f"{field} differs across precision rows")
    for field in ("suffix_input_token_id", "target_token_id", "reference_top1_token"):
        values = {row.get(field) for row in group}
        if len(values) != 1:
            errors.append(f"{field} differs across precision rows")
    for field in ("reference_top1_top2_margin", "topK_entropy"):
        values = [float(row[field]) for row in group if field in row]
        if values and max(values) - min(values) > PAIRED_NUMERICAL_TOLERANCE:
            errors.append(f"{field} differs across precision rows")

    mixed = by_precision.get("MIXED", [])
    if mixed and not mixed[0].get("witness_precision_ids"):
        errors.append("MIXED actual precision IDs are missing")
    return sorted(set(errors))


def _paired_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "precision_composition": str(row["precision_composition"]),
        "precision_plan_source": row.get("precision_plan_source"),
        "kl_precision": float(row["kl_bf16_quantized"]),
        "js_precision": float(row["js_divergence"]),
        "top1_flip_precision": bool(row["top1_flip"]),
        "reference_top1_token": int(row["reference_top1_token"]),
        "reference_margin": float(row["reference_top1_top2_margin"]),
        "quantized_margin": float(row["quantized_top1_top2_margin"]),
        "topK_entropy": float(row["topK_entropy"]),
        "qdm_attention_error": float(row["max_attention_error"]),
        # Sensitivity fields are optional for the Phase 1.5 paired report;
        # real Phase 2.2 rows populate them from the offline trace.
        "logit_delta_l2": float(row.get("logit_delta_l2", 0.0)),
        "margin_abs_delta": float(row.get("margin_abs_delta", 0.0)),
        "witness_max_k_error": float(row.get("witness_max_k_error", 0.0)),
        "witness_max_v_error": float(row.get("witness_max_v_error", 0.0)),
        "witness_max_v_norm": float(row.get("witness_max_v_norm", 0.0)),
        "witness_precision_ids": [
            int(value) for value in (row.get("witness_precision_ids") or [])
        ],
        "witness_precision_id_counts": _normalise_precision_id_counts(
            row.get("witness_precision_id_counts")
        ),
        "witness_block_precision_id_counts": _normalise_precision_id_counts(
            row.get("witness_block_precision_id_counts")
        ),
    }
    for metric in PAIRED_EXACT_METRICS:
        result[metric] = float(row[metric])
    if result["precision_composition"] == "MIXED":
        result["actual_block_precision_composition"] = (
            _mixed_block_precision_composition(row)
        )
    return result


def _paired_delta(
    low_precision: str,
    high_precision: str,
    projected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    low = projected[low_precision]
    high = projected[high_precision]
    physical_metrics = (*PAIRED_EXACT_METRICS, "qdm_attention_error")
    physical = {
        metric: float(low[metric]) - float(high[metric])
        for metric in physical_metrics
    }
    logit = {
        "delta_KL": float(low["kl_precision"]) - float(high["kl_precision"]),
        "delta_JS": float(low["js_precision"]) - float(high["js_precision"]),
        "delta_margin": float(low["quantized_margin"])
        - float(high["quantized_margin"]),
        "delta_logit_delta_l2": float(low["logit_delta_l2"])
        - float(high["logit_delta_l2"]),
        "delta_margin_abs": float(low["margin_abs_delta"])
        - float(high["margin_abs_delta"]),
        "delta_top1_flip": int(bool(low["top1_flip_precision"]))
        - int(bool(high["top1_flip_precision"])),
    }
    return {
        "direction": "low_precision_minus_high_precision",
        "low_precision": low_precision,
        "high_precision": high_precision,
        "physical": physical,
        "logit": logit,
        "delta_KL": logit["delta_KL"],
        "delta_JS": logit["delta_JS"],
        "delta_margin": logit["delta_margin"],
    }


def _assign_paired_margin_buckets(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [float(record["reference_margin"]) for record in records]
    labels = ("Q1", "Q2", "Q3", "Q4")
    ordered = sorted(range(len(records)), key=lambda index: (values[index], index))
    for rank, index in enumerate(ordered):
        records[index]["margin_bucket"] = labels[
            min(len(labels) - 1, rank * len(labels) // len(records))
        ]
    return {
        "quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
        "boundaries": [_percentile(values, index / 4) for index in range(5)],
        "assignment": "stable_empirical_rank_quantiles_for_analysis_only",
        "risk_threshold_fitting": False,
    }


def _paired_monotonicity_stats(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    transitions = tuple(
        (f"{left}_to_{right}", left, right)
        for left, right in zip(
            PAIRED_PRECISION_ORDER[:-1],
            PAIRED_PRECISION_ORDER[1:],
            strict=True,
        )
    )
    result: dict[str, Any] = {"count": len(records), "metrics": {}}
    for metric in PAIRED_MONOTONIC_METRICS:
        transition_counts = {
            name: {"concordant": 0, "count": 0}
            for name, _, _ in transitions
        }
        monotonic_count = 0
        within_scores: list[float] = []
        for record in records:
            sequence = [
                float(record["precision"][precision][metric])
                for precision in PAIRED_PRECISION_ORDER
            ]
            token_monotonic = True
            for (name, _, _), left, right in zip(
                transitions,
                sequence[:-1],
                sequence[1:],
                strict=True,
            ):
                concordant = right + PAIRED_NUMERICAL_TOLERANCE >= left
                transition_counts[name]["count"] += 1
                transition_counts[name]["concordant"] += int(concordant)
                token_monotonic = token_monotonic and concordant
            monotonic_count += int(token_monotonic)
            score = _spearman(list(range(len(sequence))), sequence)
            if score is not None and math.isfinite(score):
                within_scores.append(score)
        transition_rates = {
            name: {
                **counts,
                "rate": (
                    counts["concordant"] / counts["count"]
                    if counts["count"]
                    else None
                ),
            }
            for name, counts in transition_counts.items()
        }
        result["metrics"][metric] = {
            "order": list(PAIRED_PRECISION_ORDER),
            "pairwise_concordance_rate": (
                sum(item["concordant"] for item in transition_counts.values())
                / sum(item["count"] for item in transition_counts.values())
                if records
                else None
            ),
            "pairwise_concordance_by_transition": transition_rates,
            "monotonic_token_fraction": (
                monotonic_count / len(records) if records else None
            ),
            "monotonic_token_count": monotonic_count,
            "within_token_spearman": {
                "count": len(within_scores),
                "mean": (
                    sum(within_scores) / len(within_scores)
                    if within_scores
                    else None
                ),
                "p10": _percentile(within_scores, 0.10),
                "p50": _percentile(within_scores, 0.50),
                "p90": _percentile(within_scores, 0.90),
                "positive_fraction": (
                    sum(score > 0.0 for score in within_scores) / len(within_scores)
                    if within_scores
                    else None
                ),
            },
        }
    return result


def _paired_sign(value: float) -> int:
    if value > PAIRED_NUMERICAL_TOLERANCE:
        return 1
    if value < -PAIRED_NUMERICAL_TOLERANCE:
        return -1
    return 0


def _paired_relation(
    records: Sequence[Mapping[str, Any]],
    transition: str,
    physical_metric: str,
    outcome_delta: str,
) -> dict[str, Any]:
    pairs = []
    for record in records:
        delta = record["deltas"].get(transition)
        if delta is None:
            continue
        physical = float(delta["physical"][physical_metric])
        outcome = float(delta["logit"][outcome_delta])
        if math.isfinite(physical) and math.isfinite(outcome):
            pairs.append((physical, outcome))
    physical_values = [pair[0] for pair in pairs]
    outcome_values = [pair[1] for pair in pairs]
    signs = [
        _paired_sign(physical) == _paired_sign(outcome) for physical, outcome in pairs
    ]
    absolute_physical = sorted(
        (abs(value) for value in physical_values), reverse=True
    )
    top_count = max(1, len(absolute_physical) // 20) if absolute_physical else 0
    total_absolute = sum(absolute_physical)
    ordered_pairs = sorted(pairs, key=lambda pair: abs(pair[0]), reverse=True)
    trimmed_pairs = ordered_pairs[top_count:] if len(pairs) >= 20 else pairs
    trimmed_signs = [
        _paired_sign(physical) == _paired_sign(outcome)
        for physical, outcome in trimmed_pairs
    ]
    return {
        "count": len(pairs),
        "physical_delta": _distribution_stats(physical_values),
        "outcome_delta": _distribution_stats(outcome_values),
        "spearman": _spearman(physical_values, outcome_values),
        "sign_concordance_rate": (
            sum(signs) / len(signs) if signs else None
        ),
        "positive_physical_fraction": (
            sum(_paired_sign(value) > 0 for value in physical_values) / len(pairs)
            if pairs
            else None
        ),
        "positive_outcome_fraction": (
            sum(_paired_sign(value) > 0 for value in outcome_values) / len(pairs)
            if pairs
            else None
        ),
        "both_positive_fraction": (
            sum(
                _paired_sign(physical) > 0 and _paired_sign(outcome) > 0
                for physical, outcome in pairs
            )
            / len(pairs)
            if pairs
            else None
        ),
        "top_5_percent_abs_physical_delta_share": (
            sum(absolute_physical[:top_count]) / total_absolute
            if total_absolute and top_count
            else None
        ),
        "trimmed_5_percent_by_physical_magnitude": {
            "count": len(trimmed_pairs),
            "spearman": _spearman(
                [pair[0] for pair in trimmed_pairs],
                [pair[1] for pair in trimmed_pairs],
            ),
            "sign_concordance_rate": (
                sum(trimmed_signs) / len(trimmed_signs)
                if trimmed_signs
                else None
            ),
        },
    }


def _paired_transition_summary(
    records: Sequence[Mapping[str, Any]],
    transitions: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(records), "transitions": {}}
    for name, _, _ in transitions:
        result["transitions"][name] = {
            "metrics": {
                metric: {
                    "vs_delta_KL": _paired_relation(
                        records, name, metric, "delta_KL"
                    ),
                    "vs_delta_JS": _paired_relation(
                        records, name, metric, "delta_JS"
                    ),
                }
                for metric in (*PAIRED_EXACT_METRICS, "qdm_attention_error")
            }
        }
    return result


def _paired_grouped_summary(
    records: Sequence[Mapping[str, Any]],
    field: str,
    transitions: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = record.get(field)
        if key is not None:
            groups.setdefault(str(key), []).append(record)
    return {
        key: _paired_transition_summary(group, transitions)
        for key, group in sorted(groups.items())
    }


def _paired_grouped_monotonicity(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = record.get(field)
        if key is not None:
            groups.setdefault(str(key), []).append(record)
    return {
        key: _paired_monotonicity_stats(group)
        for key, group in sorted(groups.items())
    }


def _paired_relation_is_positive(metric_summary: Mapping[str, Any]) -> bool:
    if int(metric_summary.get("count", 0)) < PAIRED_MIN_GROUP_ROWS:
        return False
    trimmed = metric_summary.get("trimmed_5_percent_by_physical_magnitude", {})
    return bool(
        metric_summary.get("spearman") is not None
        and metric_summary["spearman"] > 0.0
        and metric_summary.get("sign_concordance_rate") is not None
        and metric_summary["sign_concordance_rate"] > 0.5
        and trimmed.get("spearman") is not None
        and trimmed["spearman"] > 0.0
        and trimmed.get("sign_concordance_rate") is not None
        and trimmed["sign_concordance_rate"] > 0.5
    )


def _paired_group_is_positive(
    group_summary: Mapping[str, Any],
    metric: str,
) -> bool:
    transitions = group_summary.get("transitions", {})
    return all(
        _paired_relation_is_positive(
            transitions[transition]["metrics"][metric][outcome]
        )
        for transition, _, _ in PAIRED_DELTA_TRANSITIONS
        for outcome in ("vs_delta_KL", "vs_delta_JS")
        if transition in transitions
    ) and all(
        transition in transitions for transition, _, _ in PAIRED_DELTA_TRANSITIONS
    )


def _paired_support_summary(
    groups: Mapping[str, Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    supported = {
        key: value
        for key, value in groups.items()
        if int(value.get("count", 0)) >= PAIRED_MIN_GROUP_ROWS
    }
    positive = {
        key: value
        for key, value in supported.items()
        if _paired_group_is_positive(value, metric)
    }
    return {
        "observed_groups": len(groups),
        "supported_groups": len(supported),
        "positive_groups": len(positive),
        "minimum_group_rows": PAIRED_MIN_GROUP_ROWS,
        "positive_group_keys": sorted(positive),
    }


def _paired_composition_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    aggregate_blocks: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    count_units: dict[str, int] = {}
    for record in records:
        composition = record["precision"]["MIXED"].get(
            "actual_block_precision_composition", {}
        )
        signature = json.dumps(composition, sort_keys=True)
        counts[signature] = counts.get(signature, 0) + 1
        source = str(composition.get("block_precision_counts_source"))
        source_counts[source] = source_counts.get(source, 0) + 1
        unit = str(composition.get("block_precision_count_unit"))
        count_units[unit] = count_units.get(unit, 0) + 1
        for name, value in (
            composition.get("block_precision_counts") or {}
        ).items():
            aggregate_blocks[str(name)] = aggregate_blocks.get(str(name), 0) + int(
                value
            )
    return {
        "record_count": len(records),
        "unique_compositions": [
            {"composition": json.loads(signature), "record_count": count}
            for signature, count in sorted(counts.items())
        ],
        "aggregate_block_precision_counts": aggregate_blocks,
        "composition_source_counts": source_counts,
        "composition_count_unit_counts": count_units,
    }


def build_paired_precision_analysis(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare production precision paths on the exact same token rows."""
    normalized = [dict(row) for row in rows]
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    grouping_errors: list[str] = []
    for index, row in enumerate(normalized):
        try:
            key = _paired_token_key(row)
        except (TypeError, ValueError) as error:
            grouping_errors.append(f"row {index}: {error}")
            continue
        grouped.setdefault(key, []).append(row)

    invalid_groups: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        errors = _validate_paired_group(group)
        if errors:
            invalid_groups.append({"key": _paired_key_payload(key), "errors": errors})
            continue
        by_precision = {
            str(row["precision_composition"]): row for row in group
        }
        projected = {
            precision: _paired_projection(by_precision[precision])
            for precision in PAIRED_REQUIRED_PRECISIONS
        }
        record = {
            **_paired_key_payload(key),
            "teacher_forced": True,
            "free_running_ground_truth": False,
            "alignment_hashes": {
                field: by_precision["BF16"][field]
                for field in PAIRED_ALIGNMENT_FIELDS
            },
            "prefix_token_id_hash": by_precision["BF16"].get(
                "prefix_token_id_hash"
            ),
            "suffix_input_token_id": int(by_precision["BF16"]["suffix_input_token_id"]),
            "target_token_id": int(by_precision["BF16"]["target_token_id"]),
            "reference_top1_token": int(
                by_precision["BF16"]["reference_top1_token"]
            ),
            "reference_margin": float(
                by_precision["BF16"]["reference_top1_top2_margin"]
            ),
            "precision": projected,
            "deltas": {},
        }
        for name, high, low in PAIRED_DELTA_TRANSITIONS:
            record["deltas"][name] = _paired_delta(low, high, projected)
        for name, high, low in PAIRED_MIXED_TRANSITIONS:
            record["deltas"][name] = _paired_delta(low, high, projected)
        records.append(record)

    integrity = {
        "candidate_row_count": len(normalized),
        "candidate_token_group_count": len(grouped),
        "paired_token_count": len(records),
        "invalid_token_group_count": len(invalid_groups),
        "grouping_error_count": len(grouping_errors),
        "required_precisions": list(PAIRED_REQUIRED_PRECISIONS),
        "alignment_fields": list(PAIRED_ALIGNMENT_FIELDS),
        "fail_closed": bool(grouping_errors or invalid_groups),
        "errors": grouping_errors[:20],
        "invalid_groups": invalid_groups[:20],
    }
    if grouping_errors or invalid_groups:
        summary = {
            "analysis_version": PAIRED_ANALYSIS_VERSION,
            "available": False,
            "conclusion": "INCONCLUSIVE",
            "definition": (
                "Paired precision analysis requires one aligned BF16/K8V4/K4V2/"
                "K2V2/MIXED row for every sample/context/step token."
            ),
            "integrity": integrity,
            "criteria": {
                "paired_alignment_valid": False,
                "stable_physical_signal": False,
            },
            "limitations": [
                "Pairing failed closed; no partial precision groups were analyzed."
            ],
        }
        return {"records": [], "summary": summary}

    margin_definition = _assign_paired_margin_buckets(records)
    monotonicity = {
        "overall": _paired_monotonicity_stats(records),
        "by_prompt": _paired_grouped_monotonicity(records, "sample"),
        "by_context": _paired_grouped_monotonicity(records, "context_length"),
        "by_margin_bucket": _paired_grouped_monotonicity(records, "margin_bucket"),
        "margin_bucket_definition": margin_definition,
    }
    delta_analysis = {
        "overall": _paired_transition_summary(records, PAIRED_DELTA_TRANSITIONS),
        "by_prompt": _paired_grouped_summary(
            records, "sample", PAIRED_DELTA_TRANSITIONS
        ),
        "by_context": _paired_grouped_summary(
            records, "context_length", PAIRED_DELTA_TRANSITIONS
        ),
        "by_margin_bucket": _paired_grouped_summary(
            records, "margin_bucket", PAIRED_DELTA_TRANSITIONS
        ),
        "margin_bucket_definition": margin_definition,
    }
    mixed_analysis = {
        "monotonicity_required": False,
        "definition": (
            "MIXED is compared independently against BF16 and K2V2; it is not "
            "inserted into the homogeneous precision monotonicity chain."
        ),
        "overall": _paired_transition_summary(records, PAIRED_MIXED_TRANSITIONS),
        "by_prompt": _paired_grouped_summary(
            records, "sample", PAIRED_MIXED_TRANSITIONS
        ),
        "by_context": _paired_grouped_summary(
            records, "context_length", PAIRED_MIXED_TRANSITIONS
        ),
        "by_margin_bucket": _paired_grouped_summary(
            records, "margin_bucket", PAIRED_MIXED_TRANSITIONS
        ),
        "actual_block_precision_composition": _paired_composition_summary(records),
    }

    primary_metric = "exact_attention_output_error"
    overall_positive = _paired_group_is_positive(
        delta_analysis["overall"], primary_metric
    )
    prompt_support = _paired_support_summary(
        delta_analysis["by_prompt"], primary_metric
    )
    context_support = _paired_support_summary(
        delta_analysis["by_context"], primary_metric
    )
    margin_support = _paired_support_summary(
        delta_analysis["by_margin_bucket"], primary_metric
    )
    stable = bool(
        overall_positive
        and prompt_support["positive_groups"] >= PAIRED_MIN_PROMPT_SUPPORT
        and prompt_support["supported_groups"] >= PAIRED_MIN_PROMPT_SUPPORT
        and context_support["supported_groups"] == context_support["positive_groups"]
        and context_support["supported_groups"] >= 2
        and margin_support["positive_groups"] >= 3
        and margin_support["supported_groups"] >= 3
    )
    overall_transitions = delta_analysis["overall"]["transitions"]
    negative_overall = all(
        all(
            relation.get("spearman") is not None
            and relation["spearman"] <= 0.0
            and (relation.get("sign_concordance_rate") or 0.0) <= 0.5
            for relation in (
                overall_transitions[transition]["metrics"][primary_metric][outcome]
                for outcome in ("vs_delta_KL", "vs_delta_JS")
            )
        )
        for transition, _, _ in PAIRED_DELTA_TRANSITIONS
    )
    rejected = bool(
        not stable
        and negative_overall
        and context_support["positive_groups"] == 0
    )
    conclusion = (
        "PHYSICAL_SIGNAL_VALIDATED"
        if stable
        else "PHYSICAL_SIGNAL_REJECTED"
        if rejected
        else "INCONCLUSIVE"
    )
    summary = {
        "analysis_version": PAIRED_ANALYSIS_VERSION,
        "available": True,
        "conclusion": conclusion,
        "definition": (
            "Teacher-forced within-token counterfactual pairing. Deltas are "
            "low-precision minus high-precision on identical sample/context/step "
            "rows; no free-running trajectory is used."
        ),
        "integrity": integrity,
        "paired_token_count": len(records),
        "precision_order": list(PAIRED_PRECISION_ORDER),
        "monotonicity": monotonicity,
        "delta_analysis": delta_analysis,
        "mixed_analysis": mixed_analysis,
        "criteria": {
            "paired_alignment_valid": True,
            "primary_metric": primary_metric,
            "stable_physical_signal": stable,
            "overall_primary_relation_positive": overall_positive,
            "prompt_support": prompt_support,
            "context_support": context_support,
            "margin_bucket_support": margin_support,
            "negative_physical_signal": rejected,
            "minimum_prompt_support": PAIRED_MIN_PROMPT_SUPPORT,
            "minimum_group_rows": PAIRED_MIN_GROUP_ROWS,
            "trimmed_extreme_diagnostic": (
                "positive relation requires the same direction after removing "
                "the largest 5% physical deltas when group size permits"
            ),
        },
        "limitations": [
            "Paired deltas establish controlled association, not a causal proof "
            "of downstream token error.",
            "Empirical margin buckets are analysis-only and are not production "
            "thresholds.",
            "Exact BF16-query metrics remain offline oracles and are not runtime "
            "telemetry."
        ],
    }
    return {"records": records, "summary": summary}


def _sensitivity_projection(row: Mapping[str, Any]) -> dict[str, Any] | None:
    available = row.get("downstream_sensitivity_available")
    if available is False:
        return None
    values: dict[str, float] = {}
    for field in SENSITIVITY_REQUIRED_FIELDS:
        value = row.get(field)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        values[field] = numeric
    values["sensitivity_signed_error_abs"] = abs(values["sensitivity_signed_error"])
    for field in (
        "directional_cancellation_ratio",
        "hidden_amplified_transition_fraction",
        "hidden_attenuated_transition_fraction",
        "hidden_peak_to_final_ratio",
    ):
        value = row.get(field)
        if value is not None:
            numeric = float(value)
            if not math.isfinite(numeric):
                return None
            values[field] = numeric
    return {
        **values,
        "kl_precision": float(row["kl_bf16_quantized"]),
        "js_precision": float(row["js_divergence"]),
        "top1_flip_precision": bool(row["top1_flip"]),
        "reference_margin": float(row["reference_top1_top2_margin"]),
        "quantized_margin": float(row["quantized_top1_top2_margin"]),
    }


def _sensitivity_delta(
    low_precision: str,
    high_precision: str,
    projected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    low = projected[low_precision]
    high = projected[high_precision]
    features = {
        feature: float(low[feature]) - float(high[feature])
        for feature in SENSITIVITY_FEATURES
    }
    return {
        "direction": "low_precision_minus_high_precision",
        "low_precision": low_precision,
        "high_precision": high_precision,
        "features": features,
        "delta_KL": float(low["kl_precision"]) - float(high["kl_precision"]),
        "delta_JS": float(low["js_precision"]) - float(high["js_precision"]),
        "delta_margin": float(low["quantized_margin"])
        - float(high["quantized_margin"]),
        "delta_logit_delta_l2": float(low["logit_delta_l2"])
        - float(high["logit_delta_l2"]),
        "delta_margin_abs": float(low["margin_abs_delta"])
        - float(high["margin_abs_delta"]),
        "delta_top1_flip": int(bool(low["top1_flip_precision"]))
        - int(bool(high["top1_flip_precision"])),
    }


def _sensitivity_relation(
    records: Sequence[Mapping[str, Any]],
    transition: str,
    feature: str,
    outcome_delta: str,
) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for record in records:
        delta = record.get("deltas", {}).get(transition)
        if not isinstance(delta, Mapping):
            continue
        physical = delta.get("features", {}).get(feature)
        outcome = delta.get(outcome_delta)
        if physical is None or outcome is None:
            continue
        physical_value = float(physical)
        outcome_value = float(outcome)
        if math.isfinite(physical_value) and math.isfinite(outcome_value):
            pairs.append((physical_value, outcome_value))
    physical_values = [pair[0] for pair in pairs]
    outcome_values = [pair[1] for pair in pairs]
    signs = [
        _paired_sign(physical) == _paired_sign(outcome)
        for physical, outcome in pairs
    ]
    ordered = sorted(pairs, key=lambda pair: abs(pair[0]), reverse=True)
    trim_count = max(1, len(ordered) // 20) if len(ordered) >= 20 else 0
    trimmed = ordered[trim_count:] if trim_count else ordered
    trimmed_signs = [
        _paired_sign(physical) == _paired_sign(outcome)
        for physical, outcome in trimmed
    ]
    return {
        "feature": feature,
        "outcome_delta": outcome_delta,
        "count": len(pairs),
        "feature_delta": _distribution_stats(physical_values),
        "outcome_delta_distribution": _distribution_stats(outcome_values),
        "spearman": _spearman(physical_values, outcome_values),
        "sign_concordance_rate": (
            sum(signs) / len(signs) if signs else None
        ),
        "positive_feature_fraction": (
            sum(_paired_sign(value) > 0 for value in physical_values) / len(pairs)
            if pairs
            else None
        ),
        "positive_outcome_fraction": (
            sum(_paired_sign(value) > 0 for value in outcome_values) / len(pairs)
            if pairs
            else None
        ),
        "trimmed_5_percent_by_feature_magnitude": {
            "count": len(trimmed),
            "spearman": _spearman(
                [pair[0] for pair in trimmed], [pair[1] for pair in trimmed]
            ),
            "sign_concordance_rate": (
                sum(trimmed_signs) / len(trimmed_signs)
                if trimmed_signs
                else None
            ),
        },
    }


def _sensitivity_direct_relation(
    records: Sequence[Mapping[str, Any]], feature: str, target: str
) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for record in records:
        feature_value = record.get(feature)
        target_value = record.get(target)
        if feature_value is None or target_value is None:
            continue
        feature_numeric = float(feature_value)
        target_numeric = float(target_value)
        if math.isfinite(feature_numeric) and math.isfinite(target_numeric):
            pairs.append((feature_numeric, target_numeric))
    return {
        "feature": feature,
        "target": target,
        "count": len(pairs),
        "spearman": _spearman(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        ),
    }


def _sensitivity_transition_summary(
    records: Sequence[Mapping[str, Any]],
    transitions: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    return {
        "count": len(records),
        "transitions": {
            name: {
                "metrics": {
                    feature: {
                        "vs_delta_KL": _sensitivity_relation(
                            records, name, feature, "delta_KL"
                        ),
                        "vs_delta_JS": _sensitivity_relation(
                            records, name, feature, "delta_JS"
                        ),
                        "vs_delta_top1_flip": _sensitivity_relation(
                            records, name, feature, "delta_top1_flip"
                        ),
                        "vs_delta_logit_delta_l2": _sensitivity_relation(
                            records, name, feature, "delta_logit_delta_l2"
                        ),
                        "vs_delta_margin_abs": _sensitivity_relation(
                            records, name, feature, "delta_margin_abs"
                        ),
                    }
                    for feature in SENSITIVITY_FEATURES
                }
            }
            for name, _, _ in transitions
        },
    }


def _sensitivity_grouped_summary(
    records: Sequence[Mapping[str, Any]],
    field: str,
    transitions: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = record.get(field)
        if key is not None:
            groups.setdefault(str(key), []).append(record)
    return {
        key: _sensitivity_transition_summary(group, transitions)
        for key, group in sorted(groups.items())
    }


def _sensitivity_conditioned_stats(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    enriched_rows = []
    for row in rows:
        projection = _sensitivity_projection(row)
        if projection is None:
            continue
        enriched = dict(row)
        enriched.update(projection)
        enriched_rows.append(enriched)
    return {
        precision: {
            "count": len(group),
            "metrics": {
                feature: {
                    "classification": _classification_metric(
                        group, feature, risk_direction="higher_is_risk"
                    ),
                    "ground_truth": {
                        target: _sensitivity_direct_relation(group, feature, target)
                        for target in ("logit_delta_l2", "margin_abs_delta")
                    },
                    "quantile_analysis": _quantile_analysis(group, feature),
                }
                for feature in SENSITIVITY_FEATURES
            },
        }
        for precision in PAIRED_REQUIRED_PRECISIONS
        for group in [
            [
                row
                for row in enriched_rows
                if row.get("precision_composition") == precision
            ]
        ]
    }


def _sensitivity_relation_is_positive(relation: Mapping[str, Any]) -> bool:
    trimmed = relation.get("trimmed_5_percent_by_feature_magnitude", {})
    return bool(
        int(relation.get("count", 0)) >= SENSITIVITY_MIN_GROUP_ROWS
        and relation.get("spearman") is not None
        and float(relation["spearman"]) > 0.0
        and relation.get("sign_concordance_rate") is not None
        and float(relation["sign_concordance_rate"]) > 0.5
        and trimmed.get("spearman") is not None
        and float(trimmed["spearman"]) > 0.0
        and trimmed.get("sign_concordance_rate") is not None
        and float(trimmed["sign_concordance_rate"]) > 0.5
    )


def _sensitivity_feature_beats(
    metric_summary: Mapping[str, Any],
    physical_feature: str = "physical_norm_only",
    sensitivity_feature: str = "sensitivity_weighted_error",
) -> dict[str, Any]:
    physical = metric_summary.get(physical_feature, {})
    sensitivity = metric_summary.get(sensitivity_feature, {})
    result: dict[str, Any] = {
        "physical_feature": physical_feature,
        "sensitivity_feature": sensitivity_feature,
    }
    for outcome in SENSITIVITY_OUTCOMES:
        physical_relation = physical.get(outcome, {})
        sensitivity_relation = sensitivity.get(outcome, {})
        physical_spearman = physical_relation.get("spearman")
        sensitivity_spearman = sensitivity_relation.get("spearman")
        physical_sign = physical_relation.get("sign_concordance_rate")
        sensitivity_sign = sensitivity_relation.get("sign_concordance_rate")
        result[outcome] = {
            "physical_spearman": physical_spearman,
            "sensitivity_spearman": sensitivity_spearman,
            "spearman_gain": (
                float(sensitivity_spearman) - float(physical_spearman)
                if physical_spearman is not None and sensitivity_spearman is not None
                else None
            ),
            "physical_sign_concordance": physical_sign,
            "sensitivity_sign_concordance": sensitivity_sign,
            "sensitivity_beats_physical": bool(
                physical_spearman is not None
                and sensitivity_spearman is not None
                and float(sensitivity_spearman)
                > float(physical_spearman) + SENSITIVITY_NUMERICAL_TOLERANCE
                and physical_sign is not None
                and sensitivity_sign is not None
                and float(sensitivity_sign) >= float(physical_sign)
            ),
        }
    return result


def _sensitivity_group_is_improved(group: Mapping[str, Any]) -> bool:
    transitions = group.get("transitions", {})
    for transition, _, _ in SENSITIVITY_TRANSITIONS[1:]:
        transition_summary = transitions.get(transition)
        if not isinstance(transition_summary, Mapping):
            return False
        comparison = _sensitivity_feature_beats(
            transition_summary.get("metrics", {})
        )
        if not all(
            comparison.get(outcome, {}).get("sensitivity_beats_physical", False)
            for outcome in ("vs_delta_KL", "vs_delta_JS")
        ):
            return False
    return True


def _sensitivity_group_support(
    groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    supported = {
        key: value
        for key, value in groups.items()
        if int(value.get("count", 0)) >= SENSITIVITY_MIN_GROUP_ROWS
    }
    positive = {
        key: value
        for key, value in supported.items()
        if _sensitivity_group_is_improved(value)
    }
    return {
        "observed_groups": len(groups),
        "supported_groups": len(supported),
        "improved_groups": len(positive),
        "improved_group_keys": sorted(positive),
        "minimum_group_rows": SENSITIVITY_MIN_GROUP_ROWS,
    }


def _sensitivity_cross_layer_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = (
        "directional_cancellation_ratio",
        "hidden_amplified_transition_fraction",
        "hidden_attenuated_transition_fraction",
        "hidden_peak_to_final_ratio",
    )
    return {
        field: _distribution_stats(
            [float(row[field]) for row in rows if row.get(field) is not None]
        )
        for field in fields
    }


def _sensitivity_layer_values(row: Mapping[str, Any]) -> dict[int, dict[str, float]]:
    trace = row.get("_layer_drift_trace")
    if not isinstance(trace, Mapping):
        return {}
    layers = trace.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        return {}
    values: dict[int, dict[str, float]] = {}
    for item in layers:
        if not isinstance(item, Mapping):
            continue
        try:
            layer = int(item["layer"])
            physical = float(item["delta_attn_norm"])
            directional = float(item["directional_error"])
            signed = float(item["signed_directional_error"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (physical, directional, signed)):
            continue
        values[layer] = {
            "physical_norm_only": physical,
            "sensitivity_weighted_error": directional,
            "sensitivity_signed_error_abs": abs(signed),
        }
    return values


def _sensitivity_layer_conditioned_summary(
    records: Sequence[Mapping[str, Any]],
    transitions: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_precision = record.get("layer_projection", {})
        layer_ids = {
            layer
            for precision_values in by_precision.values()
            for layer in precision_values
        }
        for layer in layer_ids:
            layer_record: dict[str, Any] = {
                "sample": record.get("sample"),
                "context_length": record.get("context_length"),
                "margin_bucket": record.get("margin_bucket"),
                "deltas": {},
            }
            for name, high, low in transitions:
                high_values = by_precision.get(high, {}).get(layer)
                low_values = by_precision.get(low, {}).get(layer)
                base_delta = record.get("deltas", {}).get(name, {})
                if not isinstance(high_values, Mapping) or not isinstance(
                    low_values, Mapping
                ):
                    continue
                layer_record["deltas"][name] = {
                    "features": {
                        feature: float(low_values[feature])
                        - float(high_values[feature])
                        for feature in (
                            "physical_norm_only",
                            "sensitivity_weighted_error",
                            "sensitivity_signed_error_abs",
                        )
                        if feature in high_values and feature in low_values
                    },
                    "delta_KL": base_delta.get("delta_KL"),
                    "delta_JS": base_delta.get("delta_JS"),
                    "delta_top1_flip": base_delta.get("delta_top1_flip"),
                    "delta_logit_delta_l2": base_delta.get(
                        "delta_logit_delta_l2"
                    ),
                    "delta_margin_abs": base_delta.get("delta_margin_abs"),
                }
            grouped.setdefault(int(layer), []).append(layer_record)

    return {
        str(layer): {
            "count": len(layer_records),
            "transitions": _sensitivity_transition_summary(
                layer_records, transitions
            )["transitions"],
        }
        for layer, layer_records in sorted(grouped.items())
    }


def build_downstream_sensitivity_analysis(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Analyze offline margin-gradient projections on the same paired tokens."""
    normalized = [dict(row) for row in rows]
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    grouping_errors: list[str] = []
    for index, row in enumerate(normalized):
        try:
            key = _paired_token_key(row)
        except (TypeError, ValueError) as error:
            grouping_errors.append(f"row {index}: {error}")
            continue
        grouped.setdefault(key, []).append(row)

    invalid_groups: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        errors = _validate_paired_group(group)
        for row in group:
            if _sensitivity_projection(row) is None:
                errors.append("downstream sensitivity oracle fields are missing")
            if not isinstance(row.get("_layer_drift_trace"), Mapping):
                errors.append("layer drift trace is missing")
        if errors:
            invalid_groups.append(
                {"key": _paired_key_payload(key), "errors": sorted(set(errors))}
            )
            continue
        by_precision = {
            str(row["precision_composition"]): row for row in group
        }
        projected = {
            precision: _sensitivity_projection(by_precision[precision])
            for precision in PAIRED_REQUIRED_PRECISIONS
        }
        record = {
            **_paired_key_payload(key),
            "teacher_forced": True,
            "free_running_ground_truth": False,
            "alignment_hashes": {
                field: by_precision["BF16"][field]
                for field in PAIRED_ALIGNMENT_FIELDS
            },
            "reference_margin": float(
                by_precision["BF16"]["reference_top1_top2_margin"]
            ),
            "precision": projected,
            "layer_projection": {
                precision: _sensitivity_layer_values(by_precision[precision])
                for precision in PAIRED_REQUIRED_PRECISIONS
            },
            "deltas": {},
        }
        for name, high, low in SENSITIVITY_TRANSITIONS:
            record["deltas"][name] = _sensitivity_delta(low, high, projected)
        for name, high, low in SENSITIVITY_MIXED_TRANSITIONS:
            record["deltas"][name] = _sensitivity_delta(low, high, projected)
        records.append(record)

    integrity = {
        "candidate_row_count": len(normalized),
        "candidate_token_group_count": len(grouped),
        "paired_token_count": len(records),
        "invalid_token_group_count": len(invalid_groups),
        "grouping_error_count": len(grouping_errors),
        "required_precisions": list(PAIRED_REQUIRED_PRECISIONS),
        "required_fields": list(SENSITIVITY_REQUIRED_FIELDS),
        "alignment_fields": list(PAIRED_ALIGNMENT_FIELDS),
        "fail_closed": bool(grouping_errors or invalid_groups),
        "errors": grouping_errors[:20],
        "invalid_groups": invalid_groups[:20],
    }
    if grouping_errors or invalid_groups or not records:
        return {
            "records": [],
            "summary": {
                "analysis_version": SENSITIVITY_ANALYSIS_VERSION,
                "available": False,
                "conclusion": "INCONCLUSIVE",
                "definition": (
                    "Sensitivity analysis requires one teacher-forced, aligned "
                    "offline-oracle record for every precision on every token."
                ),
                "integrity": integrity,
                "criteria": {
                    "sensitivity_oracle_available": False,
                    "stable_sensitivity_improvement": False,
                },
                "limitations": [
                    "No partial sensitivity groups are analyzed when alignment "
                    "or oracle fields fail closed.",
                    "Gradient/JVP values are offline oracles and are not runtime "
                    "QDM telemetry.",
                ],
            },
        }

    margin_definition = _assign_paired_margin_buckets(records)
    paired_transitions = {
        "overall": _sensitivity_transition_summary(records, SENSITIVITY_TRANSITIONS),
        "by_prompt": _sensitivity_grouped_summary(
            records, "sample", SENSITIVITY_TRANSITIONS
        ),
        "by_context": _sensitivity_grouped_summary(
            records, "context_length", SENSITIVITY_TRANSITIONS
        ),
        "by_margin_bucket": _sensitivity_grouped_summary(
            records, "margin_bucket", SENSITIVITY_TRANSITIONS
        ),
        "margin_bucket_definition": margin_definition,
    }
    overall_comparison = {
        transition: _sensitivity_feature_beats(
            result.get("metrics", {})
        )
        for transition, result in paired_transitions["overall"][
            "transitions"
        ].items()
    }
    comparison_by_prompt = {
        key: {
            transition: _sensitivity_feature_beats(
                result.get("metrics", {})
            )
            for transition, result in value.get("transitions", {}).items()
        }
        for key, value in paired_transitions["by_prompt"].items()
    }
    comparison_by_context = {
        key: {
            transition: _sensitivity_feature_beats(
                result.get("metrics", {})
            )
            for transition, result in value.get("transitions", {}).items()
        }
        for key, value in paired_transitions["by_context"].items()
    }
    comparison_by_margin = {
        key: {
            transition: _sensitivity_feature_beats(
                result.get("metrics", {})
            )
            for transition, result in value.get("transitions", {}).items()
        }
        for key, value in paired_transitions["by_margin_bucket"].items()
    }
    layer_conditioned = _sensitivity_layer_conditioned_summary(
        records, SENSITIVITY_TRANSITIONS
    )
    comparison_by_layer = {
        layer: {
            transition: _sensitivity_feature_beats(
                result.get("metrics", {})
            )
            for transition, result in value.get("transitions", {}).items()
        }
        for layer, value in layer_conditioned.items()
    }

    quantized_rows = [
        row
        for row in normalized
        if row.get("precision_composition") != "BF16"
        and _sensitivity_projection(row) is not None
    ]
    conditioned = _sensitivity_conditioned_stats(normalized)
    cross_layer = {
        "overall": _sensitivity_cross_layer_summary(quantized_rows),
        "by_precision": {
            precision: _sensitivity_cross_layer_summary(
                [
                    row
                    for row in quantized_rows
                    if row.get("precision_composition") == precision
                ]
            )
            for precision in PAIRED_REQUIRED_PRECISIONS
        },
        "by_context": {
            context: _sensitivity_cross_layer_summary(
                [
                    row
                    for row in quantized_rows
                    if str(row.get("context_length")) == context
                ]
            )
            for context in sorted(
                {str(row.get("context_length")) for row in quantized_rows}
            )
        },
    }

    support_by_prompt = _sensitivity_group_support(
        paired_transitions["by_prompt"]
    )
    support_by_context = _sensitivity_group_support(
        paired_transitions["by_context"]
    )
    support_by_margin = _sensitivity_group_support(
        paired_transitions["by_margin_bucket"]
    )

    def overall_improved() -> bool:
        for transition, _, _ in SENSITIVITY_TRANSITIONS[1:]:
            comparison = overall_comparison[transition]
            if not all(
                comparison.get(outcome, {}).get("sensitivity_beats_physical", False)
                for outcome in ("vs_delta_KL", "vs_delta_JS")
            ):
                return False
        return True

    def overall_physical_positive() -> bool:
        for transition, _, _ in SENSITIVITY_TRANSITIONS[1:]:
            metrics = paired_transitions["overall"]["transitions"][transition][
                "metrics"
            ]
            for outcome in ("vs_delta_KL", "vs_delta_JS"):
                if not _sensitivity_relation_is_positive(
                    metrics["physical_norm_only"][outcome]
                ):
                    return False
        return True

    def overall_sensitivity_positive() -> bool:
        for transition, _, _ in SENSITIVITY_TRANSITIONS[1:]:
            metrics = paired_transitions["overall"]["transitions"][transition][
                "metrics"
            ]
            for outcome in ("vs_delta_KL", "vs_delta_JS"):
                if not _sensitivity_relation_is_positive(
                    metrics["sensitivity_weighted_error"][outcome]
                ):
                    return False
        return True

    stable_improvement = bool(
        overall_improved()
        and support_by_prompt["improved_groups"] >= PAIRED_MIN_PROMPT_SUPPORT
        and support_by_prompt["supported_groups"] >= PAIRED_MIN_PROMPT_SUPPORT
        and support_by_context["supported_groups"] >= 2
        and support_by_context["improved_groups"]
        == support_by_context["supported_groups"]
        and support_by_margin["improved_groups"] >= 3
        and support_by_margin["supported_groups"] >= 3
    )
    sensitivity_positive = overall_sensitivity_positive()
    local_sensitivity_improvement = bool(
        support_by_prompt["improved_groups"]
        or support_by_context["improved_groups"]
        or support_by_margin["improved_groups"]
    )
    conclusion = (
        "SENSITIVITY_MISSING_FACTOR"
        if stable_improvement
        else "INCONCLUSIVE"
        if local_sensitivity_improvement
        else "SENSITIVITY_REJECTED"
    )
    summary = {
        "analysis_version": SENSITIVITY_ANALYSIS_VERSION,
        "available": True,
        "conclusion": conclusion,
        "definition": (
            "BF16 top1-vs-top2 margin gradients are projected onto the exact "
            "same-token attention-output delta. This is an offline oracle only."
        ),
        "integrity": integrity,
        "paired_token_count": len(records),
        "paired_analysis": paired_transitions,
        "sensitivity_vs_physical": {
            "overall": overall_comparison,
            "by_prompt": comparison_by_prompt,
            "by_context": comparison_by_context,
            "by_margin_bucket": comparison_by_margin,
            "by_layer": comparison_by_layer,
        },
        "layer_conditioned": layer_conditioned,
        "precision_conditioned": conditioned,
        "cross_layer": cross_layer,
        "criteria": {
            "sensitivity_oracle_available": True,
            "stable_sensitivity_improvement": stable_improvement,
            "overall_sensitivity_beats_physical": overall_improved(),
            "overall_sensitivity_relation_positive": sensitivity_positive,
            "local_sensitivity_improvement": local_sensitivity_improvement,
            "physical_error_stable": overall_physical_positive(),
            "prompt_support": support_by_prompt,
            "context_support": support_by_context,
            "margin_bucket_support": support_by_margin,
            "primary_physical_feature": "physical_norm_only",
            "primary_sensitivity_feature": "sensitivity_weighted_error",
        },
        "limitations": [
            "The gradient/JVP is an offline BF16 oracle; it is not a runtime QDM "
            "feature and its backward cost is excluded from detector overhead.",
            "Layer traces store scalar norms/dot products; full activation vectors "
            "are transient and are not persisted.",
            "Paired deltas establish controlled association, not a causal proof "
            "of downstream token error.",
        ],
    }
    return {"records": records, "summary": summary}


def _stratification_summary(
    groups: Mapping[str, Mapping[str, Any]], *, minimum_rows: int = 32
) -> dict[str, Any]:
    counts = [int(group.get("count", 0)) for group in groups.values()]
    supported = [
        group for group in groups.values() if int(group.get("count", 0)) >= minimum_rows
    ]
    positive_spearman = sum(
        1
        for group in supported
        if (
            group.get("metric_comparison", {})
            .get("max_attention_error", {})
            .get("spearman_vs_kl")
            is not None
            and group["metric_comparison"]["max_attention_error"]["spearman_vs_kl"]
            > 0.0
        )
    )
    return {
        "observed_strata": len(groups),
        "minimum_rows_per_stratum": minimum_rows,
        "supported_strata": len(supported),
        "positive_qdm_vs_kl_supported_strata": positive_spearman,
        "count_min": min(counts) if counts else 0,
        "count_median": _percentile(counts, 0.5) if counts else None,
        "count_max": max(counts) if counts else 0,
        "attribution_note": (
            "worst_layer/worst_kv_head is the max aggregate QDM attribution; "
            "it is not a causal per-head intervention estimate"
        ),
    }


def summarize_qdm_validation_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize overall and precision-bucket Phase 1.5 evidence."""
    normalized = [dict(row) for row in rows]
    if any(not row.get("teacher_forced", False) for row in normalized):
        raise ValueError("QDM validation summary accepts teacher-forced rows only")
    if any(
        not row.get(field, False)
        for row in normalized
        for field in ("prefix_aligned", "suffix_aligned", "target_aligned")
    ):
        raise ValueError("QDM validation rows must pass strict token alignment")
    groups: dict[str, list[dict[str, Any]]] = {
        precision: [] for precision in PRECISION_COMPOSITIONS
    }
    for row in normalized:
        precision = str(row.get("precision_composition", ""))
        if precision not in groups:
            raise ValueError(f"unknown precision composition: {precision}")
        groups[precision].append(row)
    overall = _diagnostic_group_stats(normalized)
    quantized_rows = [
        row for row in normalized if row["precision_composition"] != "BF16"
    ]
    quantized_overall = _diagnostic_group_stats(quantized_rows)
    buckets = {
        precision: _diagnostic_group_stats(group) for precision, group in groups.items()
    }
    context_values = sorted(
        {
            int(row["context_length"])
            for row in normalized
            if row.get("context_length") is not None
        }
    )
    context_buckets = {
        str(context): _diagnostic_group_stats(
            [row for row in normalized if int(row["context_length"]) == context]
        )
        for context in context_values
    }
    quantized_context_buckets = {
        str(context): _diagnostic_group_stats(
            [
                row
                for row in normalized
                if int(row["context_length"]) == context
                and row["precision_composition"] != "BF16"
            ]
        )
        for context in context_values
    }
    quantized_layer_buckets = {
        key: _diagnostic_group_stats(group)
        for key, group in _group_rows_by_key(
            quantized_rows,
            lambda row: row.get("worst_layer"),
        ).items()
    }
    quantized_kv_head_buckets = {
        key: _diagnostic_group_stats(group)
        for key, group in _group_rows_by_key(
            quantized_rows,
            lambda row: row.get("worst_kv_head"),
        ).items()
    }
    quantized_layer_head_buckets = {
        key: _diagnostic_group_stats(group)
        for key, group in _group_rows_by_key(
            quantized_rows,
            lambda row: (
                f"layer={int(row['worst_layer'])},kv_head={int(row['worst_kv_head'])}"
                if row.get("worst_layer") is not None
                and row.get("worst_kv_head") is not None
                else None
            ),
        ).items()
    }
    bf16_witness = buckets["BF16"]["witness"]
    bf16_witness_max = max(
        bf16_witness.get("max_k_error", 0.0),
        bf16_witness.get("max_v_error", 0.0),
    )
    bf16_rows = [row for row in normalized if row["precision_composition"] == "BF16"]
    bf16_physical_keys = (
        "max_tv_bound",
        "p95_tv_bound",
        "max_v_error",
        "max_attention_error",
    )
    bf16_physical_max = max(
        (
            max((float(row[key]) for row in bf16_rows), default=0.0)
            for key in bf16_physical_keys
        ),
        default=0.0,
    )
    formula_max = max(
        (
            float(row.get("attention_error_formula_max_abs_error", 0.0))
            for row in normalized
        ),
        default=0.0,
    )
    precision_order = ("K2V2", "K4V2", "K8V4", "BF16")
    precision_conditioning = {
        precision: {
            "count": buckets[precision]["count"],
            "mean_max_attention_error": buckets[precision]["means"].get(
                "max_attention_error"
            ),
            "mean_kl": buckets[precision]["means"].get("kl_bf16_quantized"),
            "mean_js": buckets[precision]["means"].get("js_divergence"),
            "top1_flip_rate": buckets[precision]["means"].get("top1_flip_rate"),
        }
        for precision in precision_order
    }

    def nonincreasing(values: Sequence[float | None]) -> bool:
        finite = [float(value) for value in values if value is not None]
        return len(finite) == len(values) and all(
            left >= right for left, right in zip(finite, finite[1:], strict=False)
        )

    saturation_report = build_saturation_report(normalized)
    incremental_value = build_incremental_value_report(normalized)
    bound_tightness = build_bound_tightness_report(normalized)
    exact_drift_analysis = build_exact_drift_report(normalized)
    paired_analysis = build_paired_precision_analysis(normalized)
    paired_counterfactual = paired_analysis["summary"]
    sensitivity_analysis = build_downstream_sensitivity_analysis(normalized)
    downstream_sensitivity = sensitivity_analysis["summary"]
    exact_oracle_available = bool(exact_drift_analysis.get("available", False))
    exact_probability_error = max(
        (
            float(row.get("reference_block_probability_max_abs_error", 0.0))
            for row in normalized
            if row.get("reference_block_probability_max_abs_error") is not None
        ),
        default=0.0,
    )
    bf16_exact_rows = [
        row
        for row in bf16_rows
        if all(metric in row for metric in EXACT_DRIFT_METRIC_NAMES)
    ]
    bf16_exact_max = max(
        (
            float(row[metric])
            for row in bf16_exact_rows
            for metric in EXACT_DRIFT_METRIC_NAMES
        ),
        default=0.0,
    )
    exact_oracle_integrity = {
        "available": exact_oracle_available,
        "row_count": len(
            [
                row
                for row in normalized
                if all(metric in row for metric in EXACT_DRIFT_METRIC_NAMES)
            ]
        ),
        "max_reference_block_probability_abs_error": exact_probability_error,
        "reference_block_probability_aligned": bool(
            exact_oracle_available
            and exact_probability_error <= EXACT_ORACLE_PROBABILITY_TOLERANCE
        ),
        "bf16_rows_with_exact_metrics": len(bf16_exact_rows),
        "bf16_exact_max_error": bf16_exact_max,
        "bf16_exact_zero": bool(
            exact_oracle_available
            and len(bf16_exact_rows) == len(bf16_rows)
            and bf16_rows
            and bf16_exact_max <= EXACT_ORACLE_BF16_ZERO_TOLERANCE
        ),
    }
    exact_oracle_integrity["integrity_ok"] = bool(
        exact_oracle_integrity["available"]
        and exact_oracle_integrity["reference_block_probability_aligned"]
        and exact_oracle_integrity["bf16_exact_zero"]
    )

    return {
        "protocol": VALIDATION_PROTOCOL,
        "analysis_version": QDM_VALIDATION_ANALYSIS_VERSION,
        "precision_compositions": list(PRECISION_COMPOSITIONS),
        "teacher_forced_alignment": {
            "enabled": True,
            "prefix_equality_checked": True,
            "free_running_primary_ground_truth": False,
            "additional_monitoring_forward": False,
        },
        "witness_source": "production_quantizer_payload_and_reference_dequantization",
        "kv_compression_scope": KV_COMPRESSION_SCOPE,
        "attention_matrix_materialized": False,
        "risk_state": {
            "labels": [
                "SAFE",
                "MODEL_FRAGILE",
                "KV_DRIFT_ROBUST",
                "KV_TOKEN_RISK",
            ],
            "calibrated": False,
            "threshold_policy": (
                "per-precision empirical quantiles for diagnostics only"
            ),
        },
        "overall": overall,
        "quantized_overall": quantized_overall,
        "precision_buckets": buckets,
        "context_buckets": context_buckets,
        "quantized_context_buckets": quantized_context_buckets,
        "quantized_layer_buckets": quantized_layer_buckets,
        "quantized_kv_head_buckets": quantized_kv_head_buckets,
        "quantized_layer_head_buckets": quantized_layer_head_buckets,
        "layer_head_stratification": {
            "layer": _stratification_summary(quantized_layer_buckets),
            "kv_head": _stratification_summary(quantized_kv_head_buckets),
            "layer_kv_head": _stratification_summary(quantized_layer_head_buckets),
        },
        "precision_conditioning": {
            "order": list(precision_order),
            "by_precision": precision_conditioning,
            "mean_qdm_nonincreasing_with_compression": nonincreasing(
                [
                    precision_conditioning[precision]["mean_max_attention_error"]
                    for precision in precision_order
                ]
            ),
            "mean_kl_nonincreasing_with_compression": nonincreasing(
                [
                    precision_conditioning[precision]["mean_kl"]
                    for precision in precision_order
                ]
            ),
            "mean_js_nonincreasing_with_compression": nonincreasing(
                [
                    precision_conditioning[precision]["mean_js"]
                    for precision in precision_order
                ]
            ),
            "diagnostic_only": True,
        },
        "saturation_report": saturation_report,
        "incremental_value": incremental_value,
        "bound_tightness": bound_tightness,
        "exact_drift_analysis": exact_drift_analysis,
        "paired_counterfactual": paired_counterfactual,
        "downstream_sensitivity": downstream_sensitivity,
        "integrity": {
            "row_count": len(normalized),
            "sample_count": len({str(row.get("sample")) for row in normalized}),
            "all_teacher_forced": all(
                bool(row.get("teacher_forced", False)) for row in normalized
            ),
            "all_prefix_aligned": all(
                bool(row.get("prefix_aligned", False)) for row in normalized
            ),
            "all_suffix_aligned": all(
                bool(row.get("suffix_aligned", False)) for row in normalized
            ),
            "all_target_aligned": all(
                bool(row.get("target_aligned", False)) for row in normalized
            ),
            "max_attention_error_formula_abs_error": formula_max,
            "exact_oracle": exact_oracle_integrity,
        },
        "bf16_witness": {
            "observed": buckets["BF16"]["count"] > 0,
            "max_k_error": bf16_witness.get("max_k_error", 0.0),
            "max_v_error": bf16_witness.get("max_v_error", 0.0),
            "max_v_norm": bf16_witness.get("max_v_norm", 0.0),
            "witness_error_max": bf16_witness_max,
            "all_error_witness_zero": (
                buckets["BF16"]["count"] > 0 and bf16_witness_max == 0.0
            ),
            "physical_drift_max": bf16_physical_max,
            "all_physical_drift_zero": (
                buckets["BF16"]["count"] > 0 and bf16_physical_max == 0.0
            ),
            "exact_oracle_max_score_error": max(
                (
                    float(row["exact_max_score_error"])
                    for row in bf16_exact_rows
                ),
                default=0.0,
            ),
            "exact_oracle_attention_tv": max(
                (
                    float(row["exact_attention_TV"])
                    for row in bf16_exact_rows
                ),
                default=0.0,
            ),
            "exact_oracle_output_error": max(
                (
                    float(row["exact_attention_output_error"])
                    for row in bf16_exact_rows
                ),
                default=0.0,
            ),
            "exact_oracle_all_physical_drift_zero": exact_oracle_integrity[
                "bf16_exact_zero"
            ],
        },
    }


def assess_qdm_validation(
    summary: Mapping[str, Any],
    *,
    required_context_lengths: Sequence[int] = (),
    run: Mapping[str, Any] | None = None,
    min_rows_per_precision: int = 32,
) -> dict[str, Any]:
    """Classify evidence without fitting thresholds or a risk predictor."""
    run_info = dict(run or summary.get("run", {}))
    buckets = summary.get("precision_buckets", {})
    missing_precisions = [
        precision
        for precision in PRECISION_COMPOSITIONS
        if int(buckets.get(precision, {}).get("count", 0)) == 0
    ]
    context_buckets = summary.get("context_buckets", {})
    missing_contexts = [
        int(context)
        for context in required_context_lengths
        if str(int(context)) not in context_buckets
    ]
    bf16 = summary.get("bf16_witness", {})
    structural_failures: list[str] = []
    if bf16.get("observed") and not bf16.get("all_error_witness_zero", False):
        structural_failures.append("BF16 witness residual is non-zero")
    if bf16.get("observed") and not bf16.get("all_physical_drift_zero", False):
        structural_failures.append("BF16 physical QDM drift is non-zero")
    integrity = summary.get("integrity", {})
    if integrity.get("row_count", 0) and not all(
        integrity.get(field, False)
        for field in (
            "all_teacher_forced",
            "all_prefix_aligned",
            "all_suffix_aligned",
            "all_target_aligned",
        )
    ):
        structural_failures.append("teacher-forced token alignment invariant failed")
    for failure in run_info.get("failures", []):
        if failure.get("failure_kind") == "alignment":
            structural_failures.append(
                f"alignment failure for sample {failure.get('sample')}"
            )
        elif failure.get("failure_kind") == "invariant":
            structural_failures.append(
                f"QDM invariant failure for sample {failure.get('sample')}"
            )

    overall = summary.get("quantized_overall", summary.get("overall", {}))
    overall_metrics = overall.get("metric_comparison", {})
    qdm_metric = overall_metrics.get("max_attention_error", {})
    margin_metric = overall_metrics.get("top1_margin", {})
    qdm_plus_margin = overall.get("qdm_plus_margin", {})
    enrichment = overall.get("diagnostic_enrichment", {})
    incremental = summary.get("incremental_value", {})
    incremental_overall = incremental.get("overall", {})
    incremental_max_attention = incremental_overall.get("by_qdm_metric", {}).get(
        "max_attention_error", {}
    )
    saturation = summary.get("saturation_report", {})
    saturation_coverage = saturation.get("quantized_overall", {}).get(
        "diagnostic_coverage"
    )
    saturation_diagnostic_complete = saturation_coverage == 1.0
    qdm_high_small_rate = enrichment.get("qdm_high_small_margin_flip_rate")
    qdm_high_rate = enrichment.get("qdm_high_flip_rate")
    small_margin_rate = enrichment.get("small_margin_flip_rate")
    all_flip_rate = enrichment.get("all_flip_rate")
    successful_prompt_count = int(
        run_info.get(
            "sample_count_success",
            summary.get("integrity", {}).get("sample_count", 0),
        )
        or 0
    )
    minimum_prompt_support = (
        successful_prompt_count >= QDM_MINIMUM_VALIDATION_PROMPTS
    )
    no_run_failures = int(run_info.get("failure_count", 0) or 0) == 0
    minimum_rows_per_precision = not missing_precisions and all(
        int(buckets.get(precision, {}).get("count", 0)) >= min_rows_per_precision
        for precision in PRECISION_COMPOSITIONS
    )
    positive_qdm_drift = (
        qdm_metric.get("spearman_vs_kl") is not None
        and qdm_metric["spearman_vs_kl"] > 0.0
        and qdm_metric.get("spearman_vs_js") is not None
        and qdm_metric["spearman_vs_js"] > 0.0
    )
    qdm_plus_margin_better = (
        qdm_plus_margin.get("auroc_top1_flip") is not None
        and qdm_metric.get("auroc_top1_flip") is not None
        and margin_metric.get("auroc_top1_flip") is not None
        and qdm_plus_margin["auroc_top1_flip"]
        > max(qdm_metric["auroc_top1_flip"], margin_metric["auroc_top1_flip"])
        and qdm_high_small_rate is not None
        and qdm_high_rate is not None
        and small_margin_rate is not None
        and qdm_high_small_rate > max(qdm_high_rate, small_margin_rate)
    )

    def _strictly_better(
        combined: Any, qdm_only: Any, margin_only: Any
    ) -> bool:
        if not all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in (combined, qdm_only, margin_only)
        ):
            return False
        return float(combined) > max(float(qdm_only), float(margin_only))

    def _margin_controlled_incremental_evidence(group: Mapping[str, Any]) -> bool:
        qdm_only = group.get("qdm_only", {})
        margin_only = group.get("margin_only", {})
        combined = group.get("margin_plus_qdm", {})
        enrichment_values = group.get("top1_flip_enrichment", {})
        combined_enrichment = enrichment_values.get("margin_plus_qdm", {})
        qdm_enrichment = enrichment_values.get("qdm_only", {})
        margin_enrichment = enrichment_values.get("margin_only", {})
        combined_enrichment_value = (
            combined_enrichment.get("enrichment")
            if isinstance(combined_enrichment, Mapping)
            else None
        )
        qdm_enrichment_value = (
            qdm_enrichment.get("enrichment")
            if isinstance(qdm_enrichment, Mapping)
            else None
        )
        margin_enrichment_value = (
            margin_enrichment.get("enrichment")
            if isinstance(margin_enrichment, Mapping)
            else None
        )
        conditioned = group.get("margin_conditioned", {})
        return (
            _strictly_better(
                combined.get("auroc_top1_flip"),
                qdm_only.get("auroc_top1_flip"),
                margin_only.get("auroc_top1_flip"),
            )
            and _strictly_better(
                combined.get("pr_auc_top1_flip"),
                qdm_only.get("pr_auc_top1_flip"),
                margin_only.get("pr_auc_top1_flip"),
            )
            and _strictly_better(
                combined_enrichment_value,
                qdm_enrichment_value,
                margin_enrichment_value,
            )
            and int(conditioned.get("supported_bin_count", 0)) >= 2
            and int(conditioned.get("positive_flip_support_count", 0)) >= 1
            and int(conditioned.get("positive_kl_support_count", 0)) >= 1
            and int(conditioned.get("positive_js_support_count", 0)) >= 1
        )

    margin_controlled_incremental_value = _margin_controlled_incremental_evidence(
        incremental_max_attention
    )
    positive_precision_support = 0
    for precision in ("K2V2", "K4V2", "K8V4", "MIXED"):
        metric = (
            buckets.get(precision, {})
            .get("metric_comparison", {})
            .get("max_attention_error", {})
        )
        if (
            metric.get("spearman_vs_kl") is not None
            and metric.get("spearman_vs_js") is not None
            and metric["spearman_vs_kl"] > 0.0
            and metric["spearman_vs_js"] > 0.0
        ):
            positive_precision_support += 1
    positive_context_support = 0
    for context in required_context_lengths:
        metric = (
            summary.get("quantized_context_buckets", {})
            .get(str(int(context)), {})
            .get("metric_comparison", {})
            .get("max_attention_error", {})
        )
        if (
            metric.get("spearman_vs_kl") is not None
            and metric.get("spearman_vs_js") is not None
            and metric["spearman_vs_kl"] > 0.0
            and metric["spearman_vs_js"] > 0.0
        ):
            positive_context_support += 1
    positive_incremental_precision_support = 0
    for precision in ("K2V2", "K4V2", "K8V4", "MIXED"):
        group = (
            incremental.get("by_precision", {})
            .get(precision, {})
            .get("by_qdm_metric", {})
            .get("max_attention_error", {})
        )
        positive_incremental_precision_support += int(
            _margin_controlled_incremental_evidence(group)
        )
    positive_incremental_context_support = 0
    for context in required_context_lengths:
        group = (
            incremental.get("by_context", {})
            .get(str(int(context)), {})
            .get("by_qdm_metric", {})
            .get("max_attention_error", {})
        )
        positive_incremental_context_support += int(
            _margin_controlled_incremental_evidence(group)
        )
    required_context_support = (
        max(1, min(2, len(required_context_lengths))) if required_context_lengths else 0
    )
    criteria = {
        "required_compositions_present": not missing_precisions,
        "required_context_lengths_present": not missing_contexts,
        "minimum_rows_per_precision": minimum_rows_per_precision,
        "minimum_prompt_support": minimum_prompt_support,
        "no_run_failures": no_run_failures,
        "saturation_diagnostic_complete": saturation_diagnostic_complete,
        "qdm_vs_kl_js_positive_spearman": positive_qdm_drift,
        "qdm_plus_margin_beats_single_predictors": qdm_plus_margin_better,
        "margin_controlled_incremental_value": margin_controlled_incremental_value,
        "positive_qdm_supporting_precision_buckets": positive_precision_support >= 2,
        "positive_qdm_supporting_context_buckets": (
            positive_context_support >= required_context_support
        ),
        "positive_incremental_precision_buckets": (
            positive_incremental_precision_support >= 2
        ),
        "positive_incremental_context_buckets": (
            positive_incremental_context_support >= required_context_support
        ),
        "nonzero_flip_observed": bool(all_flip_rate and all_flip_rate > 0.0),
    }
    limitations = [
        "QDM attention error is a bound/diagnostic and is not token error.",
        "Empirical quantiles are analysis-only and are not production thresholds.",
        "No classifier, fitted parameter, calibration set, or shadow decode is used.",
    ]
    if run_info.get("environment_status") == "gpu_unavailable":
        limitations.append(
            "A CUDA device was unavailable; no real Qwen3-8B run was executed."
        )
    if missing_precisions:
        limitations.append(
            "Missing precision buckets: " + ", ".join(missing_precisions)
        )
    if missing_contexts:
        limitations.append(
            "Missing requested context lengths: "
            + ", ".join(str(value) for value in missing_contexts)
        )
    if not minimum_prompt_support:
        limitations.append(
            "At least "
            f"{QDM_MINIMUM_VALIDATION_PROMPTS} distinct successful prompts are "
            "required for a real-model VALIDATED conclusion."
        )
    if not saturation_diagnostic_complete:
        limitations.append(
            "TV saturation diagnostics are incomplete; missing raw_A/raw_tv_bound/"
            "log_A/saturated fields are not treated as zero evidence."
        )
    if not no_run_failures:
        limitations.append(
            "One or more requested prompt/context runs failed; the result is "
            "not eligible for VALIDATED."
        )
    if not margin_controlled_incremental_value:
        limitations.append(
            "Overall max_attention_error has not demonstrated stable margin-"
            "controlled incremental value across AUROC, PR-AUC, enrichment, "
            "and margin-conditioned KL/JS/flip checks."
        )
    evidence_complete = (
        not missing_precisions
        and not missing_contexts
        and minimum_rows_per_precision
        and minimum_prompt_support
        and saturation_diagnostic_complete
        and no_run_failures
        and bool(all_flip_rate and all_flip_rate > 0.0)
    )
    no_physical_signal = (
        not positive_qdm_drift
        and positive_precision_support == 0
        and positive_context_support == 0
    )
    if structural_failures:
        status = "FAILED"
    elif evidence_complete and no_physical_signal:
        status = "FAILED"
    elif all(criteria.values()):
        status = "VALIDATED"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "criteria": criteria,
        "structural_failures": structural_failures,
        "missing_precision_compositions": missing_precisions,
        "missing_context_lengths": missing_contexts,
        "minimum_rows_per_precision": min_rows_per_precision,
        "supporting_evidence": {
            "overall_max_attention_error_metric": qdm_metric,
            "overall_qdm_plus_margin": qdm_plus_margin,
            "overall_enrichment": enrichment,
            "overall_incremental_max_attention_error": incremental_max_attention,
            "saturation_diagnostic_coverage": saturation_coverage,
            "positive_precision_bucket_count": positive_precision_support,
            "positive_context_bucket_count": positive_context_support,
            "positive_incremental_precision_bucket_count": (
                positive_incremental_precision_support
            ),
            "positive_incremental_context_bucket_count": (
                positive_incremental_context_support
            ),
            "required_context_bucket_support": required_context_support,
            "successful_prompt_count": successful_prompt_count,
            "minimum_prompt_count": QDM_MINIMUM_VALIDATION_PROMPTS,
        },
        "limitations": limitations,
    }


def _strictly_better_values(
    combined: Any, first: Any, second: Any
) -> bool:
    if not all(
        isinstance(item, (int, float)) and math.isfinite(float(item))
        for item in (combined, first, second)
    ):
        return False
    return float(combined) > max(float(first), float(second))


def _exact_incremental_evidence(group: Mapping[str, Any]) -> bool:
    if not group.get("available", False):
        return False
    metrics = group.get("metrics", {})
    for metric in EXACT_DRIFT_METRIC_NAMES:
        result = metrics.get(metric, {})
        qdm_only = result.get("metric_comparison", {})
        margin_only = result.get("margin_only", {})
        combined = result.get("margin_plus_exact", {})
        enrichment = result.get("top1_flip_enrichment", {})
        exact_enrichment = enrichment.get("exact_only", {})
        margin_enrichment = enrichment.get("margin_only", {})
        combined_enrichment = enrichment.get("margin_plus_exact", {})
        conditioned = result.get("margin_conditioned", {})
        if (
            _strictly_better_values(
                combined.get("auroc_top1_flip"),
                qdm_only.get("auroc_top1_flip"),
                margin_only.get("auroc_top1_flip"),
            )
            and _strictly_better_values(
                combined.get("pr_auc_top1_flip"),
                qdm_only.get("pr_auc_top1_flip"),
                margin_only.get("pr_auc_top1_flip"),
            )
            and _strictly_better_values(
                combined_enrichment.get("enrichment"),
                exact_enrichment.get("enrichment"),
                margin_enrichment.get("enrichment"),
            )
            and int(conditioned.get("supported_bin_count", 0)) >= 2
            and int(conditioned.get("positive_flip_support_count", 0)) >= 1
            and int(conditioned.get("positive_kl_support_count", 0)) >= 1
            and int(conditioned.get("positive_js_support_count", 0)) >= 1
        ):
            return True
    return False


def _strictly_positive_exact_correlation(group: Mapping[str, Any]) -> bool:
    if not group.get("available", False):
        return False
    return any(
        group.get("metrics", {}).get(metric, {}).get("spearman_vs_kl") is not None
        and group["metrics"][metric].get("spearman_vs_js") is not None
        and group["metrics"][metric]["spearman_vs_kl"] > 0.0
        and group["metrics"][metric]["spearman_vs_js"] > 0.0
        for metric in EXACT_DRIFT_METRIC_NAMES
    )


def assess_exact_drift(
    summary: Mapping[str, Any],
    *,
    run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify whether looseness or lack of signal explains QDM results."""
    del run
    exact = summary.get("exact_drift_analysis", {})
    tightness = summary.get("bound_tightness", {})
    if not exact.get("available", False) or not tightness.get("available", False):
        return {
            "conclusion": "INCONCLUSIVE",
            "criteria": {
                "exact_oracle_available": False,
                "exact_oracle_integrity_ok": False,
                "stable_exact_incremental_value": False,
                "bound_looseness_evidence": False,
            },
            "supporting_evidence": {},
            "limitations": [
                "Exact oracle and bound tightness fields are required before "
                "classifying witness looseness versus signal usefulness."
            ],
        }

    exact_oracle_integrity = summary.get("integrity", {}).get(
        "exact_oracle", {}
    )
    if not exact_oracle_integrity.get("integrity_ok", False):
        return {
            "conclusion": "INCONCLUSIVE",
            "criteria": {
                "exact_oracle_available": True,
                "exact_oracle_integrity_ok": False,
                "stable_exact_incremental_value": False,
                "bound_looseness_evidence": False,
            },
            "supporting_evidence": {
                "exact_oracle_integrity": exact_oracle_integrity,
            },
            "limitations": [
                "Exact oracle integrity failed closed: reference block "
                "probabilities must align and BF16 exact drift must be zero.",
            ],
        }

    quantized = exact.get("quantized_overall", {})
    overall_metrics = quantized.get("metrics", {})
    by_precision = exact.get("by_precision", {})
    by_context = exact.get("by_context", {})
    precision_support = {
        metric: sum(
            _exact_incremental_evidence(
                {
                    "available": bucket.get("available", False),
                    "metrics": {metric: bucket.get("metrics", {}).get(metric, {})},
                }
            )
            for bucket in by_precision.values()
            if bucket.get("available", False)
        )
        for metric in EXACT_DRIFT_METRIC_NAMES
    }
    context_support = {
        metric: sum(
            _exact_incremental_evidence(
                {
                    "available": bucket.get("available", False),
                    "metrics": {metric: bucket.get("metrics", {}).get(metric, {})},
                }
            )
            for bucket in by_context.values()
            if bucket.get("available", False)
        )
        for metric in EXACT_DRIFT_METRIC_NAMES
    }
    positive_precision = sum(
        _strictly_positive_exact_correlation(bucket)
        for bucket in by_precision.values()
        if bucket.get("available", False)
    )
    positive_context = sum(
        _strictly_positive_exact_correlation(bucket)
        for bucket in by_context.values()
        if bucket.get("available", False)
    )
    overall_incremental = {
        metric: _exact_incremental_evidence(
            {
                "available": quantized.get("available", False),
                "metrics": {metric: overall_metrics.get(metric, {})},
            }
        )
        for metric in EXACT_DRIFT_METRIC_NAMES
    }
    tightness_overall = tightness.get("quantized_overall", {})
    tv_ratio = tightness_overall.get("metrics", {}).get(
        "tv_bound_exact_ratio", {}
    ).get("distribution", {})
    output_ratio = tightness_overall.get("metrics", {}).get(
        "output_bound_exact_ratio", {}
    ).get("distribution", {})
    saturation_rate = tightness_overall.get("saturation_rate")
    bound_looseness_evidence = bool(
        isinstance(saturation_rate, (int, float))
        and saturation_rate > 0.5
        and (
            float(tv_ratio.get("p95") or 0.0) > 1.0
            or float(output_ratio.get("p95") or 0.0) > 1.0
        )
    )
    stable_exact_incremental = bool(
        any(overall_incremental.values())
        and max(precision_support.values(), default=0) >= 2
        and max(context_support.values(), default=0) >= 1
    )
    positive_exact_signal = bool(
        any(
            result.get("spearman_vs_kl") is not None
            and result.get("spearman_vs_js") is not None
            and result["spearman_vs_kl"] > 0.0
            and result["spearman_vs_js"] > 0.0
            for result in overall_metrics.values()
        )
    )
    if stable_exact_incremental and bound_looseness_evidence:
        conclusion = "WITNESS_TOO_LOOSE"
    elif not positive_exact_signal and not stable_exact_incremental:
        conclusion = "SIGNAL_NOT_USEFUL"
    else:
        conclusion = "INCONCLUSIVE"
    return {
        "conclusion": conclusion,
        "criteria": {
            "exact_oracle_available": True,
            "exact_oracle_integrity_ok": True,
            "stable_exact_incremental_value": stable_exact_incremental,
            "bound_looseness_evidence": bound_looseness_evidence,
            "positive_exact_signal": positive_exact_signal,
        },
        "supporting_evidence": {
            "overall_incremental_by_metric": overall_incremental,
            "precision_incremental_support_by_metric": precision_support,
            "context_incremental_support_by_metric": context_support,
            "positive_precision_bucket_count": positive_precision,
            "positive_context_bucket_count": positive_context,
            "quantized_saturation_rate": saturation_rate,
            "tv_bound_exact_ratio_distribution": tv_ratio,
            "output_bound_exact_ratio_distribution": output_ratio,
        },
        "limitations": [
            "Exact metrics are offline BF16-query oracles and are not production "
            "runtime metrics.",
            "The conclusion does not alter the clamped QDM TV certificate or any "
            "production threshold.",
        ],
    }


def write_paired_counterfactual_report(
    summary: Mapping[str, Any], output_dir: str | Path
) -> Path:
    """Write the paired precision counterfactual validation report."""
    paired = summary.get("paired_counterfactual", summary)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paired Precision Counterfactual Report",
        "",
        f"**Conclusion: {paired.get('conclusion', 'INCONCLUSIVE')}**",
        "",
        f"Analysis version: `{paired.get('analysis_version')}`",
        "Protocol: same sample/context/step/target token across all precision "
        "rows; teacher-forced only.",
        "",
        "## Integrity",
        "",
        f"- `{json.dumps(paired.get('integrity', {}), sort_keys=True)}`",
    ]
    if not paired.get("available", False):
        lines.extend(
            [
                "",
                "Pairing failed closed. No partial precision groups were analyzed.",
                "",
                "## Criteria",
                "",
                f"- `{json.dumps(paired.get('criteria', {}), sort_keys=True)}`",
            ]
        )
    else:
        monotonicity = paired.get("monotonicity", {})
        lines.extend(["", "## Monotonicity", ""])
        lines.append(
            "The homogeneous order is `BF16 <= K8V4 <= K4V2 <= K2V2`; this is "
            "measured, not imposed."
        )
        for metric, result in monotonicity.get("overall", {}).get(
            "metrics", {}
        ).items():
            within = result.get("within_token_spearman", {})
            lines.append(
                f"- `{metric}`: pairwise="
                f"{result.get('pairwise_concordance_rate')}, "
                f"monotonic_token_fraction="
                f"{result.get('monotonic_token_fraction')}, "
                f"within_token_spearman_mean={within.get('mean')}, "
                f"positive_fraction={within.get('positive_fraction')}"
            )

        def append_delta_section(
            title: str,
            analysis: Mapping[str, Any],
            metric: str = "exact_attention_output_error",
        ) -> None:
            lines.extend(["", title, ""])
            for transition, result in analysis.get("transitions", {}).items():
                physical = result.get("metrics", {}).get(metric, {})
                kl = physical.get("vs_delta_KL", {})
                js = physical.get("vs_delta_JS", {})
                trimmed_kl = kl.get(
                    "trimmed_5_percent_by_physical_magnitude", {}
                ).get("spearman")
                lines.append(
                    f"- `{transition}` / `{metric}`: "
                    f"delta-KL Spearman/sign="
                    f"{kl.get('spearman')}/{kl.get('sign_concordance_rate')}, "
                    f"delta-JS Spearman/sign="
                    f"{js.get('spearman')}/{js.get('sign_concordance_rate')}, "
                    f"trimmed-KL Spearman="
                    f"{trimmed_kl}"
                )

        append_delta_section(
            "## Paired Delta Overall",
            paired.get("delta_analysis", {}).get("overall", {}),
        )
        lines.extend(["", "### By Context", ""])
        for context, analysis in paired.get("delta_analysis", {}).get(
            "by_context", {}
        ).items():
            result = analysis.get("transitions", {}).get(
                "K8V4_to_K4V2", {}
            )
            metric = result.get("metrics", {}).get(
                "exact_attention_output_error", {}
            )
            lines.append(
                f"- `context={context}`: count={analysis.get('count')}, "
                f"K8->K4 delta-KL/JS="
                f"{metric.get('vs_delta_KL', {}).get('spearman')}/"
                f"{metric.get('vs_delta_JS', {}).get('spearman')}"
            )
        lines.extend(["", "### By Prompt", ""])
        for prompt, analysis in paired.get("delta_analysis", {}).get(
            "by_prompt", {}
        ).items():
            result = analysis.get("transitions", {}).get(
                "K8V4_to_K4V2", {}
            )
            metric = result.get("metrics", {}).get(
                "exact_attention_output_error", {}
            )
            lines.append(
                f"- `{prompt}`: count={analysis.get('count')}, "
                f"K8->K4 delta-KL/JS="
                f"{metric.get('vs_delta_KL', {}).get('spearman')}/"
                f"{metric.get('vs_delta_JS', {}).get('spearman')}"
            )
        lines.extend(["", "### By Margin Bucket", ""])
        for bucket, analysis in paired.get("delta_analysis", {}).get(
            "by_margin_bucket", {}
        ).items():
            result = analysis.get("transitions", {}).get(
                "K8V4_to_K4V2", {}
            )
            metric = result.get("metrics", {}).get(
                "exact_attention_output_error", {}
            )
            lines.append(
                f"- `{bucket}`: count={analysis.get('count')}, "
                f"K8->K4 delta-KL/JS="
                f"{metric.get('vs_delta_KL', {}).get('spearman')}/"
                f"{metric.get('vs_delta_JS', {}).get('spearman')}"
            )

        mixed = paired.get("mixed_analysis", {})
        append_delta_section("## MIXED Counterfactual", mixed)
        composition_text = json.dumps(
            mixed.get("actual_block_precision_composition", {}),
            sort_keys=True,
        )
        lines.append(
            "- Actual composition: `"
            f"{composition_text}`"
        )
        lines.extend(
            [
                "",
                "## Criteria",
                "",
                f"- `{json.dumps(paired.get('criteria', {}), sort_keys=True)}`",
                "",
                "## Limitations",
                "",
            ]
        )
        for item in paired.get("limitations", []):
            lines.append(f"- {item}")
    report_path = output / "paired_counterfactual_report.md"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def write_downstream_sensitivity_report(
    summary: Mapping[str, Any], output_dir: str | Path
) -> Path:
    """Write the offline downstream-sensitivity oracle report."""
    sensitivity = summary.get("downstream_sensitivity", summary)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Downstream Sensitivity Report",
        "",
        f"**Conclusion: {sensitivity.get('conclusion', 'INCONCLUSIVE')}**",
        "",
        f"Analysis version: `{sensitivity.get('analysis_version')}`",
        "Protocol: same teacher-forced token across BF16/K8V4/K4V2/K2V2/MIXED.",
        "The margin gradient is an offline BF16 oracle and is not runtime telemetry.",
        "",
        "## Integrity",
        "",
        f"- `{json.dumps(sensitivity.get('integrity', {}), sort_keys=True)}`",
    ]
    if not sensitivity.get("available", False):
        lines.extend(
            [
                "",
                "Sensitivity oracle data is unavailable; no partial groups were "
                "analyzed.",
                "",
                "## Criteria",
                "",
                f"- `{json.dumps(sensitivity.get('criteria', {}), sort_keys=True)}`",
            ]
        )
    else:
        lines.extend(["", "## Physical Versus Sensitivity-Weighted", ""])
        overall = sensitivity.get("sensitivity_vs_physical", {}).get("overall", {})
        for transition, comparison in overall.items():
            kl = comparison.get("vs_delta_KL", {})
            js = comparison.get("vs_delta_JS", {})
            flip = comparison.get("vs_delta_top1_flip", {})
            logit = comparison.get("vs_delta_logit_delta_l2", {})
            margin = comparison.get("vs_delta_margin_abs", {})
            lines.append(
                f"- `{transition}`: KL gain={kl.get('spearman_gain')}, "
                f"JS gain={js.get('spearman_gain')}, "
                f"top1-flip gain={flip.get('spearman_gain')}, "
                f"logit-L2/margin-drift gain={logit.get('spearman_gain')}/"
                f"{margin.get('spearman_gain')}, "
                f"beats(KL/JS)={kl.get('sensitivity_beats_physical')}/"
                f"{js.get('sensitivity_beats_physical')}"
            )
        lines.extend(["", "## Precision Conditioned", ""])
        conditioned = sensitivity.get("precision_conditioned", {})
        for precision, result in conditioned.items():
            physical = result.get("metrics", {}).get("physical_norm_only", {})
            weighted = result.get("metrics", {}).get(
                "sensitivity_weighted_error", {}
            )
            p = physical.get("classification", {})
            s = weighted.get("classification", {})
            physical_ground_truth = physical.get("ground_truth", {})
            sensitivity_ground_truth = weighted.get("ground_truth", {})
            physical_logit = physical_ground_truth.get("logit_delta_l2", {}).get(
                "spearman"
            )
            physical_margin = physical_ground_truth.get("margin_abs_delta", {}).get(
                "spearman"
            )
            sensitivity_logit = sensitivity_ground_truth.get(
                "logit_delta_l2", {}
            ).get("spearman")
            sensitivity_margin = sensitivity_ground_truth.get(
                "margin_abs_delta", {}
            ).get("spearman")
            lines.append(
                f"- `{precision}`: count={result.get('count')}, "
                f"physical KL/JS={p.get('spearman_vs_kl')}/{p.get('spearman_vs_js')}, "
                f"sensitivity KL/JS={s.get('spearman_vs_kl')}/"
                f"{s.get('spearman_vs_js')}, "
                f"physical logit-L2/margin={physical_logit}/{physical_margin}, "
                f"sensitivity logit-L2/margin="
                f"{sensitivity_logit}/{sensitivity_margin}"
            )
        lines.extend(["", "## Prompt/Context/Margin Stratification", ""])
        comparisons = sensitivity.get("sensitivity_vs_physical", {})
        for dimension, groups in (
            ("prompt", comparisons.get("by_prompt", {})),
            ("context", comparisons.get("by_context", {})),
            ("margin", comparisons.get("by_margin_bucket", {})),
        ):
            for key, transitions in groups.items():
                for transition, comparison in transitions.items():
                    kl = comparison.get("vs_delta_KL", {})
                    js = comparison.get("vs_delta_JS", {})
                    logit = comparison.get("vs_delta_logit_delta_l2", {})
                    margin = comparison.get("vs_delta_margin_abs", {})
                    lines.append(
                        f"- `{dimension}={key}` / `{transition}`: "
                        f"KL physical/sensitivity="
                        f"{kl.get('physical_spearman')}/"
                        f"{kl.get('sensitivity_spearman')}, "
                        f"JS physical/sensitivity="
                        f"{js.get('physical_spearman')}/"
                        f"{js.get('sensitivity_spearman')}, "
                        f"logit-L2/margin physical/sensitivity="
                        f"{logit.get('physical_spearman')}/"
                        f"{logit.get('sensitivity_spearman')}/"
                        f"{margin.get('physical_spearman')}/"
                        f"{margin.get('sensitivity_spearman')}, "
                        f"beats(KL/JS)="
                        f"{kl.get('sensitivity_beats_physical')}/"
                        f"{js.get('sensitivity_beats_physical')}"
                    )
        lines.extend(["", "## Layer-Conditioned", ""])
        layer_comparisons = comparisons.get("by_layer", {})
        layer_conditioned = sensitivity.get("layer_conditioned", {})
        for layer, transitions in layer_comparisons.items():
            count = layer_conditioned.get(layer, {}).get("count")
            for transition, comparison in transitions.items():
                kl = comparison.get("vs_delta_KL", {})
                js = comparison.get("vs_delta_JS", {})
                logit = comparison.get("vs_delta_logit_delta_l2", {})
                margin = comparison.get("vs_delta_margin_abs", {})
                lines.append(
                    f"- `layer={layer}` / `{transition}`: count={count}, "
                    f"KL physical/sensitivity="
                    f"{kl.get('physical_spearman')}/"
                    f"{kl.get('sensitivity_spearman')}, "
                    f"JS physical/sensitivity="
                    f"{js.get('physical_spearman')}/"
                    f"{js.get('sensitivity_spearman')}, "
                    f"logit-L2/margin physical/sensitivity="
                    f"{logit.get('physical_spearman')}/"
                    f"{logit.get('sensitivity_spearman')}/"
                    f"{margin.get('physical_spearman')}/"
                    f"{margin.get('sensitivity_spearman')}, "
                    f"beats(KL/JS)="
                    f"{kl.get('sensitivity_beats_physical')}/"
                    f"{js.get('sensitivity_beats_physical')}"
                )
        lines.extend(["", "## Cross-Layer Behavior", ""])
        for name, values in sensitivity.get("cross_layer", {}).get(
            "overall", {}
        ).items():
            lines.append(f"- `{name}`: `{json.dumps(values, sort_keys=True)}`")
        for dimension, groups in (
            ("precision", sensitivity.get("cross_layer", {}).get("by_precision", {})),
            ("context", sensitivity.get("cross_layer", {}).get("by_context", {})),
        ):
            for key, values in groups.items():
                lines.append(
                    f"- `{dimension}={key}`: "
                    f"`{json.dumps(values, sort_keys=True)}`"
                )
        lines.extend(
            [
                "",
                "## Criteria",
                "",
                f"- `{json.dumps(sensitivity.get('criteria', {}), sort_keys=True)}`",
            ]
        )
    lines.extend(["", "## Limitations", ""])
    for item in sensitivity.get("limitations", []):
        lines.append(f"- {item}")
    report_path = output / "sensitivity_report.md"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def write_validation_report(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a concise, evidence-oriented Markdown classification report."""
    validation = summary.get("validation") or assess_qdm_validation(summary)
    exact_validation = summary.get("exact_validation") or assess_exact_drift(
        summary
    )
    sensitivity = summary.get("downstream_sensitivity", {})
    sensitivity_status = (
        sensitivity.get("conclusion")
        if sensitivity.get("available", False)
        else None
    )
    report_status = sensitivity_status or exact_validation.get(
        "conclusion", "INCONCLUSIVE"
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis_version = summary.get("analysis_version", QDM_VALIDATION_ANALYSIS_VERSION)
    precision_counts = {
        key: value.get("count", 0)
        for key, value in summary.get("precision_buckets", {}).items()
    }
    context_counts = {
        key: value.get("count", 0)
        for key, value in summary.get("context_buckets", {}).items()
    }
    run_info = summary.get("run", {})
    integrity = summary.get("integrity", {})
    requested_contexts = json.dumps(run_info.get("requested_context_lengths", []))
    covered_contexts = json.dumps(run_info.get("covered_context_lengths", []))
    alignment_status = "/".join(
        str(bool(integrity.get(field, False)))
        for field in (
            "all_prefix_aligned",
            "all_suffix_aligned",
            "all_target_aligned",
        )
    )
    bf16_witness_zero = summary.get("bf16_witness", {}).get(
        "all_error_witness_zero", False
    )
    bf16_physical_zero = summary.get("bf16_witness", {}).get(
        "all_physical_drift_zero", False
    )
    quantized_witness = summary.get("quantized_overall", {}).get("witness", {})
    saturation = summary.get("saturation_report", {})
    quantized_saturation = saturation.get("quantized_overall", {})
    incremental = summary.get("incremental_value", {})
    enrichment = summary.get("quantized_overall", {}).get(
        "diagnostic_enrichment", {}
    )

    def _finite_rates(value: Any) -> list[float]:
        if not isinstance(value, Mapping):
            return []
        return [
            float(item["saturation_rate"])
            for item in value.values()
            if isinstance(item, Mapping)
            and isinstance(item.get("saturation_rate"), (int, float))
            and math.isfinite(float(item["saturation_rate"]))
        ]

    layer_saturation_rates = _finite_rates(
        saturation.get("saturation_rate_by_layer", {})
    )
    layer_head_saturation_rates = _finite_rates(
        saturation.get("saturation_rate_by_layer_head", {})
    )
    overall_saturation_rate = quantized_saturation.get("saturation_rate")
    if (
        layer_saturation_rates
        and layer_head_saturation_rates
        and isinstance(overall_saturation_rate, (int, float))
        and float(overall_saturation_rate) > 0.0
        and min(layer_saturation_rates) >= float(overall_saturation_rate) * 0.85
        and min(layer_head_saturation_rates)
        >= float(overall_saturation_rate) * 0.85
    ):
        saturation_conclusion = "broad_model_wide_saturation"
    elif layer_saturation_rates and layer_head_saturation_rates:
        saturation_conclusion = "mixed_or_concentrated_saturation"
    else:
        saturation_conclusion = "insufficient_stratified_saturation_data"
    lines = [
        "# QDM Validation Report",
        "",
        f"**Conclusion: {report_status}**",
        f"QDM Phase 1.5 validity status: `{validation['status']}`",
        "",
        f"Analysis version: `{analysis_version}`",
        f"Protocol: `{summary.get('protocol', VALIDATION_PROTOCOL)}`",
        "",
        "## Protocol",
        "",
        "- Teacher-forced prefix/suffix/target token hashes were checked.",
        "- BF16 is the reference; quantized paths use the same fixed token suffix.",
        "- Witnesses come from the production quantizer and its dequantization "
        "payloads.",
        "- Attention weights and full attention matrices are not materialized.",
        "- The appended teacher-forced suffix remains BF16 and is included in "
        "visible `v_norm_max`.",
        "",
        "## Run",
        "",
        f"- Target: `{run_info.get('model_target_path')}`",
        f"- Environment: `{run_info.get('environment_status')}`",
        f"- Successful prompts: `{run_info.get('sample_count_success')}`",
        "- Successful prompt/context runs: "
        f"`{run_info.get('sample_context_runs_success')}`",
        f"- Requested context lengths: `{requested_contexts}`",
        f"- Covered context lengths: `{covered_contexts}`",
        f"- Run failures: `{run_info.get('failure_count', 0)}`",
        "- Formula max absolute residual: "
        f"`{integrity.get('max_attention_error_formula_abs_error')}`",
        f"- Prefix/suffix/target alignment: `{alignment_status}`",
        "",
        "## Evidence",
        "",
        f"Rows: `{summary.get('integrity', {}).get('row_count', 0)}`",
        f"Precision buckets: `{json.dumps(precision_counts, sort_keys=True)}`",
        f"Context buckets: `{json.dumps(context_counts, sort_keys=True)}`",
        f"BF16 witness zero: `{bf16_witness_zero}`",
        f"BF16 physical QDM drift zero: `{bf16_physical_zero}`",
        f"Quantized witness max `v_norm`: `{quantized_witness.get('max_v_norm')}`",
        "",
        "Metric comparison (overall):",
        "",
        "The validity comparison below excludes the BF16 control; the control "
        "remains in `overall` and its zero invariants are reported separately.",
        "",
    ]
    metric_comparison = summary.get("quantized_overall", {}).get(
        "metric_comparison", {}
    )
    for metric in (*QDM_ANALYSIS_METRIC_NAMES, "top1_margin"):
        result = metric_comparison.get(metric, {})
        lines.append(
            f"- `{metric}`: Spearman(KL)={result.get('spearman_vs_kl')}, "
            f"Spearman(JS)={result.get('spearman_vs_js')}, "
            f"AUROC(flip)={result.get('auroc_top1_flip')}, "
            f"PR-AUC(flip)={result.get('pr_auc_top1_flip')}"
        )
    combined = summary.get("quantized_overall", {}).get("qdm_plus_margin", {})
    lines.extend(
        [
            f"- `QDM + small margin`: AUROC(flip)="
            f"{combined.get('auroc_top1_flip')}, "
            f"PR-AUC(flip)={combined.get('pr_auc_top1_flip')}",
            "",
            "Danger-token enrichment:",
            "",
            f"- QDM-only: `{enrichment.get('qdm_high_flip_enrichment')}`",
            f"- Small-margin-only: `{enrichment.get('small_margin_flip_enrichment')}`",
            f"- QDM-high + small-margin: `"
            f"{enrichment.get('qdm_high_small_margin_flip_enrichment')}`",
        ]
    )
    raw_a_summary = json.dumps(
        quantized_saturation.get("raw_A", {}), sort_keys=True
    )
    raw_tv_summary = json.dumps(
        quantized_saturation.get("raw_tv_bound", {}), sort_keys=True
    )
    log_a_summary = json.dumps(
        quantized_saturation.get("log_A", {}), sort_keys=True
    )
    lines.extend(
        [
            "",
            "## TV Saturation Diagnostics",
            "",
            "These values are diagnostic only. They do not replace the clamped "
            "production TV certificate.",
            "- Saturation diagnostic coverage: "
            f"`{quantized_saturation.get('diagnostic_coverage')}`",
            "- Observation saturation rate: "
            f"`{quantized_saturation.get('saturation_rate')}`",
            "- Token-level worst-observation saturation rate: "
            f"`{quantized_saturation.get('token_saturation_rate')}`",
            f"- `raw_A`: `{raw_a_summary}`",
            f"- `raw_tv_bound`: `{raw_tv_summary}`",
            f"- `log_A`: `{log_a_summary}`",
        ]
    )
    for name, values in (
        ("precision", saturation.get("saturation_rate_by_precision", {})),
        ("context", saturation.get("saturation_rate_by_context", {})),
    ):
        lines.append(
            f"- Saturation rate by {name}: "
            f"`{json.dumps(values, sort_keys=True)}`"
        )
    for field, label in (
        ("saturation_rate_by_layer", "layer"),
        ("saturation_rate_by_kv_head", "KV head"),
        ("saturation_rate_by_layer_head", "layer/KV head"),
    ):
        buckets = saturation.get(field, {})
        top = sorted(
            buckets.items(),
            key=lambda item: (
                float(item[1].get("saturation_rate") or 0.0),
                int(item[1].get("observation_count", 0)),
            ),
            reverse=True,
        )[:8]
        lines.append(
            f"- Highest saturation by {label}: "
            f"`{json.dumps(dict(top), sort_keys=True)}`"
        )
    incremental_overall = incremental.get("overall", {})
    incremental_metric = incremental_overall.get("by_qdm_metric", {}).get(
        "max_attention_error", {}
    )
    qdm_only_summary = json.dumps(
        incremental_metric.get("qdm_only", {}), sort_keys=True
    )
    margin_only_summary = json.dumps(
        incremental_metric.get("margin_only", {}), sort_keys=True
    )
    combined_summary = json.dumps(
        incremental_metric.get("margin_plus_qdm", {}), sort_keys=True
    )
    enrichment_summary = json.dumps(
        incremental_metric.get("top1_flip_enrichment", {}), sort_keys=True
    )
    conditioned_summary = json.dumps(
        incremental_metric.get("margin_conditioned", {}), sort_keys=True
    )
    conditioned = incremental_metric.get("margin_conditioned", {})
    supporting_evidence = validation.get("supporting_evidence", {})
    layer_saturation_min = (
        min(layer_saturation_rates) if layer_saturation_rates else None
    )
    layer_saturation_max = (
        max(layer_saturation_rates) if layer_saturation_rates else None
    )
    layer_head_saturation_min = (
        min(layer_head_saturation_rates) if layer_head_saturation_rates else None
    )
    layer_head_saturation_max = (
        max(layer_head_saturation_rates) if layer_head_saturation_rates else None
    )
    context_saturation_summary = json.dumps(
        saturation.get("saturation_rate_by_context", {}), sort_keys=True
    )
    incremental_gain_summary = json.dumps(
        incremental_metric.get("incremental_gain", {}), sort_keys=True
    )
    margin_controlled_evidence = validation.get("criteria", {}).get(
        "margin_controlled_incremental_value"
    )
    positive_precision_count = supporting_evidence.get(
        "positive_precision_bucket_count"
    )
    positive_context_count = supporting_evidence.get(
        "positive_context_bucket_count"
    )
    positive_incremental_precision_count = supporting_evidence.get(
        "positive_incremental_precision_bucket_count"
    )
    positive_incremental_context_count = supporting_evidence.get(
        "positive_incremental_context_bucket_count"
    )
    required_context_support = supporting_evidence.get(
        "required_context_bucket_support"
    )
    lines.extend(
        [
            "",
            "## Margin-Controlled Incremental Value",
            "",
            "Empirical quantiles are used only for this offline analysis; no "
            "production threshold is fitted.",
            f"- `max_attention_error` QDM-only: `{qdm_only_summary}`",
            f"- Margin-only: `{margin_only_summary}`",
            f"- Margin + QDM: `{combined_summary}`",
            f"- Flip enrichment comparison: `{enrichment_summary}`",
            f"- Margin-conditioned bins: `{conditioned_summary}`",
            "",
            "### Answer A: why TV saturates",
            "",
            f"- Diagnostic conclusion: `{saturation_conclusion}`.",
            "- Quantized observation saturation rate: "
            f"`{quantized_saturation.get('saturation_rate')}`; token-level "
            "worst-observation rate: "
            f"`{quantized_saturation.get('token_saturation_rate')}`.",
            f"- Layer saturation range across `{len(layer_saturation_rates)}` "
            f"layers: `{layer_saturation_min}`-`{layer_saturation_max}`.",
            f"- Layer/KV-head saturation range across "
            f"`{len(layer_head_saturation_rates)}` strata: "
            f"`{layer_head_saturation_min}`-`{layer_head_saturation_max}`.",
            f"- Context saturation rates: `{context_saturation_summary}`.",
            "This is a diagnostic interpretation only; it does not alter the "
            "clamped production certificate or introduce a production threshold.",
            "",
            "### Answer B: independent information after margin control",
            "",
            "- Overall classification evidence: "
            f"`{margin_controlled_evidence}`.",
            "- Overall incremental gain over the best single predictor: "
            f"`{incremental_gain_summary}`.",
            "- Margin-conditioned positive bins: "
            f"flip `{conditioned.get('positive_flip_support_count')}`/"
            f"`{conditioned.get('supported_bin_count')}`, "
            f"KL `{conditioned.get('positive_kl_support_count')}`/"
            f"`{conditioned.get('supported_bin_count')}`, "
            f"JS `{conditioned.get('positive_js_support_count')}`/"
            f"`{conditioned.get('supported_bin_count')}`.",
            "- Positive QDM correlation buckets: "
            f"precision `{positive_precision_count}`/4, "
            f"context `{positive_context_count}`/`{required_context_support}`; "
            "positive incremental buckets: "
            f"precision `{positive_incremental_precision_count}`/4, "
            f"context `{positive_incremental_context_count}`/"
            f"`{required_context_support}`.",
            "Conclusion: the aggregate and margin-conditioned results show "
            "physical drift signal, but the precision-conditioned support is "
            "not stable enough to claim validated independent value or enter "
            "kernel fusion.",
        ]
    )
    lines.extend(["", "## Precision Conditioning", ""])
    for precision in ("K2V2", "K4V2", "K8V4", "MIXED", "BF16"):
        bucket = summary.get("precision_buckets", {}).get(precision, {})
        means = bucket.get("means", {})
        lines.append(
            f"- `{precision}`: count={bucket.get('count', 0)}, "
            f"QDM={means.get('max_attention_error')}, "
            f"KL={means.get('kl_bf16_quantized')}, "
            f"JS={means.get('js_divergence')}, "
            f"flip_rate={means.get('top1_flip_rate')}"
        )
    conditioning = summary.get("precision_conditioning", {})
    lines.append(
        "- compression-order monotonicity (diagnostic only): "
        f"QDM={conditioning.get('mean_qdm_nonincreasing_with_compression')}, "
        f"KL={conditioning.get('mean_kl_nonincreasing_with_compression')}, "
        f"JS={conditioning.get('mean_js_nonincreasing_with_compression')}"
    )
    lines.extend(["", "## Stratified Validity", ""])
    lines.append(
        "Per-bucket correlations and combined scores are diagnostic only; they "
        "are not production thresholds."
    )
    for dimension, groups in (
        ("precision", summary.get("precision_buckets", {})),
        ("context", summary.get("quantized_context_buckets", {})),
    ):
        incremental_groups = incremental.get(f"by_{dimension}", {})
        for key, bucket in groups.items():
            metric = bucket.get("metric_comparison", {}).get(
                "max_attention_error", {}
            )
            incremental_bucket = (
                incremental_groups.get(str(key), {})
                .get("by_qdm_metric", {})
                .get("max_attention_error", {})
            )
            combined_bucket = incremental_bucket.get("margin_plus_qdm", {})
            lines.append(
                f"- `{dimension}={key}`: count={bucket.get('count', 0)}, "
                f"QDM Spearman(KL/JS)="
                f"{metric.get('spearman_vs_kl')}/{metric.get('spearman_vs_js')}, "
                f"QDM AUROC/PR-AUC="
                f"{metric.get('auroc_top1_flip')}/{metric.get('pr_auc_top1_flip')}, "
                f"margin+QDM AUROC/PR-AUC="
                f"{combined_bucket.get('auroc_top1_flip')}/"
                f"{combined_bucket.get('pr_auc_top1_flip')}"
            )
    lines.extend(["", "## Layer/Head Stratification", ""])
    stratification = summary.get("layer_head_stratification", {})
    for dimension in ("layer", "kv_head", "layer_kv_head"):
        result = stratification.get(dimension, {})
        lines.append(
            f"- `{dimension}`: observed={result.get('observed_strata')}, "
            f"supported={result.get('supported_strata')}, "
            f"positive_qdm_vs_kl={result.get('positive_qdm_vs_kl_supported_strata')}, "
            f"count_min/median/max={result.get('count_min')}/"
            f"{result.get('count_median')}/{result.get('count_max')}"
        )
    layer_head_buckets = summary.get("quantized_layer_head_buckets", {})
    top_layer_heads = sorted(
        layer_head_buckets.items(),
        key=lambda item: int(item[1].get("count", 0)),
        reverse=True,
    )[:8]
    for key, bucket in top_layer_heads:
        means = bucket.get("means", {})
        lines.append(
            f"- `{key}`: count={bucket.get('count')}, "
            f"QDM={means.get('max_attention_error')}, "
            f"KL={means.get('kl_bf16_quantized')}, "
            f"JS={means.get('js_divergence')}, "
            f"flip_rate={means.get('top1_flip_rate')}"
        )
    for key in (
        "all_flip_rate",
        "qdm_high_flip_rate",
        "small_margin_flip_rate",
        "qdm_high_small_margin_flip_rate",
        "qdm_high_small_margin_flip_enrichment",
        "qdm_high_small_margin_vs_qdm_high_enrichment",
        "qdm_high_small_margin_vs_small_margin_enrichment",
    ):
        lines.append(f"- `{key}`: `{enrichment.get(key)}`")
    exact_report = summary.get("exact_drift_analysis", {})
    tightness_report = summary.get("bound_tightness", {})
    lines.extend(["", "## Exact Query-Conditioned Drift", ""])
    lines.append(
        "Exact metrics are offline BF16-query oracles using the actual production "
        "restore; they are not runtime QDM metrics or token-error certificates."
    )
    lines.append(
        f"- Phase 1.5 exact-drift diagnostic conclusion: `"
        f"{exact_validation.get('conclusion')}`"
    )
    exact_integrity = integrity.get("exact_oracle", {})
    lines.append(
        "- Exact oracle integrity: "
        f"available={exact_integrity.get('available')}, "
        "probability_aligned="
        f"{exact_integrity.get('reference_block_probability_aligned')}, "
        f"BF16_zero={exact_integrity.get('bf16_exact_zero')}, "
        f"max_probability_error="
        f"{exact_integrity.get('max_reference_block_probability_abs_error')}"
    )
    if not exact_report.get("available", False):
        lines.append("- Exact oracle fields are unavailable in this artifact.")
    else:
        exact_quantized = exact_report.get("quantized_overall", {})
        exact_metrics = exact_quantized.get("metrics", {})
        for metric in EXACT_DRIFT_METRIC_NAMES:
            result = exact_metrics.get(metric, {})
            comparison = result.get("metric_comparison", {})
            combined_exact = result.get("margin_plus_exact", {})
            lines.append(
                f"- `{metric}`: Spearman(KL/JS/flip)="
                f"{result.get('spearman_vs_kl')}/"
                f"{result.get('spearman_vs_js')}/"
                f"{result.get('spearman_vs_top1_flip')}, "
                f"AUROC/PR-AUC="
                f"{comparison.get('auroc_top1_flip')}/"
                f"{comparison.get('pr_auc_top1_flip')}, "
                f"margin+exact AUROC/PR-AUC="
                f"{combined_exact.get('auroc_top1_flip')}/"
                f"{combined_exact.get('pr_auc_top1_flip')}"
            )
        lines.append(
            "- Exact precision/context stratification (output error):"
        )
        for dimension, groups in (
            ("precision", exact_report.get("by_precision", {})),
            ("context", exact_report.get("by_context", {})),
        ):
            for key, bucket in groups.items():
                result = bucket.get("metrics", {}).get(
                    "exact_attention_output_error", {}
                )
                comparison = result.get("metric_comparison", {})
                lines.append(
                    f"- `{dimension}={key}`: count={bucket.get('count', 0)}, "
                    f"Spearman(KL/JS)="
                    f"{result.get('spearman_vs_kl')}/"
                    f"{result.get('spearman_vs_js')}, "
                    f"AUROC/PR-AUC="
                    f"{comparison.get('auroc_top1_flip')}/"
                    f"{comparison.get('pr_auc_top1_flip')}"
                )
        for field, label in (
            ("by_exact_worst_layer", "exact worst layer"),
            ("by_exact_worst_kv_head", "exact worst KV head"),
            ("by_exact_worst_layer_head", "exact worst layer/head"),
        ):
            groups = exact_report.get(field, {})
            top = sorted(
                groups.items(),
                key=lambda item: int(item[1].get("count", 0)),
                reverse=True,
            )[:8]
            for key, bucket in top:
                result = bucket.get("metrics", {}).get(
                    "exact_attention_output_error", {}
                )
                lines.append(
                    f"- `{label}={key}`: count={bucket.get('count', 0)}, "
                    f"mean_exact_output="
                    f"{result.get('mean')}, "
                    f"Spearman(KL/JS)="
                    f"{result.get('spearman_vs_kl')}/"
                    f"{result.get('spearman_vs_js')}"
                )
    lines.extend(["", "## Bound Tightness Decomposition", ""])
    lines.append(
        "The ratios below are offline diagnostics for max-L2 -> Cauchy -> "
        "block-max -> TV transform looseness; they do not replace `raw_tv_bound` "
        "or the clamped production TV certificate."
    )
    if not tightness_report.get("available", False):
        lines.append("- Bound tightness fields are unavailable in this artifact.")
    else:
        tight_quantized = tightness_report.get("quantized_overall", {})
        lines.append(
            f"- Quantized saturation rate: `{tight_quantized.get('saturation_rate')}`"
        )
        for metric in BOUND_TIGHTNESS_METRIC_NAMES:
            distribution = tight_quantized.get("metrics", {}).get(metric, {}).get(
                "distribution", {}
            )
            lines.append(f"- `{metric}` distribution: `{json.dumps(distribution)}`")
        lines.append(
            "- Production raw TV distribution: `"
            f"{json.dumps(tight_quantized.get('production_raw_tv_bound', {}))}`"
        )
        lines.append(
            "- Exact attention TV distribution: `"
            f"{json.dumps(tight_quantized.get('exact_attention_TV', {}))}`"
        )
        production_output_distribution = json.dumps(
            tight_quantized.get("production_attention_error_bound", {})
        )
        lines.append(
            "- Production output-bound distribution: `"
            f"{production_output_distribution}`"
        )
        lines.append(
            "- Exact output-error distribution: `"
            f"{json.dumps(tight_quantized.get('exact_attention_output_error', {}))}`"
        )
        for dimension, groups in (
            ("precision", tightness_report.get("by_precision", {})),
            ("context", tightness_report.get("by_context", {})),
        ):
            for key, bucket in groups.items():
                ratio = bucket.get("metrics", {}).get(
                    "tv_bound_exact_ratio", {}
                ).get("distribution", {})
                output_ratio = bucket.get("metrics", {}).get(
                    "output_bound_exact_ratio", {}
                ).get("distribution", {})
                lines.append(
                    f"- `{dimension}={key}`: saturation="
                    f"{bucket.get('saturation_rate')}, "
                    f"TV bound/exact p95={ratio.get('p95')}, "
                    f"output bound/exact p95={output_ratio.get('p95')}"
                )
    exact_support = exact_validation.get("supporting_evidence", {})
    lines.extend(
        [
            "",
            "## Exact Conclusion Criteria",
            "",
            "- `criteria`: `"
            f"{json.dumps(exact_validation.get('criteria', {}), sort_keys=True)}`",
            f"- `supporting_evidence`: `{json.dumps(exact_support, sort_keys=True)}`",
            "- `WITNESS_TOO_LOOSE` requires stable exact precision/margin "
            "incremental value plus clear production-bound looseness.",
            "- Otherwise the result remains `INCONCLUSIVE` unless exact metrics "
            "also lack physical signal, in which case it is `SIGNAL_NOT_USEFUL`.",
        ]
    )
    paired = summary.get("paired_counterfactual", {})
    lines.extend(
        [
            "",
            "## Paired Precision Counterfactual",
            "",
            "Same-token precision deltas are reported separately in "
            "`paired_counterfactual_report.md`; they are not pooled token "
            "correlations.",
            f"- Conclusion: `{paired.get('conclusion', 'INCONCLUSIVE')}`",
            f"- Paired rows: `{paired.get('paired_token_count', 0)}`",
            f"- Integrity: `{json.dumps(paired.get('integrity', {}), sort_keys=True)}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Downstream Sensitivity Oracle",
            "",
            "Gradient/JVP projections are offline-only and are not production "
            "QDM metrics.",
            f"- Conclusion: `{sensitivity.get('conclusion', 'INCONCLUSIVE')}`",
            f"- Available: `{sensitivity.get('available', False)}`",
            "- Integrity: `"
            f"{json.dumps(sensitivity.get('integrity', {}), sort_keys=True)}`",
            "- Criteria: `"
            f"{json.dumps(sensitivity.get('criteria', {}), sort_keys=True)}`",
        ]
    )
    lines.extend(["", "## Classification Criteria", ""])
    for key, value in validation.get("criteria", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if validation.get("structural_failures"):
        lines.extend(["", "Structural failures:", ""])
        for item in validation["structural_failures"]:
            lines.append(f"- {item}")
    lines.extend(["", "Limitations:", ""])
    for item in validation.get("limitations", []):
        lines.append(f"- {item}")
    report_path = output / "validation_report.md"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def write_qdm_validation_artifacts(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the per-token JSONL and summary JSON artifacts."""
    normalized = [dict(row) for row in rows]
    _assign_diagnostic_labels(normalized)
    summary = summarize_qdm_validation_rows(normalized)
    summary["validation"] = assess_qdm_validation(summary)
    summary["exact_validation"] = assess_exact_drift(summary)
    paired_analysis = build_paired_precision_analysis(normalized)
    summary["paired_counterfactual"] = paired_analysis["summary"]
    sensitivity_analysis = build_downstream_sensitivity_analysis(normalized)
    summary["downstream_sensitivity"] = sensitivity_analysis["summary"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "per_token_qdm.jsonl"
    temporary_jsonl = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with temporary_jsonl.open("w", encoding="utf-8") as handle:
        for row in normalized:
            public_row = {
                key: value for key, value in row.items() if key != "_layer_drift_trace"
            }
            handle.write(
                json.dumps(public_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary_jsonl.replace(jsonl_path)

    summary_path = output / "summary.json"
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    for filename, payload in (
        ("saturation_report.json", summary["saturation_report"]),
        ("incremental_value.json", summary["incremental_value"]),
        ("bound_tightness.json", summary["bound_tightness"]),
        ("exact_drift_analysis.json", summary["exact_drift_analysis"]),
        ("downstream_sensitivity.json", summary["downstream_sensitivity"]),
    ):
        report_path = output / filename
        temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_report.replace(report_path)
    paired_jsonl_path = output / "paired_precision_qdm.jsonl"
    temporary_paired_jsonl = paired_jsonl_path.with_suffix(
        paired_jsonl_path.suffix + ".tmp"
    )
    with temporary_paired_jsonl.open("w", encoding="utf-8") as handle:
        for record in paired_analysis["records"]:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_paired_jsonl.replace(paired_jsonl_path)
    precision_monotonicity_path = output / "precision_monotonicity.json"
    temporary_monotonicity = precision_monotonicity_path.with_suffix(
        precision_monotonicity_path.suffix + ".tmp"
    )
    temporary_monotonicity.write_text(
        json.dumps(
            summary["paired_counterfactual"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_monotonicity.replace(precision_monotonicity_path)
    layer_trace_path = output / "layer_drift_trace.jsonl"
    temporary_layer_trace = layer_trace_path.with_suffix(
        layer_trace_path.suffix + ".tmp"
    )
    with temporary_layer_trace.open("w", encoding="utf-8") as handle:
        for row in normalized:
            trace = row.get("_layer_drift_trace")
            if not isinstance(trace, Mapping):
                continue
            trace_row = {
                "sample": row.get("sample"),
                "context_length": row.get("context_length"),
                "requested_context_length": row.get("requested_context_length"),
                "step": row.get("step"),
                "precision_composition": row.get("precision_composition"),
                "suffix_input_token_id": row.get("suffix_input_token_id"),
                "target_token_id": row.get("target_token_id"),
                "teacher_forced": row.get("teacher_forced"),
                "free_running_ground_truth": row.get("free_running_ground_truth"),
                "alignment_hashes": {
                    field: row.get(field) for field in PAIRED_ALIGNMENT_FIELDS
                },
                "trace": trace,
            }
            handle.write(
                json.dumps(trace_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary_layer_trace.replace(layer_trace_path)
    write_paired_counterfactual_report(summary, output)
    write_downstream_sensitivity_report(summary, output)
    write_validation_report(summary, output)
    return summary


__all__ = [
    "PRECISION_COMPOSITIONS",
    "QDM_METRIC_NAMES",
    "QDM_DIAGNOSTIC_METRIC_NAMES",
    "QDM_ANALYSIS_METRIC_NAMES",
    "EXACT_DRIFT_METRIC_NAMES",
    "BOUND_TIGHTNESS_METRIC_NAMES",
    "PAIRED_PRECISION_ORDER",
    "PAIRED_REQUIRED_PRECISIONS",
    "PAIRED_MONOTONIC_METRICS",
    "PAIRED_ANALYSIS_VERSION",
    "SENSITIVITY_ANALYSIS_VERSION",
    "SENSITIVITY_FEATURES",
    "SENSITIVITY_OUTCOMES",
    "QDM_MINIMUM_VALIDATION_PROMPTS",
    "QDM_VALIDATION_ANALYSIS_VERSION",
    "QDM_ATTENTION_IMPLEMENTATION",
    "QDMBlockObservation",
    "KV_COMPRESSION_SCOPE",
    "StreamingQDMCollector",
    "VALIDATION_PROTOCOL",
    "aggregate_exact_attention_step",
    "aggregate_qdm_step",
    "assert_teacher_forced_prefix_alignment",
    "assert_teacher_forced_sequence_alignment",
    "assess_qdm_validation",
    "assess_exact_drift",
    "build_bound_tightness_report",
    "build_exact_drift_report",
    "build_incremental_value_report",
    "build_paired_precision_analysis",
    "build_downstream_sensitivity_analysis",
    "build_saturation_report",
    "compute_exact_attention_drift",
    "compute_layer_drift_trace",
    "make_teacher_forced_rows",
    "qdm_reference_attention",
    "qdm_streaming_attention",
    "summarize_qdm_validation_rows",
    "teacher_forced_logit_metrics",
    "write_qdm_validation_artifacts",
    "write_paired_counterfactual_report",
    "write_downstream_sensitivity_report",
    "write_validation_report",
]
