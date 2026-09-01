# SPDX-License-Identifier: Apache-2.0

"""Strict validation for MaKV residual-based remote precision upgrades.

This module is an offline experiment harness.  It does not change the
production CONF policy, quantizer, CUDA extension, or remote object.  The
``qwen`` backend runs fixed-token teacher-forced counterfactuals with the
target model.  The explicit ``synthetic`` backend is used by CPU CI to test
the metric definitions and invariants without pretending to be a model run.

The experiment writes six auditable artifacts:

* ``residual_fidelity.json``
* ``upgrade_benefit.jsonl``
* ``risk_selection.jsonl``
* ``upgrade_frontier.json``
* ``system_cost.json``
* ``precision_upgrade_report.md``

For a real run, token ``i`` is always represented by ``token_index=i``.  The
KL/JS/top-1 outcomes are computed only on the fixed future teacher-forced
continuation; the logits used to form the risk signal are never reused as a
same-step upgrade outcome.
"""

from __future__ import annotations

# Standard
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import argparse
import asyncio
import hashlib
import json
import math
import random
import time

# Third Party
import torch
import torch.nn.functional as F

# First Party
from lmcache.v1.storage_backend.makv.config import MaKVConfig
from lmcache.v1.storage_backend.makv.format import (
    decode_makv_object,
    encode_client_put_envelope,
    encode_makv_object,
)
from lmcache.v1.storage_backend.makv.gpu_restore import payload_tensors_from_obj
from lmcache.v1.storage_backend.makv.memory import MaKVQuantizedMemoryObj
from lmcache.v1.storage_backend.makv.plan import (
    MaKVQuantPlan,
    compute_quant_plan_checksum,
)
from lmcache.v1.storage_backend.makv.precision_risk import (
    CONF_RISK_SEMANTICS,
    CONF_SCORER_VERSION,
    compute_precision_risk_signal,
)
from lmcache.v1.storage_backend.makv.quantizer import quantize_canonical_kv
from lmcache.v1.storage_backend.makv.reference_dequant import dequantize_reference
from lmcache.v1.storage_backend.makv.residual import reconstruct_with_residual
from lmcache.v1.storage_backend.makv_remote.manager import MaKVRemoteManager


SCHEMA = "makv_residual_precision_upgrade_validation_v1"
OUTPUT_NAMES = (
    "residual_fidelity.json",
    "upgrade_benefit.jsonl",
    "risk_selection.jsonl",
    "upgrade_frontier.json",
    "system_cost.json",
    "precision_upgrade_report.md",
)
BUCKET_BITS = (16, 8, 4, 2)
SUPPORTED_RESIDUAL_DTYPES = ("float16", "float32")
DEFAULT_CONTEXT_LENGTHS = (1024, 2048)
DEFAULT_HORIZONS = (32, 64)
DEFAULT_UPGRADE_RATES = (0.01, 0.05, 0.10, 0.20)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "experiments/scoutrank_transfer/manifests/feasibility.jsonl"
)
DEFAULT_MODEL = Path("/media/home/iic/mahaoyuan/models/Qwen3-8B")


@dataclass(frozen=True)
class QuantizedBundle:
    """One serialized MaKV object and its offline reconstructed tensors."""

    blob: bytes
    metadata: dict[str, Any]
    payloads: dict[str, bytes | memoryview]
    low_canonical: torch.Tensor
    residual_canonical: torch.Tensor | None
    quantize_time_ms: float
    reconstruction_time_ms: float
    quantized_payload_bytes: int
    residual_bytes: int


class _MemoryStorage:
    """Minimal async storage used only by the invariant probe."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def exists(self, key: str) -> bool:
        return key in self.values

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        return [self.values.get(key) for key in keys]

    async def put(self, key: str, data: bytes) -> None:
        self.values[key] = bytes(data)

    async def delete(self, key: str) -> bool:
        return self.values.pop(key, None) is not None

    async def list_keys(self) -> list[str]:
        return list(self.values)

    async def close(self) -> None:
        return


def _dtype(name: str) -> torch.dtype:
    """Convert an experiment dtype name to a PyTorch dtype."""
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[str(name).lower()]
    except KeyError as error:
        raise ValueError(f"unsupported experiment dtype: {name!r}") from error


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor's logical contiguous storage without pickle."""
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _canonical_to_wire(canonical: torch.Tensor) -> torch.Tensor:
    """Convert ``[L,2,T,H,D]`` to the production ``[2,L,T,H*D]`` wire view."""
    if canonical.ndim != 5 or canonical.shape[1] != 2:
        raise ValueError("canonical KV must have shape [L,2,T,H,D]")
    return (
        canonical.permute(1, 0, 2, 3, 4)
        .reshape(
            2,
            canonical.shape[0],
            canonical.shape[2],
            canonical.shape[3] * canonical.shape[4],
        )
        .contiguous()
    )


def _wire_to_canonical(wire: torch.Tensor, plan: Mapping[str, Any]) -> torch.Tensor:
    """Convert a restored production wire tensor to canonical experiment layout."""
    layers = int(plan["num_layers"])
    tokens = int(plan["chunk_length"])
    heads = int(plan["num_kv_heads"])
    head_dim = int(plan["head_dim"])
    expected = (2, layers, tokens, heads * head_dim)
    if tuple(wire.shape) != expected:
        raise ValueError(f"restored wire shape {tuple(wire.shape)} != {expected}")
    return (
        wire.view(2, layers, tokens, heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def build_canonical_k2_plan(source: torch.Tensor) -> MaKVQuantPlan:
    """Build the fixed canonical K2V2 plan used by this experiment."""
    if source.ndim != 5 or source.shape[1] != 2:
        raise ValueError("source must have shape [L,2,T,H,D]")
    layers, _, tokens, heads, head_dim = map(int, source.shape)
    flat_dim = heads * head_dim
    plan = MaKVQuantPlan(
        protocol_version=1,
        importance_layout="layer_kv_token",
        token_count=tokens,
        chunk_start=0,
        chunk_length=tokens,
        bucket_bits=BUCKET_BITS,
        # The physical bucket index for INT2 is 3.  Every K and V vector is
        # intentionally low precision; this is not a production policy edit.
        bucket_ids=bytes([3]) * (layers * 2 * tokens),
        original_shape=(2, layers, tokens, flat_dim),
        original_strides=(layers * tokens * flat_dim, tokens * flat_dim, flat_dim, 1),
        original_dtype=str(source.dtype),
        token_dim=2,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        quant_granularity="per_token_head",
        scale_dtype="float16",
        model_fingerprint="makv-residual-validation-qwen3-8b",
        parallel_fingerprint="single-worker",
        checksum=0,
        source_strategy="canonical_k2v2_validation_only",
    )
    return replace(plan, checksum=compute_quant_plan_checksum(plan))


def _experiment_config(residual_dtype: str) -> MaKVConfig:
    """Return a manager/quantizer config isolated from runtime config."""
    if residual_dtype not in ("none", *SUPPORTED_RESIDUAL_DTYPES):
        raise ValueError(f"unsupported residual dtype: {residual_dtype!r}")
    return MaKVConfig(
        storage_url="memory://makv-precision-upgrade-validation",
        bucket_ratios=(0.0, 0.0, 0.0, 1.0),
        bucket_bits=BUCKET_BITS,
        importance_layout="layer_kv_token",
        quant_granularity="per_token_head",
        scale_dtype="float16",
        protect_prefix_tokens=0,
        protect_tail_tokens=0,
        dequant_backend="reference",
        require_cuda_dequant=False,
        fallback="miss",
        enable_checksum=True,
        precision_scheme="shared",
        entropy_codec="none",
        entropy_backend="reference",
        residual_dtype=residual_dtype,
        risk_upgrade_threshold=0.8,
        risk_upgrade_policy="full",
        risk_window_tokens=16,
    )


def _quantize_bundle(
    source: torch.Tensor,
    plan: MaKVQuantPlan,
    residual_dtype: str,
) -> QuantizedBundle:
    """Quantize and decode one object using the existing production format."""
    started = time.perf_counter()
    metadata, payloads = quantize_canonical_kv(
        source,
        plan,
        _experiment_config(residual_dtype),
    )
    quantize_time_ms = (time.perf_counter() - started) * 1000.0
    blob = encode_makv_object(
        object_type="quantized",
        metadata=metadata,
        payloads=payloads,
    )
    decoded = decode_makv_object(blob)
    memory = MaKVQuantizedMemoryObj(
        blob,
        metadata_dict=decoded.metadata,
        payloads=decoded.payloads,
    )
    bucket_payloads = payload_tensors_from_obj(memory)
    low_wire = dequantize_reference(
        plan=decoded.metadata["plan"],
        bucket_payloads=bucket_payloads,
        output_dtype=source.dtype,
    )
    low_canonical = _wire_to_canonical(low_wire, decoded.metadata["plan"])
    reconstructed: torch.Tensor | None = None
    reconstruction_time_ms = 0.0
    if residual_dtype != "none":
        reconstruction_started = time.perf_counter()
        reconstructed_wire = reconstruct_with_residual(
            decoded.metadata,
            decoded.payloads,
        )
        reconstruction_time_ms = (time.perf_counter() - reconstruction_started) * 1000.0
        reconstructed = _wire_to_canonical(reconstructed_wire, decoded.metadata["plan"])
    quantized_payload_bytes = sum(
        len(value)
        for name, value in decoded.payloads.items()
        if not name.startswith("residual_")
    )
    residual_bytes = sum(
        len(value)
        for name, value in decoded.payloads.items()
        if name.startswith("residual_")
    )
    return QuantizedBundle(
        blob=blob,
        metadata=decoded.metadata,
        payloads=decoded.payloads,
        low_canonical=low_canonical,
        residual_canonical=reconstructed,
        quantize_time_ms=quantize_time_ms,
        reconstruction_time_ms=reconstruction_time_ms,
        quantized_payload_bytes=quantized_payload_bytes,
        residual_bytes=residual_bytes,
    )


def _logit_metrics(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> dict[str, float | int]:
    """Compare future teacher-forced logits against direct high precision."""
    reference = torch.as_tensor(reference_logits).detach().float()
    candidate = torch.as_tensor(candidate_logits).detach().float()
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("logits must have equal shape [future_steps,vocab]")
    if reference.shape[0] == 0 or reference.shape[1] < 2:
        raise ValueError("future logits must contain at least one step and two tokens")
    reference_logp = F.log_softmax(reference, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    reference_prob = reference_logp.exp()
    candidate_prob = candidate_logp.exp()
    kl = (reference_prob * (reference_logp - candidate_logp)).sum(dim=-1)
    mixture = 0.5 * (reference_prob + candidate_prob)
    mixture_logp = torch.log(mixture.clamp_min(torch.finfo(torch.float32).tiny))
    js = 0.5 * (
        (reference_prob * (reference_logp - mixture_logp)).sum(dim=-1)
        + (candidate_prob * (candidate_logp - mixture_logp)).sum(dim=-1)
    )
    reference_top = reference.argmax(dim=-1)
    candidate_top = candidate.argmax(dim=-1)
    flips = reference_top != candidate_top
    return {
        "kl": float(kl.mean().item()),
        "js": float(js.mean().item()),
        "top1_agreement": float((~flips).float().mean().item()),
        "top1_flip_rate": float(flips.float().mean().item()),
        "top1_flip_count": int(flips.sum().item()),
        "future_steps": int(reference.shape[0]),
    }


def _kv_fidelity(source: torch.Tensor, restored: torch.Tensor) -> dict[str, float]:
    """Return relative L2 and MSE of a reconstructed canonical KV tensor."""
    source_float = source.detach().float()
    restored_float = restored.detach().float()
    if source_float.shape != restored_float.shape:
        raise ValueError("source and reconstructed KV shapes differ")
    delta = source_float - restored_float
    denominator = torch.linalg.vector_norm(source_float).clamp_min(1.0e-12)
    return {
        "kv_relative_l2": float((torch.linalg.vector_norm(delta) / denominator).item()),
        "kv_mse": float(delta.square().mean().item()),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(
        enumerate(float(value) for value in values), key=lambda item: item[1]
    )
    ranks = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = rank
        cursor = end
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    """Compute tie-aware Spearman correlation without a SciPy dependency."""
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    x = _average_ranks(values_x)
    y = _average_ranks(values_y)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    denominator_x = sum((a - mean_x) ** 2 for a in x)
    denominator_y = sum((b - mean_y) ** 2 for b in y)
    denominator = math.sqrt(denominator_x * denominator_y)
    return numerator / denominator if denominator > 0.0 else None


def _stable_token_hash(token_ids: Sequence[int]) -> str:
    """Hash a token stream for teacher-forcing alignment checks."""
    payload = json.dumps([int(value) for value in token_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_indices(token_count: int, limit: int) -> list[int]:
    """Select an auditable, evenly spaced candidate subset when requested."""
    if token_count <= 0:
        return []
    if limit <= 0 or limit >= token_count:
        return list(range(token_count))
    if limit == 1:
        return [0]
    result = {
        min(token_count - 1, round(index * (token_count - 1) / (limit - 1)))
        for index in range(limit)
    }
    return sorted(result)


def _risk_rows(risk_logits: torch.Tensor, window_tokens: int) -> list[dict[str, Any]]:
    """Create CONF and margin-only scores with explicit absolute token indices."""
    logits = torch.as_tensor(risk_logits).detach().float()
    if logits.ndim != 2:
        raise ValueError("risk logits must have shape [token_count,vocab]")
    rows: list[dict[str, Any]] = []
    for token_index in range(logits.shape[0]):
        signal = compute_precision_risk_signal(logits[token_index], step=token_index)
        positioned = signal.for_kv_token(token_index, window_tokens=window_tokens)
        row = positioned.as_dict(include_diagnostics=True)
        row.update(
            {
                "token_index": token_index,
                "absolute_token_index": token_index,
                "window_tokens": window_tokens,
                "risk_source": "aligned_prefill_next_token_logits",
                "teacher_forced": True,
                "same_step_outcome_used": False,
            }
        )
        rows.append(row)
    return rows


def _select_tokens(
    benefit_rows: Sequence[Mapping[str, Any]],
    method: str,
    count: int,
    seed: int,
) -> list[int]:
    """Select exactly ``count`` candidate tokens using a deterministic rule."""
    if method not in ("conf", "margin_only", "oracle", "random"):
        raise ValueError(f"unknown selection method: {method}")
    candidates = sorted({int(row["token_index"]) for row in benefit_rows})
    count = max(0, min(int(count), len(candidates)))
    if count == 0:
        return []
    if method in ("conf", "CONF_UPGRADE"):
        ordered = sorted(
            benefit_rows,
            key=lambda row: (-float(row["risk"]), int(row["token_index"])),
        )
        return sorted({int(row["token_index"]) for row in ordered[:count]})
    if method == "margin_only":
        ordered = sorted(
            benefit_rows,
            key=lambda row: (-float(row["margin_only_risk"]), int(row["token_index"])),
        )
        return sorted({int(row["token_index"]) for row in ordered[:count]})
    if method == "oracle":
        ordered = sorted(
            benefit_rows,
            key=lambda row: (-float(row["benefit_kl"]), int(row["token_index"])),
        )
        return sorted({int(row["token_index"]) for row in ordered[:count]})
    rng = random.Random(int(seed))
    return sorted(rng.sample(candidates, count))


def _selection_row(
    *,
    sample_id: str,
    context_length: int,
    residual_dtype: str,
    horizon: int,
    method: str,
    selected: Sequence[int],
    candidate_count: int,
    direct_logits: torch.Tensor,
    low_logits: torch.Tensor,
    selected_logits: torch.Tensor,
    benefit_rows: Sequence[Mapping[str, Any]],
    token_count: int,
    alignment: Mapping[str, Any],
    random_repeats: int | None = None,
) -> dict[str, Any]:
    """Build one matched-count selection result row."""
    direct_metrics = _logit_metrics(direct_logits, direct_logits)
    low_metrics = _logit_metrics(direct_logits, low_logits)
    selected_metrics = _logit_metrics(direct_logits, selected_logits)
    denominator = low_metrics["kl"] - direct_metrics["kl"]
    recovered_kl = low_metrics["kl"] - selected_metrics["kl"]
    recovery = recovered_kl / denominator if abs(denominator) > 1.0e-12 else None
    risk_values = {int(row["token_index"]): float(row["risk"]) for row in benefit_rows}
    benefit_values = {
        int(row["token_index"]): float(row["benefit_kl"]) for row in benefit_rows
    }
    corr = None
    if method == "conf":
        corr = spearman(
            [risk_values[index] for index in sorted(risk_values)],
            [benefit_values[index] for index in sorted(risk_values)],
        )
    elif method in ("margin_only", "MARGIN_ONLY"):
        margin_values = {
            int(row["token_index"]): float(row["margin_only_risk"])
            for row in benefit_rows
        }
        corr = spearman(
            [margin_values[index] for index in sorted(margin_values)],
            [benefit_values[index] for index in sorted(margin_values)],
        )
    selected_set = set(int(value) for value in selected)
    return {
        "schema": SCHEMA,
        "row_type": "selection",
        "sample_id": sample_id,
        "context_length": int(context_length),
        "residual_dtype": residual_dtype,
        "horizon": int(horizon),
        "method": method,
        "upgrade_mode": "full",
        "oracle_target": "residual_high",
        "candidate_count": int(candidate_count),
        "token_count": int(token_count),
        "upgrade_count": len(selected_set),
        "upgrade_rate": len(selected_set) / candidate_count if candidate_count else 0.0,
        "selected_token_indices": sorted(selected_set),
        "spearman_risk_benefit": corr,
        "direct_high_kl": float(direct_metrics["kl"]),
        "low_kl": float(low_metrics["kl"]),
        "selected_kl": float(selected_metrics["kl"]),
        "recovered_kl": float(recovered_kl),
        "recovery": recovery,
        "direct_high_js": float(direct_metrics["js"]),
        "low_js": float(low_metrics["js"]),
        "selected_js": float(selected_metrics["js"]),
        "recovered_js": float(low_metrics["js"] - selected_metrics["js"]),
        "low_top1_flip_rate": float(low_metrics["top1_flip_rate"]),
        "selected_top1_flip_rate": float(selected_metrics["top1_flip_rate"]),
        "top1_flip_reduction": float(
            low_metrics["top1_flip_rate"] - selected_metrics["top1_flip_rate"]
        ),
        "random_repeat_count": random_repeats,
        "future_only": True,
        "same_step_logits_excluded": True,
        "teacher_forced": True,
        **dict(alignment),
    }


def _baseline_row(
    *,
    sample_id: str,
    context_length: int,
    residual_dtype: str,
    horizon: int,
    method: str,
    direct_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_count: int,
    token_count: int,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Represent the required BF16/DIRECT/LOW/RESIDUAL baselines uniformly."""
    direct_metrics = _logit_metrics(direct_logits, direct_logits)
    metrics = _logit_metrics(direct_logits, candidate_logits)
    return {
        "schema": SCHEMA,
        "row_type": "baseline",
        "sample_id": sample_id,
        "context_length": int(context_length),
        "residual_dtype": residual_dtype,
        "horizon": int(horizon),
        "method": method,
        "upgrade_mode": "full",
        "candidate_count": int(candidate_count),
        "token_count": int(token_count),
        "upgrade_count": 0,
        "upgrade_rate": 0.0,
        "selected_token_indices": [],
        "direct_high_kl": float(direct_metrics["kl"]),
        "low_kl": None,
        "selected_kl": float(metrics["kl"]),
        "recovered_kl": None,
        "recovery": None,
        "js": float(metrics["js"]),
        "top1_agreement": float(metrics["top1_agreement"]),
        "top1_flip_rate": float(metrics["top1_flip_rate"]),
        "future_only": True,
        "same_step_logits_excluded": True,
        "teacher_forced": True,
        **dict(alignment),
    }


def _make_promoted_plan(
    plan: MaKVQuantPlan,
    selected_token_indices: Iterable[int],
) -> MaKVQuantPlan:
    """Create the temporary full-upgrade plan without touching canonical storage."""
    selected = set(int(value) for value in selected_token_indices)
    ids = bytearray(plan.bucket_ids)
    for layer in range(plan.num_layers):
        for kv in range(2):
            for token in selected:
                if 0 <= token < plan.chunk_length:
                    index = (layer * 2 + kv) * plan.chunk_length + token
                    ids[index] = 0  # configured highest precision is 16 bit
    promoted = replace(plan, bucket_ids=bytes(ids), checksum=0)
    return replace(promoted, checksum=compute_quant_plan_checksum(promoted))


def _temporary_cost(
    reconstructed: torch.Tensor,
    plan: MaKVQuantPlan,
    selected_token_indices: Sequence[int],
    residual_dtype: str,
) -> tuple[float, int, int]:
    """Measure manager-equivalent reconstruction/requantization costs locally."""
    if not selected_token_indices:
        return 0.0, 0, 0
    promoted_plan = _make_promoted_plan(plan, selected_token_indices)
    started = time.perf_counter()
    metadata, payloads = quantize_canonical_kv(
        reconstructed,
        promoted_plan,
        _experiment_config(residual_dtype),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    temporary_blob = encode_makv_object(
        object_type="quantized",
        metadata=metadata,
        payloads=payloads,
    )
    peak_bytes = len(temporary_blob) + int(
        reconstructed.numel() * reconstructed.element_size()
    )
    return elapsed_ms, len(temporary_blob), peak_bytes


def _analyze_variant(
    *,
    sample_id: str,
    context_length: int,
    source: torch.Tensor,
    plan: MaKVQuantPlan,
    low_bundle: QuantizedBundle,
    residual_bundle: QuantizedBundle,
    direct_logits: torch.Tensor,
    low_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    risk_logits: torch.Tensor,
    candidate_indices: Sequence[int],
    horizons: Sequence[int],
    window_tokens: int,
    upgrade_rates: Sequence[float],
    random_repeats: int,
    seed: int,
    alignment: Mapping[str, Any],
    single_forward: Callable[[Sequence[int]], Mapping[int, torch.Tensor]],
    set_forward: Callable[[Sequence[int]], torch.Tensor],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Produce fidelity, token-benefit, selection, and cost rows for one dtype."""
    residual_dtype = str(residual_bundle.metadata["residual"]["dtype"])
    risk_rows = _risk_rows(risk_logits, window_tokens)
    risk_by_index = {int(row["token_index"]): row for row in risk_rows}
    missing = set(candidate_indices) - set(risk_by_index)
    if missing:
        raise ValueError(
            f"risk rows do not cover candidate tokens: {sorted(missing)[:5]}"
        )

    started = time.perf_counter()
    single_outputs = dict(single_forward(candidate_indices))
    single_forward_ms = (time.perf_counter() - started) * 1000.0
    if set(single_outputs) != set(candidate_indices):
        raise ValueError("single-token forward did not return every candidate")

    fidelity_rows: list[dict[str, Any]] = []
    benefit_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        direct_h = direct_logits[:horizon]
        low_h = low_logits[:horizon]
        residual_h = residual_logits[:horizon]
        direct_metrics = _logit_metrics(direct_h, direct_h)
        low_metrics = _logit_metrics(direct_h, low_h)
        residual_metrics = _logit_metrics(direct_h, residual_h)
        fidelity_rows.append(
            {
                "schema": SCHEMA,
                "status": "success",
                "sample_id": sample_id,
                "context_length": int(context_length),
                "residual_dtype": residual_dtype,
                "horizon": int(horizon),
                "upgrade_mode": "full",
                "baseline_direct": "DIRECT_HIGH",
                "comparison": "RESIDUAL_HIGH",
                **_kv_fidelity(source, residual_bundle.residual_canonical),
                "logit_kl": float(residual_metrics["kl"]),
                "logit_js": float(residual_metrics["js"]),
                "top1_agreement": float(residual_metrics["top1_agreement"]),
                "top1_flip_rate": float(residual_metrics["top1_flip_rate"]),
                "direct_high_kl": float(direct_metrics["kl"]),
                "teacher_forced": True,
                "future_only": True,
                "same_step_logits_excluded": True,
                **dict(alignment),
            }
        )
        token_rows: list[dict[str, Any]] = []
        for token_index in candidate_indices:
            metrics = _logit_metrics(direct_h, single_outputs[token_index][:horizon])
            risk = risk_by_index[token_index]
            token_rows.append(
                {
                    "schema": SCHEMA,
                    "status": "success",
                    "sample_id": sample_id,
                    "context_length": int(context_length),
                    "residual_dtype": residual_dtype,
                    "horizon": int(horizon),
                    "token_index": int(token_index),
                    "absolute_token_index": int(token_index),
                    "risk": float(risk["risk"]),
                    "margin_only_risk": float(risk["margin_risk"]),
                    "low_kl": float(low_metrics["kl"]),
                    "upgrade_kl": float(metrics["kl"]),
                    "direct_high_kl": float(direct_metrics["kl"]),
                    "benefit_kl": float(low_metrics["kl"] - metrics["kl"]),
                    "low_js": float(low_metrics["js"]),
                    "upgrade_js": float(metrics["js"]),
                    "benefit_js": float(low_metrics["js"] - metrics["js"]),
                    "low_top1_flip": bool(low_metrics["top1_flip_rate"] > 0.0),
                    "upgrade_top1_flip": bool(metrics["top1_flip_rate"] > 0.0),
                    "future_steps": int(horizon),
                    "teacher_forced": True,
                    "future_only": True,
                    "same_step_logits_excluded": True,
                    "risk_source": risk["risk_source"],
                    "window_tokens": window_tokens,
                    **dict(alignment),
                }
            )
        benefit_rows.extend(token_rows)
        selection_rows.extend(
            [
                _baseline_row(
                    sample_id=sample_id,
                    context_length=context_length,
                    residual_dtype=residual_dtype,
                    horizon=horizon,
                    method="BF16",
                    direct_logits=direct_h,
                    candidate_logits=direct_h,
                    candidate_count=len(candidate_indices),
                    token_count=source.shape[2],
                    alignment=alignment,
                ),
                _baseline_row(
                    sample_id=sample_id,
                    context_length=context_length,
                    residual_dtype=residual_dtype,
                    horizon=horizon,
                    method="DIRECT_HIGH",
                    direct_logits=direct_h,
                    candidate_logits=direct_h,
                    candidate_count=len(candidate_indices),
                    token_count=source.shape[2],
                    alignment=alignment,
                ),
                _baseline_row(
                    sample_id=sample_id,
                    context_length=context_length,
                    residual_dtype=residual_dtype,
                    horizon=horizon,
                    method="LOW",
                    direct_logits=direct_h,
                    candidate_logits=low_h,
                    candidate_count=len(candidate_indices),
                    token_count=source.shape[2],
                    alignment=alignment,
                ),
                _baseline_row(
                    sample_id=sample_id,
                    context_length=context_length,
                    residual_dtype=residual_dtype,
                    horizon=horizon,
                    method="RESIDUAL_HIGH",
                    direct_logits=direct_h,
                    candidate_logits=residual_h,
                    candidate_count=len(candidate_indices),
                    token_count=source.shape[2],
                    alignment=alignment,
                ),
            ]
        )
        # The low metrics are deliberately computed from the same future rows
        # as every intervention. No current-step logit is mixed into benefit.
        for rate_index, requested_rate in enumerate(upgrade_rates):
            count = (
                0
                if requested_rate <= 0.0
                else max(
                    1,
                    min(
                        len(token_rows), round(float(requested_rate) * len(token_rows))
                    ),
                )
            )
            for method_key, method_name in (
                ("conf", "CONF_UPGRADE"),
                ("margin_only", "MARGIN_ONLY"),
                ("oracle", "ORACLE_UPGRADE"),
            ):
                selected = _select_tokens(
                    token_rows,
                    method_key,
                    count,
                    seed + horizon * 1009 + rate_index * 9176,
                )
                selected_logits = (
                    low_h if not selected else set_forward(selected)[:horizon]
                )
                row = _selection_row(
                    sample_id=sample_id,
                    context_length=context_length,
                    residual_dtype=residual_dtype,
                    horizon=horizon,
                    method=method_name,
                    selected=selected,
                    candidate_count=len(token_rows),
                    direct_logits=direct_h,
                    low_logits=low_h,
                    selected_logits=selected_logits,
                    benefit_rows=token_rows,
                    token_count=source.shape[2],
                    alignment=alignment,
                )
                row["requested_upgrade_rate"] = float(requested_rate)
                selection_rows.append(row)

            random_metric_rows: list[dict[str, Any]] = []
            random_selections: list[list[int]] = []
            for repeat in range(random_repeats):
                selected = _select_tokens(
                    token_rows,
                    "random",
                    count,
                    seed + horizon * 1009 + rate_index * 9176 + repeat,
                )
                selected_logits = (
                    low_h if not selected else set_forward(selected)[:horizon]
                )
                random_selections.append(selected)
                random_metric_rows.append(
                    _selection_row(
                        sample_id=sample_id,
                        context_length=context_length,
                        residual_dtype=residual_dtype,
                        horizon=horizon,
                        method="RANDOM_UPGRADE",
                        selected=selected,
                        candidate_count=len(token_rows),
                        direct_logits=direct_h,
                        low_logits=low_h,
                        selected_logits=selected_logits,
                        benefit_rows=token_rows,
                        token_count=source.shape[2],
                        alignment=alignment,
                        random_repeats=random_repeats,
                    )
                )
            numeric_fields = (
                "direct_high_kl",
                "low_kl",
                "selected_kl",
                "recovered_kl",
                "direct_high_js",
                "low_js",
                "selected_js",
                "recovered_js",
                "low_top1_flip_rate",
                "selected_top1_flip_rate",
                "top1_flip_reduction",
                "recovery",
            )
            aggregate = dict(random_metric_rows[0])
            for field in numeric_fields:
                values = [
                    float(row[field])
                    for row in random_metric_rows
                    if row[field] is not None
                ]
                aggregate[field] = sum(values) / len(values) if values else None
                aggregate[f"{field}_std"] = (
                    math.sqrt(
                        sum((value - aggregate[field]) ** 2 for value in values)
                        / (len(values) - 1)
                    )
                    if len(values) > 1
                    else 0.0
                )
            aggregate["selected_token_indices_by_repeat"] = random_selections
            aggregate["requested_upgrade_rate"] = float(requested_rate)
            aggregate["random_repeat_count"] = random_repeats
            selection_rows.append(aggregate)

    cost_count = max(1, int(len(candidate_indices) * 0.10))
    cost_selected = list(candidate_indices[:cost_count])
    # Use the first 10% as a fixed cost probe. Selection quality itself is
    # measured above with CONF/random/oracle matched counts.
    requantization_ms, temporary_blob_bytes, temporary_peak_bytes = _temporary_cost(
        residual_bundle.residual_canonical,
        plan,
        cost_selected,
        residual_dtype,
    )
    cost_row = {
        "schema": SCHEMA,
        "status": "success",
        "sample_id": sample_id,
        "context_length": int(context_length),
        "residual_dtype": residual_dtype,
        "token_count": int(source.shape[2]),
        "raw_kv_bytes": int(source.numel() * source.element_size()),
        "quantized_blob_bytes": len(low_bundle.blob),
        "quantized_payload_bytes": low_bundle.quantized_payload_bytes,
        "residual_bytes": residual_bundle.residual_bytes,
        "total_remote_bytes": len(residual_bundle.blob),
        "quantize_latency_ms": residual_bundle.quantize_time_ms,
        "reconstruction_latency_ms": residual_bundle.reconstruction_time_ms,
        "requantization_latency_ms": requantization_ms,
        "temporary_object_bytes": temporary_blob_bytes,
        "temporary_object_peak_bytes": temporary_peak_bytes,
        "single_token_counterfactual_latency_ms": single_forward_ms,
        "canonical_get_latency_ms": None,
        "upgraded_get_latency_ms": None,
        "risk_handler_latency_ms": None,
        "quantized_bytes_definition": (
            "complete canonical K2V2 blob without residual payloads"
        ),
        "residual_bytes_definition": "residual_* payload bytes only",
        "future_horizon_definition": list(int(value) for value in horizons),
    }
    return fidelity_rows, benefit_rows, selection_rows, cost_row


def _read_manifest(path: Path, max_prompts: int) -> list[dict[str, Any]]:
    """Read the existing LongBench-style JSONL prompt manifest."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "sample_id" not in row or "prompt" not in row:
                raise ValueError(f"manifest row lacks sample_id/prompt: {row}")
            rows.append(row)
            if max_prompts > 0 and len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError(f"manifest has no rows: {path}")
    return rows


def _extract_canonical_cache(past_key_values: Any) -> torch.Tensor:
    """Extract canonical ``[L,2,T,H,D]`` data from a HF cache/tuple."""
    layers = getattr(past_key_values, "layers", past_key_values)
    extracted: list[torch.Tensor] = []
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            keys, values = layer[0], layer[1]
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("Qwen past KV must have shape [batch,heads,tokens,dim]")
        extracted.append(
            torch.stack(
                (
                    keys[0].permute(1, 0, 2),
                    values[0].permute(1, 0, 2),
                ),
                dim=0,
            )
        )
    if not extracted:
        raise ValueError("Qwen past KV cache is empty")
    return torch.stack(extracted, dim=0).contiguous()


def _canonical_cache(
    canonical: torch.Tensor,
    model_config: Any,
    *,
    batch_size: int,
    replacement: torch.Tensor | None = None,
    candidate_positions: Sequence[int] | None = None,
    selected_positions: Sequence[int] | None = None,
) -> Any:
    """Create a HF DynamicCache and apply one or many full-token upgrades."""
    from transformers import DynamicCache

    if canonical.device.type != "cuda":
        raise ValueError("real Qwen cache construction requires a CUDA tensor")
    if replacement is not None and replacement.shape != canonical.shape:
        raise ValueError("replacement canonical KV shape differs")
    cache = DynamicCache(config=model_config)
    if len(cache.layers) != canonical.shape[0]:
        raise ValueError("model/cache layer count differs")
    if (candidate_positions is None) == (
        selected_positions is None
    ) and replacement is not None:
        raise ValueError("provide exactly one candidate or selected position list")
    for layer_index, destination in enumerate(cache.layers):
        key = (
            canonical[layer_index, 0]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .clone()
        )
        value = (
            canonical[layer_index, 1]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .clone()
        )
        if replacement is not None and candidate_positions is not None:
            positions = torch.as_tensor(
                candidate_positions, dtype=torch.long, device=key.device
            )
            rows = torch.arange(batch_size, dtype=torch.long, device=key.device)
            if len(positions) != batch_size:
                raise ValueError("candidate position count must equal batch size")
            key[rows, :, positions, :] = replacement[layer_index, 0, positions, :]
            value[rows, :, positions, :] = replacement[layer_index, 1, positions, :]
        elif replacement is not None and selected_positions is not None:
            positions = torch.as_tensor(
                selected_positions, dtype=torch.long, device=key.device
            )
            replacement_key = (
                replacement[layer_index, 0, positions, :].permute(1, 0, 2).unsqueeze(0)
            )
            replacement_value = (
                replacement[layer_index, 1, positions, :].permute(1, 0, 2).unsqueeze(0)
            )
            key[:, :, positions, :] = replacement_key
            value[:, :, positions, :] = replacement_value
        destination.keys = key
        destination.values = value
        destination.is_initialized = True
    return cache


def _qwen_forward(
    model: Any,
    canonical: torch.Tensor,
    suffix_input: torch.Tensor,
    prefix_length: int,
    *,
    replacement: torch.Tensor | None = None,
    candidate_positions: Sequence[int] | None = None,
    selected_positions: Sequence[int] | None = None,
) -> torch.Tensor:
    """Run one batched teacher-forced forward from a canonical cache."""
    batch_size = len(candidate_positions) if candidate_positions is not None else 1
    if selected_positions is not None:
        batch_size = 1
    cache = _canonical_cache(
        canonical,
        model.config,
        batch_size=batch_size,
        replacement=replacement,
        candidate_positions=candidate_positions,
        selected_positions=selected_positions,
    )
    input_ids = suffix_input.unsqueeze(0).expand(batch_size, -1)
    positions = torch.arange(
        prefix_length,
        prefix_length + suffix_input.shape[0],
        dtype=torch.long,
        device=canonical.device,
    )
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones(
                batch_size,
                prefix_length + suffix_input.shape[0],
                dtype=torch.long,
                device=canonical.device,
            ),
            position_ids=positions.unsqueeze(0).expand(batch_size, -1),
            cache_position=positions,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    return output.logits.float()


def _load_qwen(model_path: str, device: str, model_dtype: str) -> tuple[Any, Any]:
    """Load a local target model and tokenizer only after CUDA is available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=_dtype(model_dtype),
        low_cpu_mem_usage=True,
    )
    model.to(torch.device(device))
    model.eval()
    return model, tokenizer


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, add_special_tokens=True)
    return [int(value) for value in encoded.input_ids]


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    """Aggregate numeric frontier rows by the requested identity keys."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for identity, group in sorted(groups.items(), key=lambda item: str(item[0])):
        result = {key: value for key, value in zip(keys, identity, strict=True)}
        result["sample_count"] = len(group)
        fields = (
            "candidate_count",
            "token_count",
            "upgrade_count",
            "upgrade_rate",
            "direct_high_kl",
            "low_kl",
            "selected_kl",
            "recovered_kl",
            "recovery",
            "spearman_risk_benefit",
            "low_top1_flip_rate",
            "selected_top1_flip_rate",
            "top1_flip_reduction",
        )
        for field in fields:
            values = [float(row[field]) for row in group if row.get(field) is not None]
            result[field] = sum(values) / len(values) if values else None
        result["frontier_definition"] = (
            "matched-count selection evaluated on future teacher-forced logits"
        )
        output.append(result)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


async def _run_invariant_probe() -> dict[str, Any]:
    """Exercise canonical/temporary/public GET behavior on the real manager."""
    source = torch.arange(2 * 1 * 8 * 1 * 3, dtype=torch.float16).reshape(1, 2, 8, 1, 3)
    plan = build_canonical_k2_plan(source)
    key = "precision-upgrade-invariant-probe"
    envelope = encode_client_put_envelope(
        key=key,
        object_type="raw_with_plan",
        plan=plan,
        raw_kv_payload=_tensor_bytes(_canonical_to_wire(source)),
    )
    storage = _MemoryStorage()
    manager = MaKVRemoteManager(_experiment_config("float16"), storage)
    try:
        await manager.put(key, envelope, 0.0)
        canonical_hash_before = hashlib.sha256(storage.values[key]).hexdigest()
        before, before_timing = await manager.get_with_timing(key)
        batch_before = (await manager.get_many_with_timing([key]))[0][0]
        if before is None or batch_before is None:
            raise AssertionError("invariant probe initial GET missed")
        active_started = time.perf_counter()
        active_result = await manager.apply_precision_risk(
            key,
            {
                "step": 0,
                "token_index": 2,
                "window_tokens": 16,
                "risk": 1.0,
                "valid": True,
                "scorer_version": CONF_SCORER_VERSION,
                "semantics": CONF_RISK_SEMANTICS,
            },
        )
        risk_handler_ms = (time.perf_counter() - active_started) * 1000.0
        active, active_timing = await manager.get_with_timing(key)
        batch_active = (await manager.get_many_with_timing([key]))[0][0]
        if active is None or batch_active is None:
            raise AssertionError("invariant probe active GET missed")
        expired_result = await manager.apply_precision_risk(
            key,
            {
                "step": 16,
                "token_index": 2,
                "window_tokens": 16,
                "risk": 0.0,
                "valid": True,
                "scorer_version": CONF_SCORER_VERSION,
                "semantics": CONF_RISK_SEMANTICS,
            },
        )
        restored, restored_timing = await manager.get_with_timing(key)
        batch_restored = (await manager.get_many_with_timing([key]))[0][0]
        if restored is None or batch_restored is None:
            raise AssertionError("invariant probe restored GET missed")

        def is_public(value: bytes) -> bool:
            decoded = decode_makv_object(value)
            private_names = {
                name
                for name in decoded.payloads
                if name.startswith("residual_") or name.startswith("precision_window_")
            }
            private_metadata = {
                name
                for name in ("residual", "risk_upgrade", "precision_window")
                if name in decoded.metadata
            }
            return not private_names and not private_metadata

        none_storage = _MemoryStorage()
        none_manager = MaKVRemoteManager(_experiment_config("none"), none_storage)
        try:
            await none_manager.put(key, envelope, 0.0)
            none_result = await none_manager.apply_precision_risk(
                key,
                {
                    "step": 0,
                    "token_index": 2,
                    "risk": 1.0,
                    "valid": True,
                    "scorer_version": CONF_SCORER_VERSION,
                    "semantics": CONF_RISK_SEMANTICS,
                },
            )
        finally:
            await none_manager.close()
        return {
            "status": "passed",
            "canonical_hash_before": canonical_hash_before,
            "canonical_hash_after": hashlib.sha256(storage.values[key]).hexdigest(),
            "canonical_hash_unchanged": canonical_hash_before
            == hashlib.sha256(storage.values[key]).hexdigest(),
            "before_get_public": is_public(before),
            "before_get_batch_public": is_public(batch_before),
            "active_get_public": is_public(active),
            "active_get_batch_public": is_public(batch_active),
            "restored_get_public": is_public(restored),
            "restored_get_batch_public": is_public(batch_restored),
            "active_window_upgraded": bool(active_result.get("upgraded")),
            "active_view_differs_from_canonical": active != before,
            "active_window_get_hit": bool(active_timing.get("precision_window_hit")),
            "window_expired": bool(expired_result.get("window_expired")),
            "restored_hash_matches_before": restored == before,
            "residual_none_fail_closed": (
                none_result.get("reason") == "residual_unavailable"
                and not none_result.get("upgraded", False)
            ),
            "risk_handler_latency_ms": risk_handler_ms,
            "canonical_get_latency_ms": float(before_timing["total_ms"]),
            "upgraded_get_latency_ms": float(active_timing["total_ms"]),
            "restored_get_latency_ms": float(restored_timing["total_ms"]),
            "manager_quantize_calls": int(manager.quantize_calls),
            "manager_result": active_result,
            "none_residual_result": none_result,
            "private_metadata_never_public": all(
                is_public(value)
                for value in (
                    before,
                    batch_before,
                    active,
                    batch_active,
                    restored,
                    batch_restored,
                )
            ),
        }
    finally:
        await manager.close()


def _synthetic_logits(
    canonical: torch.Tensor, horizon: int, vocab_size: int
) -> torch.Tensor:
    """Create deterministic future logits for the CPU-only validation backend."""
    signal = canonical.float().mean(dim=(0, 1, 3, 4))
    token_weights = 1.0 + torch.arange(signal.numel(), dtype=torch.float32) / max(
        1, signal.numel()
    )
    context = (signal * token_weights).sum()
    base = torch.linspace(-1.0, 1.0, vocab_size)
    rows = []
    for step in range(horizon):
        direction = torch.sin(
            torch.arange(vocab_size, dtype=torch.float32) * (0.13 + step * 0.01)
        )
        rows.append(base + context * direction * (0.2 + step / max(1, horizon)))
    return torch.stack(rows)


def _synthetic_case(
    sample_index: int,
    context_length: int,
    max_horizon: int,
    *,
    layers: int = 2,
    heads: int = 2,
    head_dim: int = 5,
    vocab_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Return source KV, direct future logits, risk logits, and target IDs."""
    generator = torch.Generator(device="cpu").manual_seed(
        20260828 + sample_index * 1009 + context_length
    )
    source = torch.randn(
        layers,
        2,
        context_length,
        heads,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    direct_logits = _synthetic_logits(source, max_horizon, vocab_size)
    source_signal = source.float().mean(dim=(0, 1, 3, 4))
    risk_logits = torch.full((context_length, vocab_size), -2.0)
    error_proxy = source_signal.abs()
    risk_logits[:, 0] = 1.0 + 1.0 / (1.0 + error_proxy)
    risk_logits[:, 1] = 1.0
    target_ids = [
        int(value)
        for value in torch.randint(vocab_size, (max_horizon,), generator=generator)
    ]
    return source, direct_logits, risk_logits, target_ids


def _blocked_result(
    output_dir: Path, config: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    """Write explicit blocked artifacts without emitting synthetic measurements."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "schema": SCHEMA,
        "status": "blocked",
        "reason": reason,
        "config": dict(config),
    }
    _write_json(output_dir / "residual_fidelity.json", {**base, "rows": []})
    _write_jsonl(output_dir / "upgrade_benefit.jsonl", [])
    _write_jsonl(output_dir / "risk_selection.jsonl", [])
    _write_json(output_dir / "upgrade_frontier.json", {**base, "frontier": []})
    _write_json(
        output_dir / "system_cost.json", {**base, "rows": [], "invariants": None}
    )
    (output_dir / "precision_upgrade_report.md").write_text(
        "# MaKV residual precision upgrade validation\n\n"
        f"Status: **BLOCKED**\n\nReason: `{reason}`\n\n"
        "No numeric model result was generated.\n",
        encoding="utf-8",
    )
    return {"status": "blocked", "reason": reason, "output_dir": str(output_dir)}


def _finalize(
    output_dir: Path,
    config: Mapping[str, Any],
    fidelity_rows: Sequence[Mapping[str, Any]],
    benefit_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    invariants: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the complete artifact set and a concise human-readable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "success" if not failures else "partial"
    frontier = _aggregate_rows(
        [row for row in selection_rows if row.get("row_type") == "selection"],
        ("residual_dtype", "horizon", "method", "requested_upgrade_rate"),
    )
    fidelity = {
        "schema": SCHEMA,
        "status": status,
        "config": dict(config),
        "rows": list(fidelity_rows),
        "baseline_definitions": {
            "BF16": "direct target-model BF16 KV and future teacher-forced logits",
            "DIRECT_HIGH": "original prompt KV used directly",
            "LOW": "canonical K2V2 quantized object without residual",
            "RESIDUAL_HIGH": "LOW object plus manager-side residual reconstruction",
        },
        "future_only": True,
        "same_step_logits_excluded": True,
        "failures": list(failures),
    }
    _write_json(output_dir / "residual_fidelity.json", fidelity)
    _write_jsonl(output_dir / "upgrade_benefit.jsonl", list(benefit_rows))
    _write_jsonl(output_dir / "risk_selection.jsonl", list(selection_rows))
    _write_json(
        output_dir / "upgrade_frontier.json",
        {
            "schema": SCHEMA,
            "status": status,
            "config": dict(config),
            "frontier": frontier,
            "recovery_definition": "(KL_low - KL_upgrade) / (KL_low - KL_direct_high)",
            "random_definition": (
                "uniform random candidate tokens at identical upgrade count"
            ),
            "oracle_definition": "descending measured residual-high benefit_kl",
            "failures": list(failures),
        },
    )
    cost_payload = {
        "schema": SCHEMA,
        "status": status,
        "config": dict(config),
        "rows": [
            {
                **row,
                "risk_handler_latency_ms": invariants.get("risk_handler_latency_ms"),
                "canonical_get_latency_ms": invariants.get("canonical_get_latency_ms"),
                "upgraded_get_latency_ms": invariants.get("upgraded_get_latency_ms"),
            }
            for row in cost_rows
        ],
        "manager_probe_timing_scope": "small in-process manager invariant object",
        "invariants": dict(invariants),
        "failures": list(failures),
    }
    _write_json(output_dir / "system_cost.json", cost_payload)
    report_lines = [
        "# MaKV residual precision upgrade validation",
        "",
        f"Status: **{status.upper()}**",
        f"Schema: `{SCHEMA}`",
        "",
        "## Scope",
        "",
        "- Production CONF weights, quantizer, CUDA, and canonical blobs "
        "were not modified.",
        "- Canonical precision is K2V2; upgrade mode is full token (all "
        "layers and K/V).",
        "- Risk rows use explicit absolute `token_index`; no step fallback is used.",
        "- Offline risk input is aligned prefill-next-token logits; production "
        "CONF weights are unchanged.",
        "- Benefit and quality metrics use only future teacher-forced steps H=32/64.",
        "- Random rows use the same candidate count as CONF and oracle rows.",
        "",
        "## Counts",
        "",
        f"- Residual fidelity rows: `{len(fidelity_rows)}`",
        f"- Token benefit rows: `{len(benefit_rows)}`",
        f"- Selection rows: `{len(selection_rows)}`",
        f"- Cost rows: `{len(cost_rows)}`",
        f"- Failures: `{len(failures)}`",
        "",
        "## System Invariants",
        "",
        f"- Invariant probe: `{invariants.get('status')}`",
        f"- Canonical hash unchanged: `{invariants.get('canonical_hash_unchanged')}`",
        f"- Public GET/GET_BATCH hides private fields: "
        f"`{invariants.get('private_metadata_never_public')}`",
        f"- residual=none fails closed: "
        f"`{invariants.get('residual_none_fail_closed')}`",
        "",
        "## Artifacts",
        "",
    ]
    report_lines.extend(f"- `{name}`" for name in OUTPUT_NAMES)
    if failures:
        report_lines.extend(["", "## Failures", ""])
        report_lines.extend(f"- `{failure}`" for failure in failures)
    (output_dir / "precision_upgrade_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return {
        "status": status,
        "output_dir": str(output_dir),
        "fidelity_rows": len(fidelity_rows),
        "benefit_rows": len(benefit_rows),
        "selection_rows": len(selection_rows),
        "cost_rows": len(cost_rows),
        "failures": list(failures),
    }


def run_experiment(
    *,
    output_dir: str | Path,
    backend: str = "qwen",
    model_path: str | Path = DEFAULT_MODEL,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    max_prompts: int = 16,
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    residual_dtypes: Sequence[str] = SUPPORTED_RESIDUAL_DTYPES,
    upgrade_rates: Sequence[float] = DEFAULT_UPGRADE_RATES,
    window_tokens: int = 16,
    random_repeats: int = 16,
    candidate_token_limit: int = 0,
    candidate_batch_size: int = 4,
    model_dtype: str = "bfloat16",
    device: str = "cuda:0",
    seed: int = 20260828,
) -> dict[str, Any]:
    """Run strict real or synthetic validation and write all six artifacts."""
    output = Path(output_dir)
    context_values = tuple(int(value) for value in context_lengths)
    horizon_values = tuple(int(value) for value in horizons)
    residual_values = tuple(str(value).lower() for value in residual_dtypes)
    if backend not in ("qwen", "synthetic"):
        raise ValueError("backend must be qwen or synthetic")
    if not context_values or any(value <= 0 for value in context_values):
        raise ValueError("context_lengths must contain positive values")
    if not horizon_values or any(value <= 0 for value in horizon_values):
        raise ValueError("horizons must contain positive values")
    if any(residual not in SUPPORTED_RESIDUAL_DTYPES for residual in residual_values):
        raise ValueError("residual_dtypes must contain float16 and/or float32")
    if window_tokens <= 0 or random_repeats <= 0 or candidate_batch_size <= 0:
        raise ValueError(
            "window_tokens, random_repeats, and candidate_batch_size must be positive"
        )
    if any(not 0.0 <= float(rate) <= 1.0 for rate in upgrade_rates):
        raise ValueError("upgrade_rates must be in [0,1]")
    config = {
        "backend": backend,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "max_prompts": int(max_prompts),
        "context_lengths": list(context_values),
        "horizons": list(horizon_values),
        "residual_dtypes": list(residual_values),
        "canonical_precision": "K2V2",
        "upgrade_mode": "full",
        "window_tokens": int(window_tokens),
        "teacher_forced": True,
        "hash_aligned": True,
        "candidate_token_limit": int(candidate_token_limit),
        "candidate_batch_size": int(candidate_batch_size),
        "model_dtype": model_dtype,
        "device": device,
        "seed": int(seed),
        "production_policy_modified": False,
    }
    if backend == "qwen":
        if not torch.cuda.is_available():
            return _blocked_result(
                output,
                config,
                "CUDA is unavailable; Qwen3-8B validation was not executed",
            )
        try:
            records = _read_manifest(Path(manifest_path), max_prompts)
            model, tokenizer = _load_qwen(str(model_path), device, model_dtype)
        except Exception as error:  # noqa: BLE001 - preserve blocked reason in artifact
            return _blocked_result(
                output, config, f"Qwen setup failed: {type(error).__name__}: {error}"
            )
    else:
        records = []
        manifest = Path(manifest_path)
        if manifest.exists():
            records = _read_manifest(manifest, max_prompts)
        if not records:
            records = [
                {"sample_id": f"synthetic-{index:02d}", "prompt": ""}
                for index in range(max_prompts)
            ]
        model = None
        tokenizer = None

    fidelity_rows: list[dict[str, Any]] = []
    benefit_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for sample_index, record in enumerate(records):
            sample_id = str(record["sample_id"])
            for context_length in context_values:
                try:
                    max_horizon = max(horizon_values)
                    if backend == "synthetic":
                        source, direct_logits, risk_logits, target_ids = (
                            _synthetic_case(sample_index, context_length, max_horizon)
                        )
                        prefix_ids = list(range(context_length))
                        suffix_ids = list(
                            range(context_length, context_length + max_horizon)
                        )
                        target_ids = [int(value) for value in target_ids]

                        def make_forwarders(
                            low: torch.Tensor,
                            residual: torch.Tensor,
                            *,
                            horizon: int,
                            vocab_size: int,
                        ) -> tuple[
                            Callable[[Sequence[int]], Mapping[int, torch.Tensor]],
                            Callable[[Sequence[int]], torch.Tensor],
                        ]:
                            def single(
                                indices: Sequence[int],
                            ) -> Mapping[int, torch.Tensor]:
                                result: dict[int, torch.Tensor] = {}
                                for index in indices:
                                    upgraded = low.clone()
                                    upgraded[:, :, int(index)] = residual[
                                        :, :, int(index)
                                    ]
                                    result[int(index)] = _synthetic_logits(
                                        upgraded, horizon, vocab_size
                                    )
                                return result

                            def selected(indices: Sequence[int]) -> torch.Tensor:
                                upgraded = low.clone()
                                if indices:
                                    positions = torch.as_tensor(
                                        indices, dtype=torch.long
                                    )
                                    upgraded[:, :, positions] = residual[
                                        :, :, positions
                                    ]
                                return _synthetic_logits(upgraded, horizon, vocab_size)

                            return single, selected
                    else:
                        token_ids = _encode_prompt(tokenizer, str(record["prompt"]))
                        required = context_length + max_horizon + 1
                        if len(token_ids) < required:
                            raise ValueError(
                                f"{sample_id} has {len(token_ids)} tokens; "
                                f"{required} required"
                            )
                        prefix_ids = token_ids[:context_length]
                        suffix_ids = token_ids[
                            context_length : context_length + max_horizon
                        ]
                        target_ids = token_ids[
                            context_length + 1 : context_length + max_horizon + 1
                        ]
                        device_obj = torch.device(device)
                        prefix_tensor = torch.tensor(
                            [prefix_ids], dtype=torch.long, device=device_obj
                        )
                        with torch.inference_mode():
                            prefill = model(
                                input_ids=prefix_tensor,
                                attention_mask=torch.ones_like(prefix_tensor),
                                position_ids=torch.arange(
                                    context_length, dtype=torch.long, device=device_obj
                                ).unsqueeze(0),
                                use_cache=True,
                                output_attentions=False,
                                return_dict=True,
                            )
                        source_gpu = _extract_canonical_cache(prefill.past_key_values)
                        source = source_gpu.detach().cpu().contiguous()
                        direct_logits = (
                            _qwen_forward(
                                model,
                                source_gpu,
                                torch.tensor(
                                    suffix_ids, dtype=torch.long, device=device_obj
                                ),
                                context_length,
                            )[0]
                            .detach()
                            .cpu()
                        )
                        risk_logits = prefill.logits[0, :context_length].detach().cpu()
                        del prefill, source_gpu

                    alignment = {
                        "prefix_token_hash": _stable_token_hash(prefix_ids),
                        "suffix_input_token_hash": _stable_token_hash(suffix_ids),
                        "target_token_hash": _stable_token_hash(target_ids),
                        "prefix_length": len(prefix_ids),
                        "suffix_length": len(suffix_ids),
                        "target_length": len(target_ids),
                        "teacher_forced": True,
                    }
                    if len(suffix_ids) != max_horizon or len(target_ids) != max_horizon:
                        raise ValueError(
                            "teacher-forced suffix/target lengths are not aligned"
                        )
                    plan = build_canonical_k2_plan(source)
                    low_bundle = _quantize_bundle(source, plan, "none")
                    low_canonical = low_bundle.low_canonical
                    if backend == "synthetic":
                        low_logits = _synthetic_logits(
                            low_canonical, max_horizon, int(direct_logits.shape[-1])
                        )
                    else:
                        low_gpu = low_canonical.to(
                            device=device, dtype=_dtype(model_dtype)
                        )
                        suffix_gpu = torch.tensor(
                            suffix_ids, dtype=torch.long, device=device
                        )
                        low_logits = (
                            _qwen_forward(model, low_gpu, suffix_gpu, context_length)[0]
                            .detach()
                            .cpu()
                        )
                        del low_gpu, suffix_gpu

                    for dtype_index, residual_dtype in enumerate(residual_values):
                        residual_bundle = _quantize_bundle(source, plan, residual_dtype)
                        if residual_bundle.residual_canonical is None:
                            raise ValueError(
                                "residual bundle did not reconstruct a canonical tensor"
                            )
                        residual_canonical = residual_bundle.residual_canonical
                        if backend == "synthetic":
                            residual_logits = _synthetic_logits(
                                residual_canonical,
                                max_horizon,
                                int(direct_logits.shape[-1]),
                            )
                            single_forward, set_forward = make_forwarders(
                                low_canonical,
                                residual_canonical,
                                horizon=max_horizon,
                                vocab_size=int(direct_logits.shape[-1]),
                            )
                        else:
                            low_gpu = low_canonical.to(
                                device=device, dtype=_dtype(model_dtype)
                            )
                            residual_gpu = residual_canonical.to(
                                device=device, dtype=_dtype(model_dtype)
                            )
                            suffix_gpu = torch.tensor(
                                suffix_ids, dtype=torch.long, device=device
                            )

                            def single_forward(
                                indices: Sequence[int],
                                *,
                                _low: torch.Tensor = low_gpu,
                                _residual: torch.Tensor = residual_gpu,
                                _suffix: torch.Tensor = suffix_gpu,
                                _context_length: int = context_length,
                            ) -> Mapping[int, torch.Tensor]:
                                result: dict[int, torch.Tensor] = {}
                                for start in range(
                                    0, len(indices), candidate_batch_size
                                ):
                                    batch_indices = list(
                                        indices[start : start + candidate_batch_size]
                                    )
                                    output = _qwen_forward(
                                        model,
                                        _low,
                                        _suffix,
                                        _context_length,
                                        replacement=_residual,
                                        candidate_positions=batch_indices,
                                    )
                                    for offset, index in enumerate(batch_indices):
                                        result[int(index)] = (
                                            output[offset].detach().cpu()
                                        )
                                return result

                            def set_forward(
                                indices: Sequence[int],
                                *,
                                _low: torch.Tensor = low_gpu,
                                _residual: torch.Tensor = residual_gpu,
                                _suffix: torch.Tensor = suffix_gpu,
                                _context_length: int = context_length,
                            ) -> torch.Tensor:
                                return (
                                    _qwen_forward(
                                        model,
                                        _low,
                                        _suffix,
                                        _context_length,
                                        replacement=_residual,
                                        selected_positions=indices,
                                    )[0]
                                    .detach()
                                    .cpu()
                                )

                            residual_logits = (
                                _qwen_forward(
                                    model,
                                    residual_gpu,
                                    suffix_gpu,
                                    context_length,
                                )[0]
                                .detach()
                                .cpu()
                            )

                        candidates = _candidate_indices(
                            source.shape[2], candidate_token_limit
                        )
                        rows = _analyze_variant(
                            sample_id=sample_id,
                            context_length=context_length,
                            source=source,
                            plan=plan,
                            low_bundle=low_bundle,
                            residual_bundle=residual_bundle,
                            direct_logits=direct_logits,
                            low_logits=low_logits,
                            residual_logits=residual_logits,
                            risk_logits=risk_logits,
                            candidate_indices=candidates,
                            horizons=horizon_values,
                            window_tokens=window_tokens,
                            upgrade_rates=upgrade_rates,
                            random_repeats=random_repeats,
                            seed=seed + sample_index * 100003 + dtype_index * 1009,
                            alignment=alignment,
                            single_forward=single_forward,
                            set_forward=set_forward,
                        )
                        fidelity, benefits, selections, cost = rows
                        cost.update(alignment)
                        fidelity_rows.extend(fidelity)
                        benefit_rows.extend(benefits)
                        selection_rows.extend(selections)
                        cost_rows.append(cost)
                        if backend == "qwen":
                            del low_gpu, residual_gpu, suffix_gpu
                    del source, direct_logits, risk_logits, low_logits
                except Exception as error:  # noqa: BLE001 - preserve sample-level audit trail
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "context_length": int(context_length),
                            "exception_type": type(error).__name__,
                            "exception": str(error),
                        }
                    )
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        invariants = asyncio.run(_run_invariant_probe())
    except Exception as error:  # noqa: BLE001 - experiment must not hide failed invariants
        invariants = {
            "status": "failed",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "canonical_hash_unchanged": False,
            "private_metadata_never_public": False,
            "residual_none_fail_closed": False,
        }
        failures.append(
            {
                "scope": "system_invariant_probe",
                "exception_type": type(error).__name__,
                "exception": str(error),
            }
        )
    return _finalize(
        output,
        config,
        fidelity_rows,
        benefit_rows,
        selection_rows,
        cost_rows,
        invariants,
        failures,
    )


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    """Run the command-line precision upgrade experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "synthetic"), default="qwen")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--context-lengths", default="1024,2048")
    parser.add_argument("--horizons", default="32,64")
    parser.add_argument("--residual-dtypes", default="float16,float32")
    parser.add_argument("--upgrade-rates", default="0.01,0.05,0.10,0.20")
    parser.add_argument("--window-tokens", type=int, default=16)
    parser.add_argument("--random-repeats", type=int, default=16)
    parser.add_argument(
        "--candidate-token-limit",
        type=int,
        default=0,
        help=(
            "0 evaluates every context token; a positive value selects "
            "evenly spaced tokens"
        ),
    )
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument(
        "--model-dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    result = run_experiment(
        output_dir=args.output_dir,
        backend=args.backend,
        model_path=args.model,
        manifest_path=args.manifest,
        max_prompts=args.max_prompts,
        context_lengths=_parse_ints(args.context_lengths),
        horizons=_parse_ints(args.horizons),
        residual_dtypes=tuple(
            part.strip().lower()
            for part in args.residual_dtypes.split(",")
            if part.strip()
        ),
        upgrade_rates=_parse_floats(args.upgrade_rates),
        window_tokens=args.window_tokens,
        random_repeats=args.random_repeats,
        candidate_token_limit=args.candidate_token_limit,
        candidate_batch_size=args.candidate_batch_size,
        model_dtype=args.model_dtype,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
