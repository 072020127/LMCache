# SPDX-License-Identifier: Apache-2.0

"""Core request handling for the independent MaKV Remote Manager."""

# Standard
from collections import OrderedDict
from dataclasses import asdict, replace
from typing import Any, Mapping
import asyncio
import math
import os
import time

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.makv.config import MaKVConfig
from lmcache.v1.storage_backend.makv.format import (
    decode_client_put_envelope,
    decode_makv_object,
    encode_makv_object,
    peek_makv_header,
)
from lmcache.v1.storage_backend.makv.entropy import encode_entropy_payloads
from lmcache.v1.storage_backend.makv.metrics import REMOTE_METRICS
from lmcache.v1.storage_backend.makv.plan import (
    MaKVQuantPlan,
    compute_quant_plan_checksum,
)
from lmcache.v1.storage_backend.makv.quantizer import quantize_canonical_kv
from lmcache.v1.storage_backend.makv.precision_risk import (
    CONF_RISK_SEMANTICS,
    CONF_SCORER_VERSION,
)
from lmcache.v1.storage_backend.makv.residual import (
    reconstruct_with_residual,
    validate_residual_metadata,
)
from lmcache.v1.storage_backend.makv_remote.storage_adapter import StorageAdapter


def plan_from_dict(data: dict[str, Any]) -> MaKVQuantPlan:
    """Build and validate a quantization plan received over the wire."""
    return MaKVQuantPlan(
        protocol_version=int(data["protocol_version"]),
        importance_layout=str(data["importance_layout"]),
        token_count=int(data["token_count"]),
        chunk_start=int(data["chunk_start"]),
        chunk_length=int(data["chunk_length"]),
        bucket_bits=tuple(int(value) for value in data["bucket_bits"]),
        bucket_ids=bytes(data["bucket_ids"]),
        original_shape=tuple(int(value) for value in data["original_shape"]),
        original_strides=tuple(int(value) for value in data["original_strides"]),
        original_dtype=str(data["original_dtype"]),
        token_dim=int(data["token_dim"]),
        num_layers=int(data["num_layers"]),
        num_kv_heads=int(data["num_kv_heads"]),
        head_dim=int(data["head_dim"]),
        quant_granularity=str(data["quant_granularity"]),
        scale_dtype=str(data["scale_dtype"]),
        model_fingerprint=str(data["model_fingerprint"]),
        parallel_fingerprint=str(data["parallel_fingerprint"]),
        checksum=int(data["checksum"]),
        nan_protected_count=int(data.get("nan_protected_count", 0)),
        inf_protected_count=int(data.get("inf_protected_count", 0)),
        source_plan_hash=str(data.get("source_plan_hash", "")),
        source_strategy=str(data.get("source_strategy", "")),
        prompt_token_hash=str(data.get("prompt_token_hash", "")),
        precision_plan_schema=str(data.get("precision_plan_schema", "")),
        precision_scheme=str(data.get("precision_scheme", "shared")),
    )


def _validate_precision_risk_signal(signal: Mapping[str, Any]) -> tuple[int, float]:
    """Validate the frozen PrecisionRiskObserver transport contract."""
    if not isinstance(signal, Mapping):
        raise ValueError("MaKV precision risk signal must be an object")
    if str(signal.get("scorer_version", "")) != CONF_SCORER_VERSION:
        raise ValueError("unsupported MaKV precision risk scorer version")
    if str(signal.get("semantics", "")) != CONF_RISK_SEMANTICS:
        raise ValueError("unsupported MaKV precision risk semantics")
    if signal.get("valid") is not True:
        raise ValueError("invalid MaKV precision risk signal")
    try:
        raw_step = signal["step"]
        raw_risk = signal["risk"]
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise TypeError("step must be an integer")
        if isinstance(raw_step, float) and not raw_step.is_integer():
            raise ValueError("step must be an integer")
        if isinstance(raw_risk, bool) or not isinstance(raw_risk, (int, float)):
            raise TypeError("risk must be numeric")
        step = int(raw_step)
        risk = float(raw_risk)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("MaKV precision risk signal has invalid fields") from error
    if step < 0 or not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise ValueError("MaKV precision risk step/risk is out of range")
    return step, risk


def _risk_token_index(signal: Mapping[str, Any], step: int) -> int:
    """Return the absolute KV token position targeted by a risk signal.

    The original five-field signal only identifies a decode step. Callers
    that can map that step to a prompt/KV position should send ``token_index``
    (``kv_token_index`` is accepted as a compatibility alias). Falling back
    to ``step`` is deterministic and keeps the original contract usable for
    objects whose request positions start at zero.
    """
    raw = signal.get("token_index", signal.get("kv_token_index", step))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("MaKV precision risk token_index must be an integer")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError("MaKV precision risk token_index must be an integer")
    token_index = int(raw)
    if token_index < 0:
        raise ValueError("MaKV precision risk token_index must be non-negative")
    return token_index


def _risk_window_tokens(signal: Mapping[str, Any], configured: int) -> int:
    """Read an optional per-signal logical window length."""
    raw = signal.get("window_tokens", configured)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("MaKV precision risk window_tokens must be an integer")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError("MaKV precision risk window_tokens must be an integer")
    window_tokens = int(raw)
    if window_tokens <= 0:
        raise ValueError("MaKV precision risk window_tokens must be positive")
    return window_tokens


def _promote_bucket_ids(
    plan: MaKVQuantPlan, policy: str
) -> tuple[bytes, dict[int, int], bool]:
    """Promote every physical bucket to the next available configured width."""
    available = sorted({int(value) for value in plan.bucket_bits})
    if not available:
        raise ValueError("MaKV plan has no precision buckets")
    if policy not in ("next", "full"):
        raise ValueError(f"unsupported MaKV risk upgrade policy: {policy!r}")
    mapping: dict[int, int] = {}
    for bit in available:
        if policy == "full":
            mapping[bit] = available[-1]
            continue
        higher = [candidate for candidate in available if candidate > bit]
        mapping[bit] = higher[0] if higher else bit
    bucket_indices = {bit: index for index, bit in enumerate(plan.bucket_bits)}
    promoted: list[int] = []
    changed = False
    for bucket_id in plan.bucket_ids:
        if bucket_id < 0 or bucket_id >= len(plan.bucket_bits):
            raise ValueError("MaKV bucket map contains an invalid bucket id")
        current_bit = int(plan.bucket_bits[bucket_id])
        target_bit = mapping[current_bit]
        if target_bit != current_bit:
            changed = True
        try:
            promoted.append(bucket_indices[target_bit])
        except KeyError as error:
            raise ValueError(
                "MaKV upgrade target is not in the plan buckets"
            ) from error
    return bytes(promoted), mapping, changed


def _promote_selected_bucket_ids(
    plan: MaKVQuantPlan,
    selected_token_indices: set[int],
    policy: str,
) -> tuple[bytes, dict[int, int], bool]:
    """Promote only physical entries belonging to selected request tokens."""
    available = sorted({int(value) for value in plan.bucket_bits})
    if not available:
        raise ValueError("MaKV plan has no precision buckets")
    if policy not in ("next", "full"):
        raise ValueError(f"unsupported MaKV risk upgrade policy: {policy!r}")
    mapping: dict[int, int] = {}
    for bit in available:
        if policy == "full":
            mapping[bit] = available[-1]
        else:
            higher = [candidate for candidate in available if candidate > bit]
            mapping[bit] = higher[0] if higher else bit

    bucket_indices = {bit: index for index, bit in enumerate(plan.bucket_bits)}
    promoted: list[int] = []
    changed = False
    for flat_index, bucket_id in enumerate(plan.bucket_ids):
        if bucket_id < 0 or bucket_id >= len(plan.bucket_bits):
            raise ValueError("MaKV bucket map contains an invalid bucket id")
        local_token_index = (
            flat_index
            if plan.importance_layout == "token"
            else flat_index % plan.chunk_length
        )
        current_bit = int(plan.bucket_bits[bucket_id])
        target_bit = (
            mapping[current_bit]
            if local_token_index in selected_token_indices
            else current_bit
        )
        if target_bit != current_bit:
            changed = True
        try:
            promoted.append(bucket_indices[target_bit])
        except KeyError as error:
            raise ValueError(
                "MaKV upgrade target is not in the plan buckets"
            ) from error
    return bytes(promoted), mapping, changed


class MaKVRemoteManager:
    """Validate, quantize and store MaKV requests in a separate process."""

    def __init__(
        self,
        config: MaKVConfig,
        storage: StorageAdapter,
        *,
        memory_cache_bytes: int = 0,
        trust_validated_objects: bool = False,
    ) -> None:
        if memory_cache_bytes < 0:
            raise ValueError("memory_cache_bytes must be non-negative")
        if not math.isfinite(config.risk_upgrade_threshold) or not 0.0 <= (
            config.risk_upgrade_threshold
        ) <= 1.0:
            raise ValueError("risk_upgrade_threshold must be in [0, 1]")
        if config.risk_upgrade_policy not in ("next", "full"):
            raise ValueError("unsupported MaKV risk upgrade policy")
        if config.risk_window_tokens <= 0:
            raise ValueError("risk_window_tokens must be positive")
        if not math.isfinite(config.risk_window_ttl_s) or (
            config.risk_window_ttl_s < 0.0
        ):
            raise ValueError(
                "risk_window_ttl_s must be finite and non-negative"
            )
        self.config = config
        self.storage = storage
        self.quantize_calls = 0
        self.started_at = time.time()
        self.memory_cache_bytes = int(memory_cache_bytes)
        self.trust_validated_objects = bool(trust_validated_objects)
        self._memory_cache: OrderedDict[str, bytes] = OrderedDict()
        self._memory_cache_size = 0
        self._upgrade_locks: dict[str, asyncio.Lock] = {}
        # Durable storage always retains the base object. These two maps are
        # process-local control-plane state for a temporary precision view;
        # residuals and upgraded blobs never cross the GET response boundary.
        self._public_blob_cache: dict[str, tuple[tuple[int, int, int], bytes]] = {}
        self._precision_windows: dict[str, dict[str, Any]] = {}
        # This records only complete objects that this manager has validated
        # and successfully written, or validated after a cold read. It avoids
        # a second CRC scan on immutable manager-owned objects while retaining
        # normal structural and bounds validation on every GET.
        self._validated_objects: dict[str, tuple[int, int, int]] = {}

    async def close(self) -> None:
        """Release the configured persistence client."""
        await self.storage.close()

    def _cache_get(self, key: str) -> bytes | None:
        if self.memory_cache_bytes <= 0:
            return None
        data = self._memory_cache.pop(key, None)
        if data is None:
            return None
        self._memory_cache[key] = data
        return data

    def _cache_put(self, key: str, data: bytes) -> None:
        if self.memory_cache_bytes <= 0:
            return
        if len(data) > self.memory_cache_bytes:
            old = self._memory_cache.pop(key, None)
            if old is not None:
                self._memory_cache_size -= len(old)
            return
        old = self._memory_cache.pop(key, None)
        if old is not None:
            self._memory_cache_size -= len(old)
        while (
            self._memory_cache
            and self._memory_cache_size + len(data) > self.memory_cache_bytes
        ):
            _, evicted = self._memory_cache.popitem(last=False)
            self._memory_cache_size -= len(evicted)
        self._memory_cache[key] = data
        self._memory_cache_size += len(data)

    def _expire_precision_window(self, key: str, *, step: int | None = None) -> bool:
        """Drop a temporary view after a logical or wall-clock expiry."""
        state = self._precision_windows.get(key)
        if state is None:
            return False
        expired = False
        if step is not None and int(step) >= int(state["expires_step"]):
            expired = True
        expires_at = state.get("expires_at")
        if expires_at is not None and time.monotonic() >= float(expires_at):
            expired = True
        if not expired:
            return False
        self._precision_windows.pop(key, None)
        REMOTE_METRICS.add(
            makv_remote_precision_window_expirations=1,
            makv_remote_precision_window_restores=1,
        )
        return True

    def _active_precision_blob(self, key: str) -> bytes | None:
        """Return a live process-local upgrade view, if one exists."""
        self._expire_precision_window(key)
        state = self._precision_windows.get(key)
        if state is None:
            return None
        REMOTE_METRICS.add(makv_remote_precision_window_hits=1)
        return state["blob"]

    def _public_blob(self, key: str, data: bytes) -> bytes:
        """Remove manager-only residual/control payloads before a GET."""
        fingerprint = peek_makv_header(data)
        cached = self._public_blob_cache.get(key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

        decoded = decode_makv_object(data, copy_payloads=True)
        hidden_payload = any(
            name.startswith("residual_") or name.startswith("precision_window_")
            for name in decoded.payloads
        )
        hidden_metadata = any(
            name in decoded.metadata
            for name in ("residual", "risk_upgrade", "precision_window")
        )
        if not hidden_payload and not hidden_metadata:
            public_data = data
        else:
            metadata = dict(decoded.metadata)
            metadata.pop("residual", None)
            metadata.pop("risk_upgrade", None)
            metadata.pop("precision_window", None)
            payloads = {
                name: bytes(payload)
                for name, payload in decoded.payloads.items()
                if not (
                    name.startswith("residual_")
                    or name.startswith("precision_window_")
                )
            }
            public_data = encode_makv_object(
                object_type=decoded.object_type,
                metadata=metadata,
                payloads=payloads,
                protocol_version=decoded.protocol_version,
            )
        self._public_blob_cache[key] = (fingerprint, public_data)
        return public_data

    async def put(self, key: str, envelope_bytes: bytes, queue_ms: float) -> int:
        """Process one raw PUT envelope and atomically expose its final object."""
        started = time.perf_counter()
        decode_started = time.perf_counter()
        envelope = decode_client_put_envelope(envelope_bytes)
        decode_ms = (time.perf_counter() - decode_started) * 1000
        if envelope.key != key:
            raise ValueError("MaKV request key does not match envelope key")

        quantize_total_ms = 0.0
        plan_canonicalize_ms = 0.0
        quantize_kernel_ms = 0.0
        object_encode_ms = 0.0
        object_validate_ms = 0.0
        if envelope.object_type == "naive_fallback":
            encode_started = time.perf_counter()
            object_bytes = encode_makv_object(
                object_type="naive_fallback",
                metadata={
                    "cache_key": key,
                    "reason": envelope.metadata.get(
                        "fallback_reason", "missing_importance"
                    ),
                },
                payloads={"raw_kv_payload": envelope.raw_kv_payload},
            )
            object_encode_ms = (time.perf_counter() - encode_started) * 1000
            REMOTE_METRICS.add(makv_naive_fallbacks=1)
        else:
            try:
                quantize_started = time.perf_counter()
                (
                    object_bytes,
                    plan_canonicalize_ms,
                    quantize_kernel_ms,
                    object_encode_ms,
                ) = self._quantize(envelope.metadata, envelope.raw_kv_payload)
                quantize_total_ms = (time.perf_counter() - quantize_started) * 1000
            except Exception:
                REMOTE_METRICS.add(makv_quantize_failures=1)
                if self.config.fallback != "naive":
                    raise
                encode_started = time.perf_counter()
                object_bytes = encode_makv_object(
                    object_type="naive_fallback",
                    metadata={"cache_key": key, "reason": "remote_quantize_failure"},
                    payloads={"raw_kv_payload": envelope.raw_kv_payload},
                )
                object_encode_ms += (time.perf_counter() - encode_started) * 1000
                REMOTE_METRICS.add(makv_naive_fallbacks=1)

        validate_started = time.perf_counter()
        stored_object = decode_makv_object(object_bytes, copy_payloads=False)
        validate_residual_metadata(stored_object.metadata, stored_object.payloads)
        object_validate_ms = (time.perf_counter() - validate_started) * 1000
        storage_started = time.perf_counter()
        await self.storage.put(key, object_bytes)
        storage_put_ms = (time.perf_counter() - storage_started) * 1000
        # A new base object invalidates any process-local precision view.
        self._precision_windows.pop(key, None)
        self._public_blob_cache.pop(key, None)
        # Keep the just-written immutable blob hot for the immediate retrieve
        # phase. Disk remains the durable source after eviction or restart.
        self._cache_put(key, object_bytes)
        if self.trust_validated_objects:
            self._validated_objects[key] = peek_makv_header(object_bytes)
        REMOTE_METRICS.add(
            makv_remote_put_requests=1,
            makv_remote_quantize_queue_time_ms=queue_ms,
            makv_remote_quantize_time_ms=quantize_total_ms,
            makv_remote_put_decode_time_ms=decode_ms,
            makv_remote_plan_canonicalize_time_ms=plan_canonicalize_ms,
            makv_remote_quantize_kernel_time_ms=quantize_kernel_ms,
            makv_remote_object_encode_time_ms=object_encode_ms,
            makv_remote_object_validate_time_ms=object_validate_ms,
            makv_remote_encode_validate_time_ms=object_encode_ms + object_validate_ms,
            makv_remote_storage_put_time_ms=storage_put_ms,
            makv_remote_put_total_time_ms=(time.perf_counter() - started) * 1000,
            makv_raw_input_bytes=len(envelope.raw_kv_payload),
            makv_stored_bytes=len(object_bytes),
            makv_remote_residual_bytes=sum(
                len(value)
                for name, value in stored_object.payloads.items()
                if name.startswith("residual_")
            ),
        )
        return len(object_bytes)

    async def apply_precision_risk(
        self, key: str, signal: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Temporarily promote the risk token in a remote-only view.

        The durable object is intentionally never overwritten. This gives the
        manager a lossless rollback point when the logical decode-step window
        expires and ensures residual payloads remain inside the manager.
        """
        step, risk = _validate_precision_risk_signal(signal)
        if not key:
            raise ValueError("MaKV precision risk requires a cache key")
        token_index = _risk_token_index(signal, step)
        window_tokens = _risk_window_tokens(
            signal, self.config.risk_window_tokens
        )
        REMOTE_METRICS.add(makv_remote_risk_signals=1)
        threshold = float(self.config.risk_upgrade_threshold)

        lock = self._upgrade_locks.setdefault(key, asyncio.Lock())
        async with lock:
            started = time.perf_counter()
            expired = self._expire_precision_window(key, step=step)
            state = self._precision_windows.get(key)
            if risk < threshold:
                return {
                    "accepted": True,
                    "upgraded": False,
                    "reason": "below_threshold",
                    "risk": risk,
                    "threshold": threshold,
                    "window_active": state is not None,
                    "window_expired": expired,
                }

            if state is not None and step <= int(state["last_step"]):
                return {
                    "accepted": True,
                    "upgraded": False,
                    "reason": "stale_signal",
                    "risk": risk,
                    "threshold": threshold,
                    "last_step": int(state["last_step"]),
                    "window_active": True,
                }

            # This is an internal read. It deliberately bypasses the public
            # GET projection so residuals never leave the manager.
            object_bytes, _ = await self._get_internal_with_timing(key)
            if object_bytes is None:
                return {
                    "accepted": True,
                    "upgraded": False,
                    "reason": "cache_miss",
                    "risk": risk,
                    "threshold": threshold,
                }
            try:
                base_header = peek_makv_header(object_bytes)
                stored_object = decode_makv_object(
                    object_bytes, copy_payloads=False
                )
                if stored_object.object_type != "quantized":
                    return {
                        "accepted": True,
                        "upgraded": False,
                        "reason": "not_quantized",
                        "risk": risk,
                        "threshold": threshold,
                    }
                validate_residual_metadata(
                    stored_object.metadata, stored_object.payloads
                )
                residual_descriptor = stored_object.metadata.get("residual")
                if not isinstance(residual_descriptor, Mapping):
                    return {
                        "accepted": True,
                        "upgraded": False,
                        "reason": "residual_unavailable",
                        "risk": risk,
                        "threshold": threshold,
                    }
                plan_data = stored_object.metadata.get("plan")
                if not isinstance(plan_data, Mapping):
                    raise ValueError("MaKV object is missing its quantization plan")
                plan = plan_from_dict(dict(plan_data))
                if plan.protocol_version != 1:
                    raise ValueError("unsupported MaKV plan version")
                if plan.bucket_bits != self.config.bucket_bits:
                    raise ValueError(
                        "MaKV object bucket bits differ from manager config"
                    )
                if plan.precision_scheme != self.config.precision_scheme:
                    raise ValueError(
                        "MaKV object precision scheme differs from manager config"
                    )
                expected_bucket_ids = (
                    plan.chunk_length
                    if plan.importance_layout == "token"
                    else plan.num_layers * 2 * plan.chunk_length
                )
                if len(plan.bucket_ids) != expected_bucket_ids:
                    raise ValueError("MaKV object bucket map length mismatch")
                local_token_index = token_index - plan.chunk_start
                if not 0 <= local_token_index < plan.chunk_length:
                    return {
                        "accepted": True,
                        "upgraded": False,
                        "reason": "token_out_of_range",
                        "risk": risk,
                        "threshold": threshold,
                        "token_index": token_index,
                        "chunk_start": plan.chunk_start,
                        "chunk_length": plan.chunk_length,
                        "window_active": state is not None,
                    }
                active_token_indices = set()
                if state is not None and state["base_header"] == base_header:
                    active_token_indices.update(
                        int(value) for value in state["active_token_indices"]
                    )
                active_token_indices.add(token_index)
                active_local_indices = {
                    value - plan.chunk_start for value in active_token_indices
                }
                if any(
                    value < 0 or value >= plan.chunk_length
                    for value in active_local_indices
                ):
                    raise ValueError("MaKV precision window token is out of range")
                promoted_ids, promotion_map, changed = _promote_selected_bucket_ids(
                    plan,
                    active_local_indices,
                    self.config.risk_upgrade_policy,
                )
                expires_step = step + window_tokens
                expires_at = (
                    time.monotonic() + self.config.risk_window_ttl_s
                    if self.config.risk_window_ttl_s > 0.0
                    else None
                )
                if not changed:
                    return {
                        "accepted": True,
                        "upgraded": False,
                        "reason": "already_highest_precision",
                        "risk": risk,
                        "threshold": threshold,
                        "token_index": token_index,
                        "window_active": False,
                    }
                if (
                    state is not None
                    and state["base_header"] == base_header
                    and state["bucket_ids"] == promoted_ids
                ):
                    state.update(
                        {
                            "active_token_indices": sorted(active_token_indices),
                            "last_step": step,
                            "expires_step": expires_step,
                            "expires_at": expires_at,
                        }
                    )
                    REMOTE_METRICS.add(
                        makv_remote_precision_window_refreshes=1
                    )
                    return {
                        "accepted": True,
                        "upgraded": False,
                        "reason": "window_extended",
                        "risk": risk,
                        "threshold": threshold,
                        "step": step,
                        "token_index": token_index,
                        "active_token_indices": sorted(active_token_indices),
                        "expires_step": expires_step,
                        "window_active": True,
                    }

                # This is an intentionally rare manager-side operation. It
                # uses the stored residual to recover the source approximation
                # without retaining the original raw KV as a second object.
                wire_tensor = reconstruct_with_residual(
                    stored_object.metadata, stored_object.payloads
                )
                canonical = (
                    wire_tensor.view(
                        2,
                        plan.num_layers,
                        plan.chunk_length,
                        plan.num_kv_heads,
                        plan.head_dim,
                    )
                    .permute(1, 0, 2, 3, 4)
                    .contiguous()
                )
                promoted_plan = replace(
                    plan,
                    bucket_ids=promoted_ids,
                    checksum=compute_quant_plan_checksum(plan, promoted_ids),
                )
                self.quantize_calls += 1
                quantize_started = time.perf_counter()
                quant_metadata, payloads = quantize_canonical_kv(
                    canonical, promoted_plan, self.config
                )
                quantize_ms = (time.perf_counter() - quantize_started) * 1000
                quant_metadata["cache_key"] = key
                entropy_metadata, payloads = encode_entropy_payloads(
                    quant_metadata,
                    payloads,
                    codec=self.config.entropy_codec,
                    backend=self.config.entropy_backend,
                    require_cuda=self.config.entropy_require_cuda,
                )
                entropy_metadata["risk_upgrade"] = {
                    "version": 1,
                    "step": step,
                    "risk": risk,
                    "threshold": threshold,
                    "policy": self.config.risk_upgrade_policy,
                    "promotion_map": {
                        str(source): int(target)
                        for source, target in promotion_map.items()
                    },
                }
                entropy_metadata["precision_window"] = {
                    "version": 1,
                    "start_step": (
                        int(state["start_step"]) if state is not None else step
                    ),
                    "last_step": step,
                    "expires_step": expires_step,
                    "window_tokens": window_tokens,
                    "token_indices": sorted(active_token_indices),
                }
                upgraded_bytes = encode_makv_object(
                    object_type="quantized",
                    metadata=entropy_metadata,
                    payloads=payloads,
                )
                upgraded_object = decode_makv_object(
                    upgraded_bytes, copy_payloads=False
                )
                validate_residual_metadata(
                    upgraded_object.metadata, upgraded_object.payloads
                )
                # The upgraded value is process-local. The original complete
                # object remains the durable rollback point in storage.
                self._precision_windows[key] = {
                    "blob": upgraded_bytes,
                    "bucket_ids": promoted_ids,
                    "base_header": base_header,
                    "active_token_indices": sorted(active_token_indices),
                    "start_step": (
                        int(state["start_step"]) if state is not None else step
                    ),
                    "last_step": step,
                    "expires_step": expires_step,
                    "expires_at": expires_at,
                }
                self._public_blob_cache.pop(key, None)
                elapsed_ms = (time.perf_counter() - started) * 1000
                residual_bytes = sum(
                    len(value)
                    for name, value in upgraded_object.payloads.items()
                    if name.startswith("residual_")
                )
                REMOTE_METRICS.add(
                    makv_remote_precision_upgrades=1,
                    makv_remote_quantize_time_ms=quantize_ms,
                    makv_remote_residual_upgrade_time_ms=elapsed_ms,
                    makv_remote_residual_bytes=residual_bytes,
                    makv_remote_precision_window_activations=(
                        int(state is None)
                    ),
                    makv_remote_precision_window_refreshes=(
                        int(state is not None)
                    ),
                )
                return {
                    "accepted": True,
                    "upgraded": True,
                    "reason": "promoted",
                    "risk": risk,
                    "threshold": threshold,
                    "step": step,
                    "token_index": token_index,
                    "active_token_indices": sorted(active_token_indices),
                    "expires_step": expires_step,
                    "window_tokens": window_tokens,
                    "promotion_map": {
                        str(source): int(target)
                        for source, target in promotion_map.items()
                    },
                    "view_bytes": len(upgraded_bytes),
                    "upgrade_time_ms": elapsed_ms,
                    "window_active": True,
                }
            except (
                IndexError,
                KeyError,
                OverflowError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                REMOTE_METRICS.add(makv_remote_precision_upgrade_failures=1)
                return {
                    "accepted": True,
                    "upgraded": False,
                    "reason": "upgrade_failed",
                    "error": f"{type(error).__name__}: {error}",
                    "risk": risk,
                    "threshold": threshold,
                }

    def _quantize(
        self, metadata: dict[str, Any], raw_payload: bytes
    ) -> tuple[bytes, float, float, float]:
        """Quantize one raw object with separate canonicalization timing."""
        canonicalize_started = time.perf_counter()
        plan_data = metadata.get("plan")
        if not isinstance(plan_data, dict):
            raise ValueError("MaKV quantized PUT requires a plan")
        plan = plan_from_dict(plan_data)
        if plan.protocol_version != 1:
            raise ValueError("unsupported MaKV plan version")
        if plan.bucket_bits != self.config.bucket_bits:
            raise ValueError("MaKV plan bucket bits differ from manager config")
        if plan.precision_scheme != self.config.precision_scheme:
            raise ValueError("MaKV plan precision scheme differs from manager config")
        expected_bucket_ids = (
            plan.chunk_length
            if plan.importance_layout == "token"
            else plan.num_layers * 2 * plan.chunk_length
        )
        if len(plan.bucket_ids) != expected_bucket_ids:
            raise ValueError("MaKV plan bucket map length mismatch")
        dtype_name = plan.original_dtype.replace("torch.", "")
        if dtype_name not in ("float16", "bfloat16"):
            raise ValueError("MaKV raw payload dtype must be float16 or bfloat16")
        dtype = getattr(torch, dtype_name)
        expected_elements = 1
        for dimension in plan.original_shape:
            expected_elements *= dimension
        if len(raw_payload) != expected_elements * dtype.itemsize:
            raise ValueError("MaKV raw payload length mismatch")
        tensor = torch.frombuffer(bytearray(raw_payload), dtype=dtype)
        tensor = tensor.view(plan.original_shape)
        if (
            len(plan.original_shape) != 4
            or plan.original_shape[0] != 2
            or plan.original_shape[1] != plan.num_layers
            or plan.original_shape[2] != plan.chunk_length
            or plan.original_shape[3] != plan.num_kv_heads * plan.head_dim
        ):
            raise ValueError("MaKV raw payload shape is not canonical KV_2LTD")
        canonical = (
            tensor.view(
                2,
                plan.num_layers,
                plan.chunk_length,
                plan.num_kv_heads,
                plan.head_dim,
            )
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        plan_canonicalize_ms = (time.perf_counter() - canonicalize_started) * 1000
        self.quantize_calls += 1
        quantize_started = time.perf_counter()
        # QDM is a manager-local shadow observer.  It is deliberately not
        # controlled by request metadata or consulted by precision planning.
        if self.config.enable_qdm:
            quant_metadata, payloads = quantize_canonical_kv(
                canonical,
                plan,
                self.config,
                enable_qdm=True,
                qdm_block_size=self.config.qdm_block_size,
                qdm_quantizer_version=self.config.qdm_quantizer_version,
            )
        else:
            quant_metadata, payloads = quantize_canonical_kv(
                canonical,
                plan,
                self.config,
            )
        quantize_kernel_ms = (time.perf_counter() - quantize_started) * 1000
        quant_metadata["cache_key"] = str(metadata["key"])
        entropy_started = time.perf_counter()
        entropy_metadata, payloads = encode_entropy_payloads(
            quant_metadata,
            payloads,
            codec=self.config.entropy_codec,
            backend=self.config.entropy_backend,
            require_cuda=self.config.entropy_require_cuda,
        )
        entropy_ms = (time.perf_counter() - entropy_started) * 1000
        if entropy_metadata is not quant_metadata:
            entropy_descriptor = entropy_metadata.get("entropy", {})
            REMOTE_METRICS.add(
                makv_remote_entropy_encode_calls=1,
                makv_remote_entropy_encode_time_ms=entropy_ms,
                makv_remote_entropy_input_bytes=int(
                    entropy_descriptor.get("input_bytes", 0)
                ),
                makv_remote_entropy_output_bytes=int(
                    entropy_descriptor.get("output_bytes", 0)
                ),
            )
        quant_metadata = entropy_metadata
        encode_started = time.perf_counter()
        encoded = encode_makv_object(
            object_type="quantized",
            metadata=quant_metadata,
            payloads=payloads,
        )
        encode_ms = (time.perf_counter() - encode_started) * 1000
        return (
            encoded,
            plan_canonicalize_ms,
            quantize_kernel_ms,
            encode_ms,
        )

    async def get(self, key: str) -> bytes | None:
        """Return the client-visible MaKV object without dequantizing it."""
        data, _ = await self.get_with_timing(key)
        return data

    async def _get_internal_with_timing(
        self, key: str
    ) -> tuple[bytes | None, dict[str, Any]]:
        """Read and validate the complete manager-owned object."""
        started = time.perf_counter()
        hot_cache_started = time.perf_counter()
        data = self._cache_get(key)
        hot_cache_ms = (time.perf_counter() - hot_cache_started) * 1000
        storage_ms = 0.0
        validate_ms = 0.0
        hot_cache_hit = data is not None
        if data is not None:
            REMOTE_METRICS.add(makv_memory_cache_hits=1)
        else:
            REMOTE_METRICS.add(makv_memory_cache_misses=1)
            storage_started = time.perf_counter()
            data = await self.storage.get(key)
            storage_ms = (time.perf_counter() - storage_started) * 1000
            if data is not None:
                # The manager validates the complete object before returning it;
                # avoid slicing every large payload just to check the key.
                validate_started = time.perf_counter()
                verify_checksum = True
                if self.trust_validated_objects:
                    try:
                        header = peek_makv_header(data)
                    except ValueError:
                        header = None
                    verify_checksum = self._validated_objects.get(key) != header
                decoded = decode_makv_object(
                    data,
                    copy_payloads=False,
                    verify_checksum=verify_checksum,
                )
                validate_residual_metadata(decoded.metadata, decoded.payloads)
                validate_ms = (time.perf_counter() - validate_started) * 1000
                if decoded.metadata.get("cache_key") != key:
                    raise ValueError("stored MaKV cache key mismatch")
                REMOTE_METRICS.add(
                    makv_remote_get_checksum_verifications=int(verify_checksum),
                    makv_remote_get_checksum_skips=int(not verify_checksum),
                )
                if self.trust_validated_objects:
                    self._validated_objects[key] = peek_makv_header(data)
                self._cache_put(key, data)

        total_ms = (time.perf_counter() - started) * 1000
        timing = {
            "hot_cache_hit": hot_cache_hit,
            "hot_cache_ms": hot_cache_ms,
            "storage_ms": storage_ms,
            "validate_ms": validate_ms,
            "total_ms": total_ms,
        }
        return data, timing

    async def get_with_timing(
        self, key: str
    ) -> tuple[bytes | None, dict[str, Any]]:
        """Return a public object and manager-local GET timing components."""
        started = time.perf_counter()
        override = self._active_precision_blob(key)
        if override is not None:
            data = override
            timing = {
                "hot_cache_hit": True,
                "precision_window_hit": True,
                "hot_cache_ms": 0.0,
                "storage_ms": 0.0,
                "validate_ms": 0.0,
                "total_ms": (time.perf_counter() - started) * 1000,
            }
        else:
            data, timing = await self._get_internal_with_timing(key)
            timing["precision_window_hit"] = False
        public_data = self._public_blob(key, data) if data is not None else None
        total_ms = (time.perf_counter() - started) * 1000
        timing["total_ms"] = total_ms
        REMOTE_METRICS.add(
            makv_get_quantized_bytes=(
                len(public_data) if public_data is not None else 0
            ),
            makv_remote_get_requests=1,
            makv_remote_get_hot_cache_time_ms=float(timing["hot_cache_ms"]),
            makv_remote_get_storage_time_ms=float(timing["storage_ms"]),
            makv_remote_get_validate_time_ms=float(timing["validate_ms"]),
            makv_remote_get_total_time_ms=total_ms,
        )
        return public_data, timing

    async def get_many_with_timing(
        self, keys: list[str]
    ) -> list[tuple[bytes | None, dict[str, Any]]]:
        """Read a batch with one adapter operation when supported.

        ``storage_ms`` is charged to the first miss so the existing per-object
        sum remains a physical batch wall time rather than N times the same
        MGET.  ``batch_*`` fields carry the exact batch critical-path values.
        The complete object validation still runs for every object.
        """
        if not keys:
            return []

        batch_started = time.perf_counter()
        results: list[tuple[bytes | None, dict[str, Any]]] = []
        values: list[bytes | None] = [None] * len(keys)
        hot_cache_times: list[float] = [0.0] * len(keys)
        hot_cache_hits: list[bool] = [False] * len(keys)
        precision_window_hits: list[bool] = [False] * len(keys)
        missing_indices: list[int] = []
        missing_keys: list[str] = []

        for index, key in enumerate(keys):
            override = self._active_precision_blob(key)
            if override is not None:
                values[index] = override
                hot_cache_hits[index] = True
                precision_window_hits[index] = True
                continue
            hot_cache_started = time.perf_counter()
            value = self._cache_get(key)
            hot_cache_times[index] = (time.perf_counter() - hot_cache_started) * 1000
            if value is None:
                missing_indices.append(index)
                missing_keys.append(key)
                REMOTE_METRICS.add(makv_memory_cache_misses=1)
            else:
                values[index] = value
                hot_cache_hits[index] = True
                REMOTE_METRICS.add(makv_memory_cache_hits=1)

        storage_ms = 0.0
        if missing_keys:
            storage_started = time.perf_counter()
            get_many = getattr(self.storage, "get_many", None)
            if callable(get_many):
                loaded = await get_many(missing_keys)
            else:
                # Keep third-party adapters written against the original
                # single-key protocol working during the transition.
                loaded = await asyncio.gather(
                    *(self.storage.get(key) for key in missing_keys)
                )
            storage_ms = (time.perf_counter() - storage_started) * 1000
            if len(loaded) != len(missing_indices):
                raise RuntimeError(
                    "MaKV storage batch returned an unexpected value count"
                )
            for index, value in zip(missing_indices, loaded, strict=True):
                values[index] = value

        validate_ms: list[float] = [0.0] * len(keys)
        validate_batch_started = time.perf_counter()
        for index, (key, value) in enumerate(zip(keys, values, strict=True)):
            if value is None:
                continue
            if not hot_cache_hits[index]:
                validate_started = time.perf_counter()
                verify_checksum = True
                if self.trust_validated_objects:
                    try:
                        header = peek_makv_header(value)
                    except ValueError:
                        header = None
                    verify_checksum = self._validated_objects.get(key) != header
                decoded = decode_makv_object(
                    value,
                    copy_payloads=False,
                    verify_checksum=verify_checksum,
                )
                validate_residual_metadata(decoded.metadata, decoded.payloads)
                validate_ms[index] = (time.perf_counter() - validate_started) * 1000
                if decoded.metadata.get("cache_key") != key:
                    raise ValueError("stored MaKV cache key mismatch")
                REMOTE_METRICS.add(
                    makv_remote_get_checksum_verifications=int(verify_checksum),
                    makv_remote_get_checksum_skips=int(not verify_checksum),
                )
                if self.trust_validated_objects:
                    self._validated_objects[key] = peek_makv_header(value)
                self._cache_put(key, value)
        validate_batch_ms = (time.perf_counter() - validate_batch_started) * 1000
        batch_total_ms = (time.perf_counter() - batch_started) * 1000
        first_missing = missing_indices[0] if missing_indices else None
        public_values = [
            self._public_blob(key, value) if value is not None else None
            for key, value in zip(keys, values, strict=True)
        ]
        for value in public_values:
            if value is not None:
                REMOTE_METRICS.add(makv_get_quantized_bytes=len(value))

        REMOTE_METRICS.add(
            makv_remote_get_batch_requests=1,
            makv_remote_get_batch_objects=len(keys),
            makv_remote_get_batch_storage_time_ms=storage_ms,
            makv_remote_get_batch_validate_time_ms=validate_batch_ms,
            makv_remote_get_batch_total_time_ms=batch_total_ms,
        )
        for index, value in enumerate(public_values):
            total_ms = (
                hot_cache_times[index]
                + (storage_ms if index == first_missing else 0.0)
                + validate_ms[index]
            )
            timing = {
                "hot_cache_hit": hot_cache_hits[index],
                "precision_window_hit": precision_window_hits[index],
                "hot_cache_ms": hot_cache_times[index],
                "storage_ms": storage_ms if index == first_missing else 0.0,
                "validate_ms": validate_ms[index],
                "total_ms": total_ms,
                "batch_storage_ms": storage_ms,
                "batch_validate_ms": validate_batch_ms,
                "batch_total_ms": batch_total_ms,
            }
            REMOTE_METRICS.add(
                makv_remote_get_requests=1,
                makv_remote_get_hot_cache_time_ms=hot_cache_times[index],
                makv_remote_get_storage_time_ms=(
                    storage_ms if index == first_missing else 0.0
                ),
                makv_remote_get_validate_time_ms=validate_ms[index],
                makv_remote_get_total_time_ms=total_ms,
            )
            results.append((value, timing))
        return results

    async def delete(self, key: str) -> bool:
        """Delete durable and hot copies of one object."""
        cached = self._memory_cache.pop(key, None)
        if cached is not None:
            self._memory_cache_size -= len(cached)
        self._public_blob_cache.pop(key, None)
        self._precision_windows.pop(key, None)
        self._validated_objects.pop(key, None)
        return await self.storage.delete(key)

    async def health(self) -> dict[str, Any]:
        """Return process identity and manager metrics."""
        metrics = asdict(REMOTE_METRICS.snapshot())
        raw_bytes = int(metrics["makv_raw_input_bytes"])
        stored_bytes = int(metrics["makv_stored_bytes"])
        return {
            "pid": os.getpid(),
            "uptime_s": time.time() - self.started_at,
            "quantize_calls": self.quantize_calls,
            "memory_cache_bytes": self._memory_cache_size,
            "memory_cache_capacity_bytes": self.memory_cache_bytes,
            "memory_cache_entries": len(self._memory_cache),
            "trust_validated_objects": self.trust_validated_objects,
            "risk_window_tokens": self.config.risk_window_tokens,
            "risk_window_ttl_s": self.config.risk_window_ttl_s,
            "precision_windows": len(self._precision_windows),
            "precision_window_tokens": sum(
                len(state["active_token_indices"])
                for state in self._precision_windows.values()
            ),
            "compression_ratio": raw_bytes / stored_bytes if stored_bytes else 0.0,
            "metrics": metrics,
        }
