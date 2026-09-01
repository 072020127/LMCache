# SPDX-License-Identifier: Apache-2.0

"""In-process MaKV metrics used by tests and benchmarks.

CUDA work is asynchronous, so host ``perf_counter`` intervals around a
``Tensor.to(..., non_blocking=True)`` call only measure submission overhead.
The restore accumulator therefore retains CUDA events and folds them into the
snapshot only after their terminal event has completed. Production callers use
``query()``; the explicit benchmark path may request an event wait.
"""

# Standard
from dataclasses import dataclass
from typing import Any
import threading


@dataclass
class MaKVMetricsSnapshot:
    makv_plan_time_ms: float = 0.0
    # ``makv_plan_time_ms`` is retained for compatibility and now measures
    # only deterministic importance/precision-plan construction. Payload
    # materialization and binary-envelope copies are reported separately.
    makv_client_plan_build_time_ms: float = 0.0
    makv_client_raw_payload_copy_time_ms: float = 0.0
    makv_client_envelope_encode_time_ms: float = 0.0
    makv_client_serialize_total_time_ms: float = 0.0
    makv_put_raw_bytes: int = 0
    makv_put_plan_bytes: int = 0
    makv_client_quantize_calls: int = 0
    makv_scout_submit_calls: int = 0
    makv_scout_submit_time_ms: float = 0.0
    makv_scout_wait_calls: int = 0
    makv_scout_wait_time_ms: float = 0.0
    makv_scout_score_time_ms: float = 0.0
    makv_scout_overlap_hidden_time_ms: float = 0.0
    makv_remote_quantize_time_ms: float = 0.0
    makv_remote_quantize_queue_time_ms: float = 0.0
    makv_remote_residual_bytes: int = 0
    makv_remote_risk_signals: int = 0
    makv_remote_precision_upgrades: int = 0
    makv_remote_precision_upgrade_failures: int = 0
    makv_remote_residual_upgrade_time_ms: float = 0.0
    makv_remote_precision_window_activations: int = 0
    makv_remote_precision_window_refreshes: int = 0
    makv_remote_precision_window_expirations: int = 0
    makv_remote_precision_window_hits: int = 0
    makv_remote_precision_window_restores: int = 0
    makv_raw_input_bytes: int = 0
    makv_stored_bytes: int = 0
    makv_quantize_failures: int = 0
    makv_naive_fallbacks: int = 0
    makv_get_quantized_bytes: int = 0
    makv_memory_cache_hits: int = 0
    makv_memory_cache_misses: int = 0
    makv_remote_put_requests: int = 0
    makv_remote_put_decode_time_ms: float = 0.0
    makv_remote_plan_canonicalize_time_ms: float = 0.0
    makv_remote_quantize_kernel_time_ms: float = 0.0
    makv_remote_entropy_encode_calls: int = 0
    makv_remote_entropy_encode_time_ms: float = 0.0
    makv_remote_entropy_input_bytes: int = 0
    makv_remote_entropy_output_bytes: int = 0
    makv_remote_object_encode_time_ms: float = 0.0
    makv_remote_object_validate_time_ms: float = 0.0
    makv_remote_encode_validate_time_ms: float = 0.0
    makv_remote_storage_put_time_ms: float = 0.0
    makv_remote_put_total_time_ms: float = 0.0
    makv_remote_get_requests: int = 0
    makv_remote_get_hot_cache_time_ms: float = 0.0
    makv_remote_get_storage_time_ms: float = 0.0
    makv_remote_get_validate_time_ms: float = 0.0
    makv_remote_get_total_time_ms: float = 0.0
    makv_remote_get_checksum_verifications: int = 0
    makv_remote_get_checksum_skips: int = 0
    makv_remote_get_batch_requests: int = 0
    makv_remote_get_batch_objects: int = 0
    makv_remote_get_batch_storage_time_ms: float = 0.0
    makv_remote_get_batch_validate_time_ms: float = 0.0
    makv_remote_get_batch_total_time_ms: float = 0.0
    makv_remote_get_batch_blob_requests: int = 0
    makv_remote_get_batch_blob_bytes: int = 0
    makv_remote_get_stream_requests: int = 0
    makv_remote_get_stream_objects: int = 0
    makv_remote_get_stream_first_object_time_ms: float = 0.0
    makv_remote_get_stream_send_time_ms: float = 0.0
    makv_remote_get_stream_total_time_ms: float = 0.0
    makv_client_put_connect_time_ms: float = 0.0
    makv_client_put_send_time_ms: float = 0.0
    makv_client_put_response_time_ms: float = 0.0
    makv_client_put_total_time_ms: float = 0.0
    makv_client_get_batches: int = 0
    makv_client_get_objects: int = 0
    makv_client_get_connect_time_ms: float = 0.0
    makv_client_get_send_time_ms: float = 0.0
    makv_client_get_first_response_time_ms: float = 0.0
    makv_client_get_receive_time_ms: float = 0.0
    makv_client_get_total_time_ms: float = 0.0
    makv_client_get_batch_blob_frames: int = 0
    makv_client_get_batch_blob_bytes: int = 0
    makv_client_get_stream_requests: int = 0
    makv_client_get_stream_frames: int = 0
    makv_client_get_stream_bytes: int = 0
    makv_client_pinned_receive_bytes: int = 0
    makv_client_pinned_receive_fallbacks: int = 0
    makv_client_deserialize_time_ms: float = 0.0
    makv_restore_calls: int = 0
    makv_restore_payload_bytes: int = 0
    makv_restore_cpu_blob_pin_time_ms: float = 0.0
    makv_restore_cpu_view_validate_time_ms: float = 0.0
    makv_restore_cpu_dtype_convert_time_ms: float = 0.0
    makv_restore_cpu_pin_time_ms: float = 0.0
    makv_restore_cpu_prepare_time_ms: float = 0.0
    makv_h2d_bytes: int = 0
    makv_h2d_time_ms: float = 0.0
    makv_dequant_kernel_time_ms: float = 0.0
    makv_entropy_decode_calls: int = 0
    makv_entropy_decode_time_ms: float = 0.0
    makv_entropy_decode_bytes: int = 0
    makv_restore_gpu_total_time_ms: float = 0.0
    makv_restore_total_time_ms: float = 0.0
    makv_kernel_launch_count: int = 0
    makv_cuda_pending_traces: int = 0


@dataclass
class _PendingCudaRestoreTrace:
    """One restore's stream-local CUDA events and ownership scope."""

    scope_id: int | None
    h2d_start: Any
    h2d_end: Any
    kernel_end: Any
    # Non-blocking H2D copies must retain their CPU sources until this event
    # completes.  Keeping the GPU inputs here also prevents allocator reuse
    # before the paged scatter has consumed them.
    keepalive: tuple[Any, ...]


class MaKVMetrics:
    """Thread-safe metrics accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = MaKVMetricsSnapshot()
        self._pending_cuda_traces: list[_PendingCudaRestoreTrace] = []
        self._scopes: dict[int, MaKVMetricsSnapshot] = {}
        self._next_scope_id = 1

    def reset(self) -> None:
        with self._lock:
            self._snapshot = MaKVMetricsSnapshot()
            self._pending_cuda_traces.clear()
            self._scopes.clear()
            self._next_scope_id = 1

    def snapshot(self) -> MaKVMetricsSnapshot:
        with self._lock:
            self._collect_ready_cuda_locked()
            return MaKVMetricsSnapshot(**self._snapshot.__dict__)

    def add(self, **kwargs) -> None:
        with self._lock:
            self._add_locked(self._snapshot, kwargs)

    def begin_restore_scope(self) -> int:
        """Create a request-local scope for a batched paged restore."""
        with self._lock:
            scope_id = self._next_scope_id
            self._next_scope_id += 1
            self._scopes[scope_id] = MaKVMetricsSnapshot()
            return scope_id

    def add_restore(self, scope_id: int | None, **kwargs) -> None:
        """Add CPU-side restore work to the global and request-local totals."""
        with self._lock:
            self._add_locked(self._snapshot, kwargs)
            if scope_id is not None and scope_id in self._scopes:
                self._add_locked(self._scopes[scope_id], kwargs)

    def record_cuda_restore(
        self,
        scope_id: int | None,
        *,
        h2d_start: Any,
        h2d_end: Any,
        kernel_end: Any,
        payload_bytes: int,
        h2d_bytes: int,
        kernel_launch_count: int,
        keepalive: tuple[Any, ...] = (),
    ) -> None:
        """Queue an asynchronous GPU timing result without synchronizing."""
        values = {
            "makv_restore_calls": 1,
            "makv_restore_payload_bytes": int(payload_bytes),
            "makv_h2d_bytes": int(h2d_bytes),
            "makv_kernel_launch_count": int(kernel_launch_count),
        }
        with self._lock:
            self._add_locked(self._snapshot, values)
            if scope_id is not None and scope_id in self._scopes:
                self._add_locked(self._scopes[scope_id], values)
            self._pending_cuda_traces.append(
                _PendingCudaRestoreTrace(
                    scope_id=scope_id,
                    h2d_start=h2d_start,
                    h2d_end=h2d_end,
                    kernel_end=kernel_end,
                    keepalive=keepalive,
                )
            )
            self._snapshot.makv_cuda_pending_traces = len(self._pending_cuda_traces)

    def finish_restore_scope(
        self, scope_id: int, *, wait: bool = False
    ) -> MaKVMetricsSnapshot:
        """Collect and return one batched restore's timing delta.

        ``wait`` is intended only for an explicit benchmark. Normal V2/V3
        callers invoke this after their existing load-stream synchronize, so
        no extra device synchronization is introduced here.
        """
        if wait:
            with self._lock:
                events = [
                    trace.kernel_end
                    for trace in self._pending_cuda_traces
                    if trace.scope_id == scope_id
                ]
            for event in events:
                event.synchronize()
        with self._lock:
            self._collect_ready_cuda_locked()
            scope = self._scopes.pop(scope_id, MaKVMetricsSnapshot())
            return MaKVMetricsSnapshot(**scope.__dict__)

    @staticmethod
    def _add_locked(snapshot: MaKVMetricsSnapshot, values: dict[str, Any]) -> None:
        for key, value in values.items():
            current = getattr(snapshot, key)
            setattr(snapshot, key, current + value)

    def _collect_ready_cuda_locked(self) -> None:
        pending: list[_PendingCudaRestoreTrace] = []
        for trace in self._pending_cuda_traces:
            try:
                complete = bool(trace.kernel_end.query())
            except RuntimeError:
                # Preserve the trace for a later poll when a stream is still
                # being initialized or the event has not reached the device.
                complete = False
            if not complete:
                pending.append(trace)
                continue
            values = {
                "makv_h2d_time_ms": trace.h2d_start.elapsed_time(trace.h2d_end),
                "makv_dequant_kernel_time_ms": trace.h2d_end.elapsed_time(
                    trace.kernel_end
                ),
                "makv_restore_gpu_total_time_ms": trace.h2d_start.elapsed_time(
                    trace.kernel_end
                ),
                # Kept for compatibility with existing dashboards. It now
                # means actual GPU restore time rather than host submission.
                "makv_restore_total_time_ms": trace.h2d_start.elapsed_time(
                    trace.kernel_end
                ),
            }
            self._add_locked(self._snapshot, values)
            if trace.scope_id is not None and trace.scope_id in self._scopes:
                self._add_locked(self._scopes[trace.scope_id], values)
        self._pending_cuda_traces = pending
        self._snapshot.makv_cuda_pending_traces = len(pending)


CLIENT_METRICS = MaKVMetrics()
REMOTE_METRICS = MaKVMetrics()
RESTORE_METRICS = MaKVMetrics()
