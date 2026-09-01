# SPDX-License-Identifier: Apache-2.0

"""Asynchronous bridge from vLLM decode logits to MaKV risk policy.

The bridge is intentionally opt-in.  It computes the frozen CONF-MaKV signal
from logits produced by the active model execution, then hands the signal to
the LMCache engine.  It never reads a reference output or changes the local
precision policy.
"""

from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import queue
import threading
from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

from .precision_risk import PrecisionRiskSignal, compute_precision_risk_signal

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeRiskWork:
    """Immutable work item retained until the producing CUDA stream is safe."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    token_index: int
    step: int
    window_tokens: int
    logits: torch.Tensor
    ready_event: Any | None
    request_configs: Mapping[str, Any] | None


@dataclass
class RuntimeRiskStats:
    """Counters for the opt-in runtime observer."""

    submitted: int = 0
    dropped: int = 0
    invalid: int = 0
    scored: int = 0
    accepted: int = 0
    failed: int = 0


RiskSink = Callable[
    [
        str,
        Sequence[int],
        int,
        PrecisionRiskSignal,
        Mapping[str, Any] | None,
    ],
    Mapping[str, Any] | None,
]


class RuntimeRiskDispatcher:
    """Score actual logits off the model thread and report them to MaKV.

    The producer only records a stream dependency and enqueues a bounded work
    item.  The worker waits on that dependency, computes the frozen scorer, and
    invokes ``sink``.  Queue overflow and scorer/transport failures are
    fail-closed: the model request continues without a precision upgrade.
    """

    def __init__(
        self,
        sink: RiskSink,
        *,
        max_queue: int = 128,
        window_tokens: int = 16,
    ) -> None:
        if max_queue <= 0:
            raise ValueError("max_queue must be positive")
        if window_tokens <= 0:
            raise ValueError("window_tokens must be positive")
        self._sink = sink
        self._window_tokens = int(window_tokens)
        self._queue: queue.Queue[RuntimeRiskWork | None] = queue.Queue(
            maxsize=max_queue
        )
        self._stats = RuntimeRiskStats()
        self._stats_lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._run,
            name="makv-runtime-risk",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        token_index: int,
        logits: torch.Tensor,
        *,
        step: int,
        request_configs: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue one real model-logit observation.

        ``token_index`` is an absolute prompt/KV position.  No step-based
        fallback is permitted here because a decode step is not a KV position.
        """
        if self._closed or not request_id or not isinstance(logits, torch.Tensor):
            self._increment("invalid")
            return False
        if (
            isinstance(token_index, bool)
            or not isinstance(token_index, int)
            or token_index < 0
            or token_index >= len(prompt_token_ids)
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            self._increment("invalid")
            return False
        if logits.ndim not in (1, 2) or logits.numel() == 0:
            self._increment("invalid")
            return False

        ready_event: Any | None = None
        if logits.is_cuda:
            # The worker uses its own CUDA stream.  This event makes the
            # producer stream dependency explicit without a device-wide sync.
            with torch.cuda.device(logits.device):
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(logits.device))

        work = RuntimeRiskWork(
            request_id=request_id,
            prompt_token_ids=tuple(int(value) for value in prompt_token_ids),
            token_index=token_index,
            step=step,
            window_tokens=self._window_tokens,
            logits=logits.detach(),
            ready_event=ready_event,
            request_configs=(
                None if request_configs is None else dict(request_configs)
            ),
        )
        try:
            self._queue.put_nowait(work)
        except queue.Full:
            self._increment("dropped")
            return False
        self._increment("submitted")
        return True

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until all queued observations have been handled."""
        if timeout is None:
            self._queue.join()
            return True
        completed = threading.Event()

        def wait_for_queue() -> None:
            self._queue.join()
            completed.set()

        waiter = threading.Thread(target=wait_for_queue, daemon=True)
        waiter.start()
        return completed.wait(timeout)

    def stats(self) -> dict[str, int]:
        """Return a snapshot of observer counters."""
        with self._stats_lock:
            return {
                "submitted": self._stats.submitted,
                "dropped": self._stats.dropped,
                "invalid": self._stats.invalid,
                "scored": self._stats.scored,
                "accepted": self._stats.accepted,
                "failed": self._stats.failed,
                "queue_size": self._queue.qsize(),
            }

    def close(self, *, wait: bool = True) -> None:
        """Stop the dispatcher, optionally draining queued observations."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if wait:
                self._queue.join()
            # ``put`` may wait briefly for a slot when a non-draining close is
            # requested, but guarantees that the daemon worker terminates.
            self._queue.put(None)
        if wait:
            self._worker.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if work is None:
                    return
                self._process(work)
            finally:
                self._queue.task_done()

    def _process(self, work: RuntimeRiskWork) -> None:
        try:
            if work.ready_event is not None:
                with torch.cuda.device(work.logits.device):
                    torch.cuda.current_stream(work.logits.device).wait_event(
                        work.ready_event
                    )
            signal = compute_precision_risk_signal(work.logits, step=work.step)
            signal = signal.for_kv_token(
                work.token_index,
                window_tokens=work.window_tokens,
            )
            self._increment("scored")
            response = self._sink(
                work.request_id,
                work.prompt_token_ids,
                work.token_index,
                signal,
                work.request_configs,
            )
            if response is not None and bool(response.get("accepted", False)):
                self._increment("accepted")
        except Exception:
            self._increment("failed")
            logger.exception(
                "Runtime MaKV risk observation failed for request %s at token %d",
                work.request_id,
                work.token_index,
            )

    def _increment(self, field: str) -> None:
        with self._stats_lock:
            setattr(self._stats, field, getattr(self._stats, field) + 1)


__all__ = ["RuntimeRiskDispatcher", "RuntimeRiskStats"]
