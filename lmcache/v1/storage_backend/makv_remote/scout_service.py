# SPDX-License-Identifier: Apache-2.0

"""Bounded asynchronous ScoutRank jobs hosted by the MaKV manager."""

from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol
import asyncio
import time


class ScoutRuntime(Protocol):
    """Minimal interface implemented by the reusable 28-layer scorer."""

    def score_token_ids(self, token_ids: list[int]) -> list[float]: ...


@dataclass(frozen=True)
class ScoutScoreResult:
    scores: list[float]
    queue_time_ms: float
    score_time_ms: float
    total_time_ms: float


@dataclass
class ScoutJob:
    request_id: str
    token_count: int
    token_sha256: str
    submitted_at: float
    future: Future[ScoutScoreResult]
    metrics_recorded: bool = False


@dataclass
class ScoutServiceMetrics:
    submitted_jobs: int = 0
    deduplicated_submissions: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    rejected_jobs: int = 0
    wait_calls: int = 0
    queue_time_ms: float = 0.0
    score_time_ms: float = 0.0
    total_job_time_ms: float = 0.0
    exposed_wait_time_ms: float = 0.0
    background_wait_time_ms: float = 0.0
    overlap_hidden_time_ms: float = 0.0


class ScoutJobService:
    """Run one persistent Scout model behind a bounded, idempotent queue."""

    def __init__(
        self,
        runtime: ScoutRuntime,
        *,
        max_pending_jobs: int = 64,
        result_ttl_s: float = 600.0,
    ) -> None:
        if max_pending_jobs <= 0 or result_ttl_s <= 0:
            raise ValueError("Scout queue depth and result TTL must be positive")
        self.runtime = runtime
        self.max_pending_jobs = max_pending_jobs
        self.result_ttl_s = result_ttl_s
        # A single worker preserves model state safety and predictable GPU use.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="makv-scout"
        )
        self._jobs: dict[str, ScoutJob] = {}
        self.metrics = ScoutServiceMetrics()

    def _cleanup(self, now: float) -> None:
        expired = [
            request_id
            for request_id, job in self._jobs.items()
            if job.future.done() and now - job.submitted_at >= self.result_ttl_s
        ]
        for request_id in expired:
            del self._jobs[request_id]

    def _run(
        self,
        token_ids: list[int],
        submitted_at: float,
    ) -> ScoutScoreResult:
        score_started = time.perf_counter()
        scores = self.runtime.score_token_ids(token_ids)
        completed_at = time.perf_counter()
        if len(scores) != len(token_ids):
            raise ValueError("ScoutRank score count does not match prompt token count")
        return ScoutScoreResult(
            scores=scores,
            queue_time_ms=(score_started - submitted_at) * 1000.0,
            score_time_ms=(completed_at - score_started) * 1000.0,
            total_time_ms=(completed_at - submitted_at) * 1000.0,
        )

    def submit(
        self,
        request_id: str,
        token_ids: list[int],
        token_sha256: str,
    ) -> dict[str, Any]:
        """Enqueue a request, returning immediately after executor submission."""
        if not request_id:
            raise ValueError("ScoutRank submission requires request_id")
        now = time.perf_counter()
        self._cleanup(now)
        existing = self._jobs.get(request_id)
        if existing is not None:
            if (
                existing.token_count != len(token_ids)
                or existing.token_sha256 != token_sha256
            ):
                raise ValueError("ScoutRank request_id was reused for another prompt")
            self.metrics.deduplicated_submissions += 1
            return {"accepted": True, "deduplicated": True}
        pending = sum(not job.future.done() for job in self._jobs.values())
        if pending >= self.max_pending_jobs:
            self.metrics.rejected_jobs += 1
            raise asyncio.QueueFull
        future = self._executor.submit(self._run, token_ids, now)
        self._jobs[request_id] = ScoutJob(
            request_id=request_id,
            token_count=len(token_ids),
            token_sha256=token_sha256,
            submitted_at=now,
            future=future,
        )
        self.metrics.submitted_jobs += 1
        return {"accepted": True, "deduplicated": False}

    async def wait(
        self,
        request_id: str,
        token_count: int,
        timeout_s: float,
        deferred: bool = False,
    ) -> tuple[ScoutScoreResult, dict[str, float]]:
        """Wait only at the MaKV store dependency boundary."""
        if timeout_s <= 0:
            raise ValueError("ScoutRank wait timeout must be positive")
        job = self._jobs.get(request_id)
        if job is None:
            raise KeyError(f"ScoutRank job not found for request {request_id!r}")
        if job.token_count != token_count:
            raise ValueError("ScoutRank wait token_count does not match submission")
        wait_started = time.perf_counter()
        self.metrics.wait_calls += 1
        try:
            deadline = wait_started + timeout_s
            while not job.future.done():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("ScoutRank job timed out")
                # Polling a concurrent Future avoids depending on an asyncio
                # cross-thread callback, which is unreliable in some serving
                # launchers. The event loop remains free to serve storage I/O.
                await asyncio.sleep(min(0.002, remaining))
            result = job.future.result()
        except Exception:
            if job.future.done() and not job.future.cancelled():
                self.metrics.failed_jobs += 1
            raise
        wait_ms = (time.perf_counter() - wait_started) * 1000.0
        hidden_ms = (
            result.total_time_ms
            if deferred
            else max(0.0, result.total_time_ms - wait_ms)
        )
        if deferred:
            self.metrics.background_wait_time_ms += wait_ms
        else:
            self.metrics.exposed_wait_time_ms += wait_ms
        if not job.metrics_recorded:
            self.metrics.completed_jobs += 1
            self.metrics.queue_time_ms += result.queue_time_ms
            self.metrics.score_time_ms += result.score_time_ms
            self.metrics.total_job_time_ms += result.total_time_ms
            self.metrics.overlap_hidden_time_ms += hidden_ms
            job.metrics_recorded = True
        return result, {
            "wait_time_ms": wait_ms,
            "overlap_hidden_time_ms": hidden_ms,
        }

    def health(self) -> dict[str, Any]:
        """Return queue state and cumulative overlap timings."""
        now = time.perf_counter()
        self._cleanup(now)
        return {
            "enabled": True,
            "pending_jobs": sum(
                not job.future.done() for job in self._jobs.values()
            ),
            "retained_jobs": len(self._jobs),
            "queue_depth": self.max_pending_jobs,
            "metrics": dict(self.metrics.__dict__),
        }

    def close(self) -> None:
        """Finish submitted work and release the executor."""
        self._executor.shutdown(wait=True, cancel_futures=False)
