# SPDX-License-Identifier: Apache-2.0

"""Tests for full-depth ScoutRank/main-model overlap plumbing."""

# Standard
from types import SimpleNamespace
import asyncio
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.makv.config import IMPORTANCE_REQUEST_KEY
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS
from lmcache.v1.storage_backend.makv.scout_overlap import (
    ScoutOverlapResult,
    resolve_scout_importance,
    submit_scout_if_needed,
)
from lmcache.v1.storage_backend.makv_remote.scout_protocol import (
    SCOUT_PROTOCOL_VERSION,
    decode_scores,
    decode_token_ids,
    encode_scores,
    encode_token_ids,
    payload_sha256,
)
from lmcache.v1.storage_backend.makv_remote.scout_service import ScoutJobService
from lmcache.v1.storage_backend.makv_remote.server import MaKVRemoteServer
from lmcache.v1.cache_engine import LMCacheEngine


class _BlockingRuntime:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def score_token_ids(self, token_ids: list[int]) -> list[float]:
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test did not release ScoutRank runtime")
        return [float(value) / 10.0 for value in token_ids]


def _config(serde: str = "makv") -> SimpleNamespace:
    return SimpleNamespace(
        remote_serde=serde,
        remote_url="makv://127.0.0.1:65432",
        chunk_size=256,
        save_unfull_chunk=False,
        extra_config={
            "makv_scout_overlap_enabled": True,
            "makv_scout_timeout_s": 1.0,
        },
    )


def test_scout_binary_payload_round_trip() -> None:
    token_ids = [0, 1, 151935, 0xFFFFFFFF]
    token_payload = encode_token_ids(token_ids)
    assert decode_token_ids(token_payload, len(token_ids)) == token_ids
    assert payload_sha256(token_payload) == payload_sha256(token_payload)

    scores = [-1.25, 0.0, 0.5, 100.0]
    restored = decode_scores(encode_scores(scores), len(scores))
    assert restored == pytest.approx(scores)
    with pytest.raises(ValueError, match="token_count"):
        decode_scores(encode_scores(scores), len(scores) - 1)


def test_scout_service_overlaps_and_deduplicates() -> None:
    async def run() -> None:
        runtime = _BlockingRuntime()
        service = ScoutJobService(runtime, max_pending_jobs=1, result_ttl_s=10)
        token_ids = [4, 5, 6]
        checksum = payload_sha256(encode_token_ids(token_ids))
        submit_started = time.perf_counter()
        accepted = service.submit("req-1", token_ids, checksum)
        assert (time.perf_counter() - submit_started) < 0.1
        assert accepted == {"accepted": True, "deduplicated": False}
        assert service.submit("req-1", token_ids, checksum)["deduplicated"]
        with pytest.raises(ValueError, match="reused"):
            service.submit("req-1", [7], payload_sha256(encode_token_ids([7])))
        with pytest.raises(asyncio.QueueFull):
            service.submit("req-2", [8], payload_sha256(encode_token_ids([8])))

        assert runtime.started.wait(timeout=1)
        await asyncio.sleep(0.02)
        runtime.release.set()
        result, timing = await service.wait("req-1", len(token_ids), 1.0)
        assert result.scores == pytest.approx([0.4, 0.5, 0.6])
        assert timing["overlap_hidden_time_ms"] > 0
        assert runtime.calls == 1
        assert service.health()["metrics"]["completed_jobs"] == 1
        service.close()

    asyncio.run(run())


def test_manager_dispatches_scout_protocol() -> None:
    class Runtime:
        def score_token_ids(self, token_ids: list[int]) -> list[float]:
            return [float(value) for value in token_ids]

    class Manager:
        async def health(self):
            return {"pid": 1, "metrics": {}}

    async def run() -> None:
        jobs = ScoutJobService(Runtime())
        server = MaKVRemoteServer(
            Manager(),
            queue_depth=1,
            workers=1,
            max_request_bytes=1024,
            scout_service=jobs,
        )
        token_ids = [11, 12, 13]
        payload = encode_token_ids(token_ids)
        submit_header = {
            "protocol_version": SCOUT_PROTOCOL_VERSION,
            "token_count": len(token_ids),
            "token_sha256": payload_sha256(payload),
        }
        response, response_payload = await server._dispatch(
            "SCOUT_SUBMIT", "req-wire", payload, submit_header
        )
        assert response["accepted"]
        assert response_payload == b""
        wait_header = {
            "protocol_version": SCOUT_PROTOCOL_VERSION,
            "token_count": len(token_ids),
            "timeout_s": 1.0,
        }
        response, response_payload = await server._dispatch(
            "SCOUT_WAIT", "req-wire", b"", wait_header
        )
        assert decode_scores(response_payload, len(token_ids)) == pytest.approx(
            token_ids
        )
        assert response["score_time_ms"] >= 0
        health, _ = await server._dispatch("HEALTH", "", b"")
        assert health["scout"]["enabled"] is True
        await server.close()

    asyncio.run(run())


def test_submit_and_join_are_makv_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeClient:
        def submit(self, request_id, token_ids):
            calls.append(("submit", (request_id, list(token_ids))))
            return {"accepted": True}

        def wait(self, request_id, token_count, *, deferred=False):
            assert deferred is False
            calls.append(("wait", (request_id, token_count)))
            return ScoutOverlapResult(
                scores=[0.1, 0.2, 0.3],
                score_time_ms=40.0,
                total_job_time_ms=42.0,
                wait_time_ms=10.0,
                overlap_hidden_time_ms=32.0,
            )

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: FakeClient(),
    )
    CLIENT_METRICS.reset()
    config = _config()
    config.save_unfull_chunk = True
    assert submit_scout_if_needed(
        config,
        request_id="req-3",
        token_ids=[1, 2, 3],
        request_configs=None,
        cached_tokens=0,
    )
    resolved = resolve_scout_importance(
        _config(),
        request_id="req-3",
        token_count=3,
        request_configs={"unrelated": True},
    )
    assert resolved is not None
    assert resolved[IMPORTANCE_REQUEST_KEY] == pytest.approx([0.1, 0.2, 0.3])
    assert resolved["unrelated"] is True
    cached_resolved = resolve_scout_importance(
        config,
        request_id="req-3",
        token_count=3,
        request_configs=None,
    )
    assert cached_resolved is not None
    assert cached_resolved[IMPORTANCE_REQUEST_KEY] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert [name for name, _ in calls] == ["submit", "wait"]
    metrics = CLIENT_METRICS.snapshot()
    assert metrics.makv_scout_submit_calls == 1
    assert metrics.makv_scout_wait_calls == 1
    assert metrics.makv_scout_overlap_hidden_time_ms == 32.0

    calls.clear()
    assert not submit_scout_if_needed(
        _config("cachegen"),
        request_id="req-native",
        token_ids=[1, 2, 3],
        request_configs=None,
        cached_tokens=0,
    )
    assert calls == []


def test_explicit_importance_bypasses_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: pytest.fail("overlap client must not be created"),
    )
    request_configs = {IMPORTANCE_REQUEST_KEY: [0.5, 0.4]}
    assert not submit_scout_if_needed(
        _config(),
        request_id="req-explicit",
        token_ids=[1, 2],
        request_configs=request_configs,
        cached_tokens=0,
    )
    assert (
        resolve_scout_importance(
            _config(),
            request_id="req-explicit",
            token_count=2,
            request_configs=request_configs,
        )
        is request_configs
    )


def test_complete_aligned_hit_does_not_submit_tail_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: pytest.fail("aligned complete hit must not submit"),
    )
    assert not submit_scout_if_needed(
        _config(),
        request_id="req-hit",
        token_ids=list(range(279)),
        request_configs=None,
        cached_tokens=256,
    )


def test_deferred_store_joins_before_batched_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Memory:
        refs = 1

        def ref_count_down(self):
            self.refs -= 1

    class StorageManager:
        def batched_put(self, keys, memory_objs, *, transfer_spec, location):
            events.append("put")
            assert keys == ["key"]
            assert transfer_spec["request_configs"][IMPORTANCE_REQUEST_KEY] == [
                0.1,
                0.2,
            ]
            assert transfer_spec["chunk_starts"] == [0]
            assert transfer_spec["chunk_ends"] == [2]
            assert location == "RemoteBackend"
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

    def resolve(*args, **kwargs):
        assert kwargs["deferred"] is True
        events.append("join")
        return {IMPORTANCE_REQUEST_KEY: [0.1, 0.2]}

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap.resolve_scout_importance",
        resolve,
    )
    engine = object.__new__(LMCacheEngine)
    engine.config = _config()
    engine.storage_manager = StorageManager()
    engine.store_location = "RemoteBackend"
    memory = Memory()
    engine._deferred_makv_put(
        ["key"],
        [memory],
        [0],
        [2],
        None,
        None,
        2,
        [1, 2],
        "req-deferred",
    )
    assert events == ["join", "put"]
    assert memory.refs == 0
