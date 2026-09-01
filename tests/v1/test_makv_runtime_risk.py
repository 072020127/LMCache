# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in runtime CONF-to-MaKV bridge."""

from types import SimpleNamespace

import pytest
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.storage_backend.makv.runtime_risk import RuntimeRiskDispatcher
from vllm.v1.worker.makv_runtime_risk import (
    RuntimeRiskRequestContext,
    submit_makv_runtime_risk,
    submit_makv_runtime_risk_v2,
)


def test_runtime_dispatcher_scores_actual_logits_and_keeps_absolute_position() -> None:
    observed: list[tuple[object, ...]] = []
    dispatcher = RuntimeRiskDispatcher(
        lambda request_id, tokens, token_index, signal, configs: observed.append(
            (request_id, tuple(tokens), token_index, signal, configs)
        )
        or {"accepted": True},
        max_queue=2,
        window_tokens=16,
    )
    try:
        assert dispatcher.submit(
            "request-1",
            [10, 11, 12],
            2,
            torch.tensor([3.0, 1.0, -1.0]),
            step=7,
            request_configs={"lmcache.tag.case": "runtime"},
        )
        assert dispatcher.wait_idle(5.0)
        assert len(observed) == 1
        signal = observed[0][3]
        assert observed[0][0:3] == ("request-1", (10, 11, 12), 2)
        assert signal.step == 7
        assert signal.token_index == 2
        assert signal.window_tokens == 16
        assert observed[0][4] == {"lmcache.tag.case": "runtime"}
        assert dispatcher.stats() == {
            "submitted": 1,
            "dropped": 0,
            "invalid": 0,
            "scored": 1,
            "accepted": 1,
            "failed": 0,
            "queue_size": 0,
        }
    finally:
        dispatcher.close()


def test_runtime_dispatcher_rejects_position_without_step_fallback() -> None:
    dispatcher = RuntimeRiskDispatcher(lambda *args: {"accepted": True})
    try:
        assert not dispatcher.submit(
            "request-1",
            [1, 2],
            2,
            torch.zeros(4),
            step=0,
        )
        assert dispatcher.stats()["invalid"] == 1
    finally:
        dispatcher.close()


def test_vllm_hook_requires_explicit_positions_and_skips_prefill() -> None:
    calls: list[tuple[object, ...]] = []

    class Connector:
        def submit_precision_risk(self, *args: object) -> bool:
            calls.append(args)
            return True

    request = SimpleNamespace(
        prompt_token_ids=[4, 5, 6, 7],
        num_computed_tokens=4,
        kv_transfer_params={
            "lmcache.makv.risk_token_indices": [3],
            "makv_risk_observer_enabled": True,
        },
    )
    assert (
        submit_makv_runtime_risk(
            Connector(),
            torch.zeros((1, 8)),
            ["request-1"],
            {"request-1": request},
            {"request-1": 1},
        )
        == 1
    )
    assert calls[0][0] == "request-1"
    assert calls[0][3] == 3
    assert calls[0][5] is request.kv_transfer_params

    request.kv_transfer_params["makv_risk_observer_enabled"] = False
    assert (
        submit_makv_runtime_risk(
            Connector(),
            torch.zeros((1, 8)),
            ["request-1"],
            {"request-1": request},
            {"request-1": 1},
        )
        == 0
    )
    request.kv_transfer_params["makv_risk_observer_enabled"] = True
    assert (
        submit_makv_runtime_risk(
            Connector(),
            torch.zeros((1, 8)),
            ["request-1"],
            {"request-1": request},
            {"request-1": 2},
        )
        == 0
    )


def test_vllm_hook_forwards_request_params_for_tracker_fallback() -> None:
    calls: list[tuple[object, ...]] = []

    class Connector:
        def submit_precision_risk(self, *args: object) -> bool:
            calls.append(args)
            return True

    request = SimpleNamespace(
        prompt_token_ids=[9, 8],
        num_computed_tokens=2,
        kv_transfer_params={
            "lmcache.makv.risk_token_indices": [1],
            "makv_risk_observer_enabled": True,
        },
    )
    assert (
        submit_makv_runtime_risk(
            Connector(),
            torch.zeros((1, 8)),
            ["request-2"],
            {"request-2": request},
            {"request-2": 1},
        )
        == 1
    )
    assert calls[0][5] is request.kv_transfer_params


def test_vllm_v2_hook_uses_decode_logits_and_absolute_position() -> None:
    calls: list[tuple[object, ...]] = []

    class Connector:
        def submit_precision_risk(self, *args: object) -> bool:
            calls.append(args)
            return True

    params = {
        "lmcache.makv.risk_token_indices": [1, 5],
        "makv_risk_observer_enabled": True,
    }
    contexts = {
        "request-v2": RuntimeRiskRequestContext(
            prompt_token_ids=(20, 21, 22, 23, 24, 25),
            request_params=params,
        )
    }
    assert (
        submit_makv_runtime_risk_v2(
            Connector(),
            torch.zeros((1, 8)),
            ["request-v2"],
            [1],
            [6],
            [6],
            contexts,
        )
        == 1
    )
    assert calls[0][2] == 0
    assert calls[0][3] == 1
    assert calls[0][5] is params


@pytest.mark.parametrize(
    "scheduled_tokens, computed_tokens, positions",
    [([2], [6], [1]), ([1], [5], [1]), ([1], [6], None)],
)
def test_vllm_v2_hook_fail_closes_invalid_decode_observations(
    scheduled_tokens: list[int], computed_tokens: list[int], positions: list[int] | None
) -> None:
    calls: list[tuple[object, ...]] = []

    class Connector:
        def submit_precision_risk(self, *args: object) -> bool:
            calls.append(args)
            return True

    params = {"makv_risk_observer_enabled": True}
    if positions is not None:
        params["lmcache.makv.risk_token_indices"] = positions
    context = RuntimeRiskRequestContext((20, 21, 22, 23, 24, 25), params)
    assert (
        submit_makv_runtime_risk_v2(
            Connector(),
            torch.zeros((1, 8)),
            ["request-v2"],
            scheduled_tokens,
            computed_tokens,
            [6],
            {"request-v2": context},
        )
        == 0
    )
    assert calls == []


def test_engine_routes_risk_using_token_database_chunk_key() -> None:
    key = CacheEngineKey("model", 1, 0, 123, torch.float16, {})
    reports: list[tuple[object, object]] = []

    class TokenDatabase:
        def process_tokens(self, *, tokens, request_configs):
            assert list(tokens) == [1, 2, 3, 4]
            assert request_configs == {"lmcache.tag.case": "runtime"}
            yield 0, 4, key

    class Backend:
        def report_precision_risk(self, target_key, signal):
            reports.append((target_key, signal))
            return {"accepted": True, "window_active": True}

    class StorageManager:
        storage_backends = {"RemoteBackend": Backend()}

        def get_non_allocator_backends(self):
            return ["RemoteBackend"]

    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(remote_serde="makv")
    engine.storage_manager = StorageManager()
    engine.token_database = TokenDatabase()

    response = engine.report_precision_risk(
        [1, 2, 3, 4],
        2,
        {"risk": 0.9},
        request_configs={"lmcache.tag.case": "runtime"},
    )
    assert response == {"accepted": True, "window_active": True}
    assert reports == [(key, {"risk": 0.9})]

    assert engine.report_precision_risk([1, 2, 3, 4], 4, {}) == {
        "accepted": False,
        "reason": "token_index_out_of_range",
    }


@pytest.mark.parametrize("remote_serde", ["naive", "cachegen"])
def test_engine_runtime_risk_is_inert_for_native_serdes(remote_serde: str) -> None:
    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace(remote_serde=remote_serde)
    assert engine.report_precision_risk([1], 0, {}) == {
        "accepted": False,
        "reason": "remote_serde_is_not_makv",
    }
