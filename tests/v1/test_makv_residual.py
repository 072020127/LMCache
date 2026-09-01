# SPDX-License-Identifier: Apache-2.0

"""Tests for optional MaKV quantization residuals and risk upgrades."""

# Standard
import asyncio
import json
from dataclasses import replace
import os
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.storage_backend.makv.config import (
    MaKVConfig,
    get_makv_config,
    validate_makv_runtime_config,
)
from lmcache.v1.storage_backend.makv.entropy import encode_entropy_payloads
from lmcache.v1.storage_backend.makv.format import (
    decode_makv_object,
    encode_client_put_envelope,
)
from lmcache.v1.storage_backend.makv.precision_risk import (
    CONF_RISK_SEMANTICS,
    CONF_SCORER_VERSION,
)
from lmcache.v1.storage_backend.makv.quantizer import quantize_canonical_kv
from lmcache.v1.storage_backend.makv.residual import (
    reconstruct_with_residual,
    validate_residual_metadata,
)
from lmcache.v1.storage_backend.makv.plan import MaKVQuantPlan
from lmcache.v1.storage_backend.makv_remote.manager import MaKVRemoteManager
from lmcache.v1.storage_backend.connector.makv_network_connector import (
    MaKVNetworkConnector,
)
from lmcache.v1.storage_backend.makv_remote.protocol import read_frame, write_frame
from lmcache.v1.storage_backend.makv_remote.server import MaKVRemoteServer


class _MemoryStorage:
    """Small async storage double with complete-value replacement semantics."""

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


def _config(
    *,
    residual_dtype: str,
    bucket_bits: tuple[int, ...] = (16, 8, 4),
    risk_window_tokens: int = 16,
) -> MaKVConfig:
    return MaKVConfig(
        storage_url="file:///tmp/makv-residual-test",
        bucket_ratios=tuple(1.0 / len(bucket_bits) for _ in bucket_bits),
        bucket_bits=bucket_bits,
        importance_layout="token",
        quant_granularity="per_token_head",
        scale_dtype="float16",
        protect_prefix_tokens=0,
        protect_tail_tokens=0,
        dequant_backend="reference",
        require_cuda_dequant=False,
        fallback="miss",
        enable_checksum=True,
        residual_dtype=residual_dtype,
        risk_upgrade_threshold=0.8,
        risk_upgrade_policy="next",
        risk_window_tokens=risk_window_tokens,
    )


def _plan(
    *,
    layout: str,
    dtype: torch.dtype,
    bucket_ids: list[int],
    bucket_bits: tuple[int, ...] = (16, 8, 4),
    layers: int = 1,
    tokens: int = 6,
    heads: int = 2,
    head_dim: int = 3,
) -> MaKVQuantPlan:
    return MaKVQuantPlan(
        protocol_version=1,
        importance_layout=layout,
        token_count=tokens,
        chunk_start=0,
        chunk_length=tokens,
        bucket_bits=bucket_bits,
        bucket_ids=bytes(bucket_ids),
        original_shape=(2, layers, tokens, heads * head_dim),
        original_strides=(
            layers * tokens * heads * head_dim,
            tokens * heads * head_dim,
            heads * head_dim,
            1,
        ),
        original_dtype=str(dtype),
        token_dim=2,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        quant_granularity="per_token_head",
        scale_dtype="float16",
        model_fingerprint="residual-test-model",
        parallel_fingerprint="residual-test-worker",
        checksum=0,
    )


def _source(
    *, layers: int, tokens: int, heads: int, head_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    values = torch.arange(
        layers * 2 * tokens * heads * head_dim, dtype=torch.float32
    )
    return (values.reshape(layers, 2, tokens, heads, head_dim) / 17).to(dtype)


def _wire_tensor(source: torch.Tensor) -> torch.Tensor:
    return source.permute(1, 0, 2, 3, 4).reshape(
        2, source.shape[0], source.shape[2], source.shape[3] * source.shape[4]
    ).contiguous()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def test_residual_config_is_optional_and_validated():
    config = SimpleNamespace(
        remote_serde="makv",
        remote_url="makv://127.0.0.1:65432",
        extra_config={
            "makv_storage_url": "file:///tmp/makv-residual-config",
            "makv_bucket_ratios": [0.25, 0.25, 0.5],
            "makv_bucket_bits": [16, 8, 4],
            "makv_require_cuda_dequant": False,
            "makv_residual_dtype": "float16",
            "makv_risk_upgrade_threshold": 0.75,
            "makv_risk_upgrade_policy": "full",
            "makv_risk_window_tokens": 7,
            "makv_risk_window_ttl_s": 1.5,
        },
    )
    runtime = get_makv_config(config)
    assert runtime.residual_dtype == "float16"
    assert runtime.risk_upgrade_threshold == pytest.approx(0.75)
    assert runtime.risk_upgrade_policy == "full"
    assert runtime.risk_window_tokens == 7
    assert runtime.risk_window_ttl_s == pytest.approx(1.5)

    invalid = SimpleNamespace(
        remote_serde="makv",
        extra_config={
            "makv_bucket_ratios": [0.5, 0.5],
            "makv_bucket_bits": [16, 8],
            "makv_require_cuda_dequant": False,
            "makv_residual_dtype": "int8",
        },
    )
    with pytest.raises(ValueError, match="makv_residual_dtype"):
        validate_makv_runtime_config(invalid)

    invalid.extra_config["makv_residual_dtype"] = "none"
    invalid.extra_config["makv_risk_window_tokens"] = 0
    with pytest.raises(ValueError, match="makv_risk_window_tokens"):
        validate_makv_runtime_config(invalid)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_residual_roundtrip_token_layout(dtype: torch.dtype):
    source = _source(layers=1, tokens=6, heads=2, head_dim=3, dtype=dtype)
    plan = _plan(
        layout="token",
        dtype=dtype,
        bucket_ids=[0, 1, 2, 1, 2, 0],
    )
    metadata, payloads = quantize_canonical_kv(
        source, plan, _config(residual_dtype="float32")
    )

    validate_residual_metadata(metadata, payloads)
    assert metadata["residual"]["dtype"] == "float32"
    assert {entry["bits"] for entry in metadata["residual"]["buckets"]} == {
        8,
        4,
    }
    reconstructed = reconstruct_with_residual(metadata, payloads)
    torch.testing.assert_close(
        reconstructed,
        _wire_tensor(source),
        rtol=0.0,
        atol=0.0,
    )


def test_residual_roundtrip_layer_kv_layout_and_legacy_default():
    layers, tokens, heads, head_dim = 2, 5, 1, 3
    source = _source(
        layers=layers,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        dtype=torch.float16,
    )
    # Flat order is [layer, kv, token].
    ids = [
        0,
        1,
        2,
        1,
        2,
        0,
        1,
        2,
        1,
        2,
        0,
        1,
        2,
        1,
        2,
        0,
        1,
        2,
        1,
        2,
    ]
    plan = _plan(
        layout="layer_kv_token",
        dtype=torch.float16,
        bucket_ids=ids,
        layers=layers,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
    )
    residual_metadata, residual_payloads = quantize_canonical_kv(
        source, plan, _config(residual_dtype="float16")
    )
    validate_residual_metadata(residual_metadata, residual_payloads)
    reconstructed = reconstruct_with_residual(residual_metadata, residual_payloads)
    torch.testing.assert_close(
        reconstructed,
        _wire_tensor(source),
        rtol=0.0,
        atol=2e-3,
    )

    legacy_metadata, legacy_payloads = quantize_canonical_kv(
        source, plan, _config(residual_dtype="none")
    )
    assert "residual" not in legacy_metadata
    assert not any(name.startswith("residual_") for name in legacy_payloads)


def test_residual_roundtrip_int2_with_odd_head_dimension():
    source = _source(layers=1, tokens=7, heads=1, head_dim=3, dtype=torch.float16)
    plan = _plan(
        layout="token",
        dtype=torch.float16,
        bucket_ids=[0, 1, 2, 3, 3, 1, 0],
        bucket_bits=(16, 8, 4, 2),
        tokens=7,
        heads=1,
        head_dim=3,
    )
    metadata, payloads = quantize_canonical_kv(
        source,
        plan,
        _config(residual_dtype="float32", bucket_bits=(16, 8, 4, 2)),
    )
    validate_residual_metadata(metadata, payloads)
    reconstructed = reconstruct_with_residual(metadata, payloads)
    torch.testing.assert_close(
        reconstructed,
        _wire_tensor(source),
        rtol=0.0,
        atol=0.0,
    )


def test_residual_survives_cachegen_arithmetic_envelope():
    source = _source(layers=1, tokens=6, heads=2, head_dim=3, dtype=torch.float16)
    plan = _plan(
        layout="token",
        dtype=torch.float16,
        bucket_ids=[0, 1, 2, 1, 2, 0],
    )
    config = replace(
        _config(residual_dtype="float32"),
        entropy_codec="cachegen_arithmetic",
        entropy_backend="reference",
    )
    metadata, payloads = quantize_canonical_kv(source, plan, config)
    metadata, payloads = encode_entropy_payloads(
        metadata,
        payloads,
        codec=config.entropy_codec,
        backend=config.entropy_backend,
    )
    validate_residual_metadata(metadata, payloads)
    reconstructed = reconstruct_with_residual(metadata, payloads)
    torch.testing.assert_close(
        reconstructed,
        _wire_tensor(source),
        rtol=0.0,
        atol=0.0,
    )


def _risk(
    step: int,
    value: float,
    *,
    token_index: int | None = None,
    window_tokens: int | None = None,
) -> dict[str, object]:
    signal: dict[str, object] = {
        "step": step,
        "risk": value,
        "scorer_version": CONF_SCORER_VERSION,
        "semantics": CONF_RISK_SEMANTICS,
        "valid": True,
    }
    if token_index is not None:
        signal["token_index"] = token_index
    if window_tokens is not None:
        signal["window_tokens"] = window_tokens
    return signal


def test_precision_signal_can_carry_a_kv_token_without_changing_default_wire():
    from lmcache.v1.storage_backend.makv.precision_risk import (
        compute_precision_risk_signal,
    )

    signal = compute_precision_risk_signal(torch.zeros(4), step=3)
    assert set(signal.as_dict()) == {
        "step",
        "risk",
        "scorer_version",
        "semantics",
        "valid",
    }
    positioned = signal.for_kv_token(19, window_tokens=5)
    assert positioned.as_dict()["token_index"] == 19
    assert positioned.as_dict()["window_tokens"] == 5


def test_manager_promotes_risk_token_for_a_window_and_restores_base():
    async def run() -> None:
        key = "residual-key"
        storage = _MemoryStorage()
        manager = MaKVRemoteManager(
            _config(residual_dtype="float32", risk_window_tokens=2), storage
        )
        source = _source(layers=1, tokens=6, heads=2, head_dim=3, dtype=torch.float16)
        plan = _plan(
            layout="token",
            dtype=torch.float16,
            bucket_ids=[0, 1, 2, 1, 2, 0],
        )
        envelope = encode_client_put_envelope(
            key=key,
            object_type="raw_with_plan",
            plan=plan,
            raw_kv_payload=_tensor_bytes(_wire_tensor(source)),
        )
        assert await manager.put(key, envelope, 0.0) > 0
        assert manager.quantize_calls == 1
        before = storage.values[key]
        before_object = decode_makv_object(before)
        assert "residual" in before_object.metadata
        public_before = await manager.get(key)
        assert public_before is not None
        public_before_object = decode_makv_object(public_before)
        assert "residual" not in public_before_object.metadata
        assert not any(
            name.startswith("residual_") for name in public_before_object.payloads
        )
        assert public_before_object.payloads
        assert public_before != before
        batch_before, _ = (await manager.get_many_with_timing([key]))[0]
        assert batch_before is not None
        assert "residual" not in decode_makv_object(batch_before).metadata

        below = await manager.apply_precision_risk(key, _risk(0, 0.5))
        assert below["reason"] == "below_threshold"
        assert manager.quantize_calls == 1
        assert storage.values[key] == before

        promoted = await manager.apply_precision_risk(key, _risk(1, 0.95))
        assert promoted["upgraded"] is True
        assert manager.quantize_calls == 2
        assert storage.values[key] == before
        active = await manager.get(key)
        assert active is not None
        active_object = decode_makv_object(active)
        assert "residual" not in active_object.metadata
        assert list(active_object.metadata["plan"]["bucket_ids"]) == [
            0,
            0,
            2,
            1,
            2,
            0,
        ]

        stale = await manager.apply_precision_risk(key, _risk(1, 0.99))
        assert stale["reason"] == "stale_signal"
        assert manager.quantize_calls == 2
        assert storage.values[key] == before

        second = await manager.apply_precision_risk(key, _risk(2, 0.99))
        assert second["upgraded"] is True
        assert manager.quantize_calls == 3
        active_two = await manager.get(key)
        assert active_two is not None
        active_two_object = decode_makv_object(active_two)
        assert list(active_two_object.metadata["plan"]["bucket_ids"]) == [
            0,
            0,
            1,
            1,
            2,
            0,
        ]
        assert storage.values[key] == before

        expired = await manager.apply_precision_risk(key, _risk(4, 0.5))
        assert expired["reason"] == "below_threshold"
        assert expired["window_expired"] is True
        restored = await manager.get(key)
        assert restored is not None
        restored_object = decode_makv_object(restored)
        assert list(restored_object.metadata["plan"]["bucket_ids"]) == [
            0,
            1,
            2,
            1,
            2,
            0,
        ]
        assert "residual" not in restored_object.metadata
        assert storage.values[key] == before

        with pytest.raises(ValueError, match="scorer version"):
            await manager.apply_precision_risk(
                key,
                {
                    **_risk(3, 0.99),
                    "scorer_version": "unknown",
                },
            )
        await manager.close()

    asyncio.run(run())


def test_precision_window_can_expire_by_wall_clock_without_a_new_signal():
    async def run() -> None:
        key = "ttl-window-key"
        storage = _MemoryStorage()
        manager = MaKVRemoteManager(
            _config(residual_dtype="float32"), storage
        )
        manager.config = replace(
            manager.config,
            risk_window_ttl_s=0.001,
            risk_window_tokens=100,
        )
        source = _source(layers=1, tokens=6, heads=2, head_dim=3, dtype=torch.float16)
        plan = _plan(
            layout="token",
            dtype=torch.float16,
            bucket_ids=[0, 1, 2, 1, 2, 0],
        )
        envelope = encode_client_put_envelope(
            key=key,
            object_type="raw_with_plan",
            plan=plan,
            raw_kv_payload=_tensor_bytes(_wire_tensor(source)),
        )
        await manager.put(key, envelope, 0.0)
        promoted = await manager.apply_precision_risk(key, _risk(1, 0.95))
        assert promoted["upgraded"] is True
        assert await manager.get(key) is not None
        await asyncio.sleep(0.01)
        restored = await manager.get(key)
        assert restored is not None
        assert list(decode_makv_object(restored).metadata["plan"]["bucket_ids"]) == [
            0,
            1,
            2,
            1,
            2,
            0,
        ]
        await manager.close()

    asyncio.run(run())


def test_layer_kv_token_window_promotes_both_planes_for_all_layers():
    async def run() -> None:
        key = "layer-window-key"
        storage = _MemoryStorage()
        manager = MaKVRemoteManager(
            _config(residual_dtype="float32", risk_window_tokens=4), storage
        )
        source = _source(layers=2, tokens=5, heads=1, head_dim=3, dtype=torch.float16)
        plan = _plan(
            layout="layer_kv_token",
            dtype=torch.float16,
            bucket_ids=(
                [0, 1, 2, 1, 2]
                + [0, 1, 2, 1, 2]
                + [0, 1, 2, 1, 2]
                + [0, 1, 2, 1, 2]
            ),
            layers=2,
            tokens=5,
            heads=1,
            head_dim=3,
        )
        envelope = encode_client_put_envelope(
            key=key,
            object_type="raw_with_plan",
            plan=plan,
            raw_kv_payload=_tensor_bytes(_wire_tensor(source)),
        )
        await manager.put(key, envelope, 0.0)
        result = await manager.apply_precision_risk(
            key, _risk(2, 0.95, token_index=3)
        )
        assert result["upgraded"] is True
        public = await manager.get(key)
        assert public is not None
        bucket_ids = list(decode_makv_object(public).metadata["plan"]["bucket_ids"])
        assert bucket_ids == [
            0,
            1,
            2,
            0,
            2,
            0,
            1,
            2,
            0,
            2,
            0,
            1,
            2,
            0,
            2,
            0,
            1,
            2,
            0,
            2,
        ]
        await manager.close()

    asyncio.run(run())


def test_remote_server_dispatches_precision_risk_request():
    async def run() -> None:
        key = "server-risk-key"
        storage = _MemoryStorage()
        manager = MaKVRemoteManager(_config(residual_dtype="float32"), storage)
        source = _source(layers=1, tokens=6, heads=2, head_dim=3, dtype=torch.float16)
        plan = _plan(
            layout="token",
            dtype=torch.float16,
            bucket_ids=[0, 1, 2, 1, 2, 0],
        )
        envelope = encode_client_put_envelope(
            key=key,
            object_type="raw_with_plan",
            plan=plan,
            raw_kv_payload=_tensor_bytes(_wire_tensor(source)),
        )
        await manager.put(key, envelope, 0.0)
        service = MaKVRemoteServer(
            manager,
            queue_depth=1,
            workers=1,
            max_request_bytes=1024 * 1024,
        )
        response, payload = await service._dispatch(
            "PRECISION_RISK",
            key,
            json.dumps(_risk(1, 0.95)).encode("utf-8"),
        )
        assert not payload
        assert response["accepted"] is True
        assert response["upgraded"] is True
        await manager.close()

    asyncio.run(run())


def test_network_connector_reports_risk_to_independent_manager(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.v1.storage_backend.makv_remote.server",
            "--listen",
            f"127.0.0.1:{port}",
            "--storage-url",
            f"file://{tmp_path}",
            "--bucket-ratios",
            "0.25,0.25,0.5",
            "--bucket-bits",
            "16,8,4",
            "--residual-dtype",
            "float32",
            "--queue-depth",
            "2",
            "--workers",
            "1",
        ],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError as error:
                if process.poll() is not None:
                    raise RuntimeError("MaKV manager exited during startup") from error
                time.sleep(0.05)
        else:
            raise RuntimeError("MaKV manager did not start")

        async def run() -> None:
            key = "network-residual-key"
            source = _source(
                layers=1, tokens=6, heads=2, head_dim=3, dtype=torch.float16
            )
            plan = _plan(
                layout="token",
                dtype=torch.float16,
                bucket_ids=[0, 1, 2, 1, 2, 0],
            )
            envelope = encode_client_put_envelope(
                key=key,
                object_type="raw_with_plan",
                plan=plan,
                raw_kv_payload=_tensor_bytes(_wire_tensor(source)),
            )
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await write_frame(writer, {"op": "PUT", "key": key}, envelope)
            put_header, put_payload = await read_frame(reader)
            writer.close()
            await writer.wait_closed()
            assert put_header["status"] == "ok"
            assert not put_payload

            async def get_object() -> tuple[dict[str, object], bytes]:
                get_reader, get_writer = await asyncio.open_connection(
                    "127.0.0.1", port
                )
                await write_frame(
                    get_writer, {"op": "GET", "key": key}, b""
                )
                get_header, get_payload = await read_frame(get_reader)
                get_writer.close()
                await get_writer.wait_closed()
                return get_header, get_payload

            _, public_before = await get_object()
            public_before_object = decode_makv_object(public_before)
            assert "residual" not in public_before_object.metadata
            assert not any(
                name.startswith("residual_")
                for name in public_before_object.payloads
            )

            class _Key:
                def to_string(self) -> str:
                    return key

            connector = MaKVNetworkConnector.__new__(MaKVNetworkConnector)
            connector.host = "127.0.0.1"
            connector.port = port
            connector.timeout = 10.0
            connector.socket_buffer_bytes = 0
            connector.tcp_nodelay = True
            connector.loop = asyncio.get_running_loop()
            result = await connector.report_precision_risk(
                _Key(), _risk(1, 0.95)
            )
            assert result["status"] == "ok"
            assert result["upgraded"] is True
            _, public_active = await get_object()
            active_object = decode_makv_object(public_active)
            assert "residual" not in active_object.metadata
            assert not any(
                name.startswith("residual_") for name in active_object.payloads
            )
            assert list(active_object.metadata["plan"]["bucket_ids"]) == [
                0,
                0,
                2,
                1,
                2,
                0,
            ]

            expired = await connector.report_precision_risk(
                _Key(), _risk(17, 0.5)
            )
            assert expired["window_expired"] is True
            _, public_restored = await get_object()
            restored_object = decode_makv_object(public_restored)
            assert list(restored_object.metadata["plan"]["bucket_ids"]) == [
                0,
                1,
                2,
                1,
                2,
                0,
            ]

            health = await connector.health()
            assert int(health["pid"]) != os.getpid()
            assert int(health["quantize_calls"]) == 2
            assert int(health["metrics"]["makv_remote_risk_signals"]) == 2
            assert int(health["metrics"]["makv_remote_precision_upgrades"]) == 1
            assert int(health["metrics"]["makv_remote_residual_bytes"]) > 0
            assert int(
                health["metrics"]["makv_remote_precision_window_expirations"]
            ) == 1

        asyncio.run(run())
    finally:
        process.terminate()
        process.wait(timeout=10)
