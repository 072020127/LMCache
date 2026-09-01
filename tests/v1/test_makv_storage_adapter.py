# SPDX-License-Identifier: Apache-2.0

"""Tests for configurable MaKV persistence adapters."""

# Standard
import asyncio

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.makv.config import get_makv_config
from lmcache.v1.storage_backend.makv_remote.storage_adapter import (
    FileStorageAdapter,
    MooncakeStorageAdapter,
    RedisStorageAdapter,
    create_storage_adapter,
    infer_storage_backend,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.closed = False
        self.get_calls = 0
        self.mget_calls = 0

    async def exists(self, key: str) -> int:
        return int(key in self.values)

    async def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        return self.values.get(key)

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        self.mget_calls += 1
        return [self.values.get(key) for key in keys]

    async def set(self, key: str, value: bytes) -> bool:
        self.values[key] = bytes(value)
        return True

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    async def scan_iter(self, *, match: str):
        prefix = match.removesuffix("*")
        for key in list(self.values):
            if key.startswith(prefix):
                yield key

    async def aclose(self) -> None:
        self.closed = True


class _FakeMooncake:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.closed = False

    def is_exist(self, key: str) -> bool:
        return key in self.values

    def put_parts(self, key: str, metadata: bytes, data: bytes) -> int:
        assert metadata == b""
        self.values[key] = bytes(data)
        return 0

    def batch_get_buffer(self, keys: list[str]) -> list[bytes | None]:
        return [self.values.get(key) for key in keys]

    def remove(self, key: str, force: bool = False) -> int:
        del self.values[key]
        return 0

    def close(self) -> None:
        self.closed = True


def test_storage_backend_inference_and_validation() -> None:
    assert infer_storage_backend("file:///tmp/makv") == "file"
    assert infer_storage_backend("redis://127.0.0.1:6379/0") == "redis"
    assert infer_storage_backend("rediss://127.0.0.1:6379/0") == "redis"
    assert infer_storage_backend("mooncake://") == "mooncake"

    with pytest.raises(ValueError, match="Unsupported MaKV storage URL scheme"):
        infer_storage_backend("s3://bucket/prefix")
    with pytest.raises(ValueError, match="does not match"):
        create_storage_adapter(
            "redis://127.0.0.1:6379/0", backend="file"
        )


def test_file_adapter_factory(tmp_path) -> None:
    adapter = create_storage_adapter(f"file://{tmp_path}")
    assert isinstance(adapter, FileStorageAdapter)


def test_lmcache_config_selects_redis_backend() -> None:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        local_cpu=True,
        remote_url="makv://127.0.0.1:65432",
        remote_serde="makv",
        extra_config={
            "makv_storage_backend": "redis",
            "makv_storage_url": "redis://127.0.0.1:6379/1",
            "makv_storage_namespace": "test:makv:",
            "makv_require_cuda_dequant": False,
        },
    )
    makv_config = get_makv_config(config)
    assert makv_config.storage_backend == "redis"
    assert makv_config.storage_url == "redis://127.0.0.1:6379/1"
    assert makv_config.storage_namespace == "test:makv:"


def test_lmcache_config_defaults_to_redis_backend() -> None:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        local_cpu=True,
        remote_url="makv://127.0.0.1:65432",
        remote_serde="makv",
        extra_config={
            "makv_require_cuda_dequant": False,
        },
    )
    makv_config = get_makv_config(config)
    assert makv_config.storage_backend == "redis"
    assert makv_config.storage_url == "redis://127.0.0.1:6379/0"


def test_redis_adapter_round_trip_and_namespace() -> None:
    async def run() -> None:
        client = _FakeRedis()
        adapter = RedisStorageAdapter(
            "redis://127.0.0.1:6379/0",
            namespace="test:makv:",
            client=client,
        )
        assert not await adapter.exists("alpha")
        await adapter.put("alpha", b"first")
        assert await adapter.exists("alpha")
        assert await adapter.get("alpha") == b"first"
        assert await adapter.get_many(["alpha", "missing"]) == [b"first", None]
        assert client.mget_calls == 1
        assert client.get_calls == 1

        # SET is the atomic overwrite operation for one complete object.
        await adapter.put("alpha", b"second")
        assert await adapter.get("alpha") == b"second"
        await adapter.put("beta", b"other")
        assert sorted(await adapter.list_keys()) == ["alpha", "beta"]
        assert "test:makv:alpha" in client.values
        assert await adapter.delete("alpha")
        assert not await adapter.exists("alpha")
        assert not await adapter.delete("alpha")
        await adapter.close()
        assert client.closed

    asyncio.run(run())


def test_redis_adapter_uses_large_blob_socket_timeout(monkeypatch) -> None:
    import redis.asyncio as redis_asyncio

    captured: dict[str, object] = {}

    def fake_from_url(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(redis_asyncio.Redis, "from_url", fake_from_url)
    RedisStorageAdapter(
        "redis://127.0.0.1:6379/0",
        socket_timeout=600.0,
        socket_connect_timeout=3.0,
    )

    assert captured["socket_timeout"] == 600.0
    assert captured["socket_connect_timeout"] == 3.0


def test_mooncake_adapter_uses_public_blob_methods() -> None:
    async def run() -> None:
        store = _FakeMooncake()
        adapter = MooncakeStorageAdapter("mooncake://", store=store)
        await adapter.put("alpha", b"payload")
        assert await adapter.get("alpha") == b"payload"
        assert await adapter.exists("alpha")
        assert await adapter.delete("alpha")
        assert not await adapter.exists("alpha")
        await adapter.close()
        assert store.closed

    asyncio.run(run())


def test_mooncake_import_is_lazy_and_reports_missing_sdk(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "mooncake.store":
            raise ModuleNotFoundError("blocked test Mooncake SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="optional 'mooncake' SDK"):
        MooncakeStorageAdapter("mooncake://")
