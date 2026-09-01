# SPDX-License-Identifier: Apache-2.0

"""Storage adapters for complete MaKV objects.

The remote manager owns the object format and quantization policy. Adapters
only store and retrieve already-encoded blobs, which makes the persistence
backend replaceable without changing the MaKV wire protocol.
"""

# Standard
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse
import asyncio
import hashlib
import inspect
import os
import uuid

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

SUPPORTED_STORAGE_BACKENDS = ("file", "redis", "mooncake")
DEFAULT_REDIS_SOCKET_TIMEOUT_S = 600.0


@runtime_checkable
class StorageAdapter(Protocol):
    """Async persistence contract used by ``MaKVRemoteManager``."""

    async def exists(self, key: str) -> bool:
        """Return whether a complete object exists."""

    async def get(self, key: str) -> bytes | None:
        """Read a complete object or return ``None``."""

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        """Read several complete objects in request order."""

    async def put(self, key: str, data: bytes) -> None:
        """Atomically expose one complete object."""

    async def delete(self, key: str) -> bool:
        """Delete one object and return whether it existed."""

    async def list_keys(self) -> list[str]:
        """Return keys available for diagnostics."""

    async def close(self) -> None:
        """Release client resources."""


class FileStorageAdapter:
    """Store complete MaKV blobs atomically in a local directory."""

    def __init__(self, storage_url: str) -> None:
        parsed = urlparse(storage_url)
        if parsed.scheme != "file" or not parsed.path:
            raise ValueError("file storage requires a file:// URL with a path")
        self.root = Path(parsed.path)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.makv"

    async def exists(self, key: str) -> bool:
        """Return whether a complete object exists."""
        return await asyncio.to_thread(self._path(key).is_file)

    async def get(self, key: str) -> bytes | None:
        """Read a complete object or return None."""
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            return None

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        """Read several files concurrently while preserving key order."""
        return list(await asyncio.gather(*(self.get(key) for key in keys)))

    async def put(self, key: str, data: bytes) -> None:
        """Atomically replace one complete object."""
        final_path = self._path(key)
        tmp_path = self.root / f".{final_path.name}.{uuid.uuid4().hex}.tmp"

        def write_and_replace() -> None:
            with tmp_path.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(tmp_path, final_path)

        try:
            await asyncio.to_thread(write_and_replace)
        finally:
            if tmp_path.exists():
                await asyncio.to_thread(tmp_path.unlink)

    async def delete(self, key: str) -> bool:
        """Delete one object and return whether it existed."""
        path = self._path(key)
        try:
            await asyncio.to_thread(path.unlink)
            return True
        except FileNotFoundError:
            return False

    async def list_keys(self) -> list[str]:
        """Return stored object digests for diagnostics."""
        return [path.stem for path in self.root.glob("*.makv")]

    async def close(self) -> None:
        """File storage has no persistent client to close."""


class RedisStorageAdapter:
    """Store MaKV blobs in Redis using namespaced binary string values."""

    def __init__(
        self,
        storage_url: str,
        *,
        namespace: str = "lmcache:makv:",
        socket_timeout: float = DEFAULT_REDIS_SOCKET_TIMEOUT_S,
        socket_connect_timeout: float = 5.0,
        client: Any | None = None,
    ) -> None:
        parsed = urlparse(storage_url)
        if parsed.scheme not in ("redis", "rediss"):
            raise ValueError("Redis storage requires a redis:// or rediss:// URL")
        if not namespace:
            raise ValueError("Redis storage namespace must not be empty")
        if socket_timeout <= 0 or socket_connect_timeout <= 0:
            raise ValueError("Redis socket timeouts must be positive")
        self.namespace = namespace
        if client is not None:
            self.client = client
            return
        try:
            import redis.asyncio as redis
        except ImportError as error:
            raise RuntimeError(
                "Redis MaKV storage requires the 'redis' Python package"
            ) from error
        self.client = redis.Redis.from_url(
            storage_url,
            decode_responses=False,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
        )

    def _key(self, key: str) -> str:
        return f"{self.namespace}{key}"

    def _unkey(self, key: str) -> str | None:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if not key.startswith(self.namespace):
            return None
        return key[len(self.namespace) :]

    async def exists(self, key: str) -> bool:
        """Return whether a complete Redis value exists."""
        return bool(await self.client.exists(self._key(key)))

    async def get(self, key: str) -> bytes | None:
        """Read a binary Redis value or return ``None``."""
        value = await self.client.get(self._key(key))
        return None if value is None else bytes(value)

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        """Read a batch with one Redis MGET round trip."""
        if not keys:
            return []
        values = await self.client.mget([self._key(key) for key in keys])
        if len(values) != len(keys):
            raise RuntimeError("Redis MGET returned an unexpected value count")
        return [None if value is None else bytes(value) for value in values]

    async def put(self, key: str, data: bytes) -> None:
        """Atomically replace a Redis value with the complete blob."""
        result = await self.client.set(self._key(key), data)
        if result is False:
            raise RuntimeError("Redis SET did not acknowledge the MaKV object")

    async def delete(self, key: str) -> bool:
        """Delete one Redis value and return whether it existed."""
        return bool(await self.client.delete(self._key(key)))

    async def list_keys(self) -> list[str]:
        """List namespaced Redis keys for diagnostics."""
        keys: list[str] = []
        async for raw_key in self.client.scan_iter(match=f"{self.namespace}*"):
            key = self._unkey(raw_key)
            if key is not None:
                keys.append(key)
        return keys

    async def close(self) -> None:
        """Close the async Redis client when the installed version supports it."""
        close = getattr(self.client, "aclose", None)
        if close is None:
            close = getattr(self.client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


class MooncakeStorageAdapter:
    """Store complete blobs through the repository's Mooncake Python API.

    Mooncake is optional. The adapter imports it only when selected, so
    importing LMCache does not require the SDK. The current public connector
    API exposes ``put_parts`` and ``batch_get_buffer`` for arbitrary byte
    buffers; those methods are used instead of private Mooncake APIs.
    """

    def __init__(
        self,
        storage_url: str,
        *,
        config_path: str | None = None,
        setup_options: dict[str, Any] | None = None,
        store: Any | None = None,
    ) -> None:
        parsed = urlparse(storage_url)
        if parsed.scheme != "mooncake":
            raise ValueError("Mooncake storage requires a mooncake:// URL")
        if store is not None:
            # Tests and embedding applications may provide an initialized
            # store; do not require the optional SDK import in that case.
            self.store = store
            return
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as error:
            raise RuntimeError(
                "Mooncake MaKV storage requires the optional 'mooncake' SDK"
            ) from error
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
            MooncakeStoreConfig,
            setup_mooncake_store,
        )

        if config_path is not None:
            config = MooncakeStoreConfig.from_file(config_path)
        elif setup_options:
            config = MooncakeStoreConfig(
                setup_config={
                    str(key): str(value) for key, value in setup_options.items()
                }
            )
        else:
            config = MooncakeStoreConfig.load_from_env()

        self.store = MooncakeDistributedStore()
        setup_mooncake_store(self.store, config)

    def _get_sync(self, key: str) -> bytes | None:
        buffers = self.store.batch_get_buffer([key])
        if not buffers:
            return None
        value = buffers[0]
        return None if value is None else bytes(value)

    def _put_sync(self, key: str, data: bytes) -> None:
        result = self.store.put_parts(key, b"", data)
        if isinstance(result, int) and result != 0:
            raise RuntimeError(f"Mooncake put_parts failed with code {result}")
        if result is False:
            raise RuntimeError("Mooncake put_parts did not acknowledge the object")

    async def exists(self, key: str) -> bool:
        """Return whether Mooncake contains a complete object."""
        result = await asyncio.to_thread(self.store.is_exist, key)
        return bool(result)

    async def get(self, key: str) -> bytes | None:
        """Read one object through Mooncake's batch byte-buffer API."""
        return await asyncio.to_thread(self._get_sync, key)

    def _get_many_sync(self, keys: list[str]) -> list[bytes | None]:
        buffers = self.store.batch_get_buffer(keys)
        if len(buffers) != len(keys):
            raise RuntimeError(
                "Mooncake batch_get_buffer returned an unexpected value count"
            )
        return [None if value is None else bytes(value) for value in buffers]

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        """Read several objects with Mooncake's native batch API."""
        if not keys:
            return []
        return await asyncio.to_thread(self._get_many_sync, keys)

    async def put(self, key: str, data: bytes) -> None:
        """Write one complete object through ``put_parts``."""
        await asyncio.to_thread(self._put_sync, key, data)

    async def delete(self, key: str) -> bool:
        """Delete one object when the installed Mooncake exposes ``remove``."""
        remove = getattr(self.store, "remove", None)
        if remove is None:
            raise RuntimeError(
                "The installed Mooncake SDK does not expose remove(); "
                "DELETE is unavailable for this adapter"
            )
        existed = await self.exists(key)
        try:
            result = await asyncio.to_thread(remove, key, True)
        except TypeError:
            result = await asyncio.to_thread(remove, key)
        if isinstance(result, int) and result < 0:
            raise RuntimeError(f"Mooncake remove failed with code {result}")
        return existed

    async def list_keys(self) -> list[str]:
        """Mooncake has no portable key enumeration API."""
        raise RuntimeError("Mooncake storage does not provide LIST enumeration")

    async def close(self) -> None:
        """Close the Mooncake store if the installed SDK exposes close()."""
        close = getattr(self.store, "close", None)
        if close is not None:
            await asyncio.to_thread(close)


def infer_storage_backend(storage_url: str) -> str:
    """Infer a backend name from a storage URL scheme."""
    scheme = urlparse(storage_url).scheme.lower()
    if scheme == "file":
        return "file"
    if scheme in ("redis", "rediss"):
        return "redis"
    if scheme == "mooncake":
        return "mooncake"
    raise ValueError(
        f"Unsupported MaKV storage URL scheme {scheme!r}; "
        "use file://, redis://, rediss://, or mooncake://"
    )


def create_storage_adapter(
    storage_url: str,
    *,
    backend: str | None = None,
    namespace: str = "lmcache:makv:",
    redis_socket_timeout: float = DEFAULT_REDIS_SOCKET_TIMEOUT_S,
    redis_socket_connect_timeout: float = 5.0,
    mooncake_config_path: str | None = None,
    mooncake_setup_options: dict[str, Any] | None = None,
) -> StorageAdapter:
    """Create the configured MaKV persistence adapter."""
    normalized = (backend or infer_storage_backend(storage_url)).strip().lower()
    if normalized not in SUPPORTED_STORAGE_BACKENDS:
        raise ValueError(
            f"Unsupported MaKV storage backend {normalized!r}; "
            f"choose from {', '.join(SUPPORTED_STORAGE_BACKENDS)}"
        )
    inferred = infer_storage_backend(storage_url)
    if normalized != inferred:
        raise ValueError(
            f"MaKV storage backend {normalized!r} does not match URL scheme "
            f"{inferred!r}"
        )
    if normalized == "file":
        return FileStorageAdapter(storage_url)
    if normalized == "redis":
        return RedisStorageAdapter(
            storage_url,
            namespace=namespace,
            socket_timeout=redis_socket_timeout,
            socket_connect_timeout=redis_socket_connect_timeout,
        )
    return MooncakeStorageAdapter(
        storage_url,
        config_path=mooncake_config_path,
        setup_options=mooncake_setup_options,
    )
