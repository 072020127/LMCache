# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded MaKV Remote Manager LIST responses."""

# Standard
import asyncio

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.makv_remote.protocol import read_frame, write_frame
from lmcache.v1.storage_backend.makv_remote.server import MaKVRemoteServer


class _Storage:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    async def list_keys(self) -> list[str]:
        return list(self.keys)


class _Manager:
    def __init__(self, keys: list[str]) -> None:
        self.storage = _Storage(keys)


def _server(keys: list[str]) -> MaKVRemoteServer:
    return MaKVRemoteServer(
        _Manager(keys),
        queue_depth=1,
        workers=1,
        max_request_bytes=1024,
    )


def test_list_filters_large_history_by_chunk_hash() -> None:
    async def run() -> None:
        history = [
            f"model@1@0@{value:x}@bfloat16" for value in range(5000)
        ]
        service = _server(history)
        response, payload = await service._dispatch(
            "LIST",
            "",
            b"",
            {"key_hashes": ["7", "1000"]},
        )
        assert payload == b""
        assert response == {
            "keys": [
                "model@1@0@7@bfloat16",
                "model@1@0@1000@bfloat16",
            ]
        }

    asyncio.run(run())


def test_list_without_filter_remains_backward_compatible() -> None:
    async def run() -> None:
        keys = ["model@1@0@abc@bfloat16"]
        response, _ = await _server(keys)._dispatch("LIST", "", b"", {})
        assert response == {"keys": keys}

    asyncio.run(run())


def test_filtered_list_stays_bounded_over_tcp() -> None:
    async def run() -> None:
        history = [
            f"model@1@0@{value:x}@bfloat16" for value in range(5000)
        ]
        service = _server(history)
        listener = await asyncio.start_server(
            service.handle_client, "127.0.0.1", 0
        )
        try:
            port = int(listener.sockets[0].getsockname()[1])
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await write_frame(
                writer,
                {"op": "LIST", "key_hashes": ["7", "1000"]},
            )
            response, payload = await read_frame(reader, max_payload_bytes=1024)
            assert payload == b""
            assert response == {
                "status": "ok",
                "keys": [
                    "model@1@0@7@bfloat16",
                    "model@1@0@1000@bfloat16",
                ],
            }
            writer.close()
            await writer.wait_closed()
        finally:
            listener.close()
            await listener.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize("value", ["abc", [1], [""]])
def test_list_rejects_invalid_hash_filters(value: object) -> None:
    async def run() -> None:
        with pytest.raises(ValueError, match="key_hashes"):
            await _server([])._dispatch(
                "LIST", "", b"", {"key_hashes": value}
            )

    asyncio.run(run())
