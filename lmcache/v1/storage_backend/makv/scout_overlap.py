# SPDX-License-Identifier: Apache-2.0

"""Client-side submit/wait helpers for ScoutRank/main-model overlap."""

from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import urlparse
import json
import socket
import threading
import time

# First Party
from lmcache.v1.storage_backend.makv.config import (
    IMPORTANCE_LAYOUT_REQUEST_KEY,
    IMPORTANCE_REQUEST_KEY,
)
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS
from lmcache.v1.storage_backend.makv_remote.protocol import (
    FRAME_HEADER,
    MAX_HEADER_BYTES,
)
from lmcache.v1.storage_backend.makv_remote.scout_protocol import (
    SCOUT_PROTOCOL_VERSION,
    decode_scores,
    encode_token_ids,
    payload_sha256,
)


@dataclass(frozen=True)
class ScoutOverlapResult:
    scores: list[float]
    score_time_ms: float
    total_job_time_ms: float
    wait_time_ms: float
    overlap_hidden_time_ms: float


_RESULT_CACHE_CAPACITY = 128
_RESULT_CACHE: OrderedDict[str, tuple[int, ScoutOverlapResult]] = OrderedDict()
_RESULT_CACHE_LOCK = threading.Lock()


class ScoutOverlapClient:
    """Stateless synchronous client used by scheduler and worker processes."""

    def __init__(self, url: str, timeout_s: float) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme != "makv"
            or parsed.hostname is None
            or parsed.port is None
        ):
            raise ValueError("ScoutRank URL must use makv://host:port")
        if timeout_s <= 0:
            raise ValueError("ScoutRank timeout must be positive")
        self.host = parsed.hostname
        self.port = parsed.port
        self.timeout_s = timeout_s

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = connection.recv(size - len(result))
            if not chunk:
                raise ConnectionError("MaKV manager closed an incomplete frame")
            result.extend(chunk)
        return bytes(result)

    def _request(
        self, header: dict[str, Any], payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes]:
        header_bytes = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(header_bytes) > MAX_HEADER_BYTES:
            raise ValueError("ScoutRank request header exceeds protocol limit")
        with socket.create_connection(
            (self.host, self.port), self.timeout_s
        ) as connection:
            connection.settimeout(self.timeout_s)
            connection.sendall(
                FRAME_HEADER.pack(len(header_bytes), len(payload))
                + header_bytes
                + payload
            )
            raw_frame = self._recv_exact(connection, FRAME_HEADER.size)
            header_length, payload_length = FRAME_HEADER.unpack(raw_frame)
            if header_length <= 0 or header_length > MAX_HEADER_BYTES:
                raise ValueError("invalid MaKV response header length")
            response = json.loads(
                self._recv_exact(connection, header_length).decode("utf-8")
            )
            response_payload = self._recv_exact(connection, payload_length)
        if response.get("status") != "ok":
            raise RuntimeError(
                str(response.get("error", "ScoutRank manager request failed"))
            )
        return response, response_payload

    def submit(self, request_id: str, token_ids: Sequence[int]) -> dict[str, Any]:
        """Submit full prompt IDs without waiting for model execution."""
        payload = encode_token_ids(token_ids)
        response, _ = self._request(
            {
                "op": "SCOUT_SUBMIT",
                "key": request_id,
                "protocol_version": SCOUT_PROTOCOL_VERSION,
                "token_count": len(token_ids),
                "token_sha256": payload_sha256(payload),
            },
            payload,
        )
        return response

    def wait(
        self, request_id: str, token_count: int, *, deferred: bool = False
    ) -> ScoutOverlapResult:
        """Fetch scores at the store boundary after overlapped prefill work."""
        response, payload = self._request(
            {
                "op": "SCOUT_WAIT",
                "key": request_id,
                "protocol_version": SCOUT_PROTOCOL_VERSION,
                "token_count": token_count,
                "timeout_s": self.timeout_s,
                "deferred": deferred,
            }
        )
        return ScoutOverlapResult(
            scores=decode_scores(payload, token_count),
            score_time_ms=float(response["score_time_ms"]),
            total_job_time_ms=float(response["total_job_time_ms"]),
            wait_time_ms=float(response["wait_time_ms"]),
            overlap_hidden_time_ms=float(response["overlap_hidden_time_ms"]),
        )


def scout_overlap_enabled(config: Any) -> bool:
    """Gate all overlap behavior behind an explicit MaKV-only switch."""
    if getattr(config, "remote_serde", None) != "makv":
        return False
    extra = getattr(config, "extra_config", None) or {}
    return bool(extra.get("makv_scout_overlap_enabled", False))


def _client_from_config(config: Any) -> ScoutOverlapClient:
    extra = getattr(config, "extra_config", None) or {}
    url = str(extra.get("makv_scout_url") or getattr(config, "remote_url", ""))
    timeout_s = float(extra.get("makv_scout_timeout_s", 60.0))
    return ScoutOverlapClient(url, timeout_s)


def submit_scout_if_needed(
    config: Any,
    *,
    request_id: str,
    token_ids: Sequence[int],
    request_configs: Optional[dict[str, Any]],
    cached_tokens: int,
) -> bool:
    """Start ScoutRank before prefill when this request will store new KV."""
    if not scout_overlap_enabled(config):
        return False
    if request_configs and request_configs.get(IMPORTANCE_REQUEST_KEY) is not None:
        return False
    chunk_size = int(getattr(config, "chunk_size", 1))
    save_unfull_chunk = bool(getattr(config, "save_unfull_chunk", False))
    cacheable_tokens = (
        len(token_ids)
        if save_unfull_chunk
        else len(token_ids) // chunk_size * chunk_size
    )
    if cached_tokens >= cacheable_tokens:
        return False
    started = time.perf_counter()
    _client_from_config(config).submit(request_id, token_ids)
    CLIENT_METRICS.add(
        makv_scout_submit_calls=1,
        makv_scout_submit_time_ms=(time.perf_counter() - started) * 1000.0,
    )
    return True


def resolve_scout_importance(
    config: Any,
    *,
    request_id: Optional[str],
    token_count: Optional[int],
    request_configs: Optional[dict[str, Any]],
    deferred: bool = False,
) -> Optional[dict[str, Any]]:
    """Wait for scores and return a request-local config copy for serialization."""
    if not scout_overlap_enabled(config):
        return request_configs
    if request_configs and request_configs.get(IMPORTANCE_REQUEST_KEY) is not None:
        return request_configs
    if not request_id or token_count is None or token_count < 0:
        raise ValueError("ScoutRank overlap requires request_id and prompt token count")
    with _RESULT_CACHE_LOCK:
        cached = _RESULT_CACHE.get(request_id)
        if cached is not None:
            if cached[0] != token_count:
                raise ValueError(
                    "cached ScoutRank result token count does not match request"
                )
            _RESULT_CACHE.move_to_end(request_id)
            result = cached[1]
        else:
            result = None
    if result is None:
        started = time.perf_counter()
        result = _client_from_config(config).wait(
            request_id, token_count, deferred=deferred
        )
        client_wait_ms = (time.perf_counter() - started) * 1000.0
        CLIENT_METRICS.add(
            makv_scout_wait_calls=1,
            makv_scout_wait_time_ms=client_wait_ms,
            makv_scout_score_time_ms=result.score_time_ms,
            makv_scout_overlap_hidden_time_ms=result.overlap_hidden_time_ms,
        )
        with _RESULT_CACHE_LOCK:
            _RESULT_CACHE[request_id] = (token_count, result)
            _RESULT_CACHE.move_to_end(request_id)
            while len(_RESULT_CACHE) > _RESULT_CACHE_CAPACITY:
                _RESULT_CACHE.popitem(last=False)
    else:
        client_wait_ms = 0.0
    resolved = dict(request_configs or {})
    resolved[IMPORTANCE_REQUEST_KEY] = result.scores
    resolved[IMPORTANCE_LAYOUT_REQUEST_KEY] = "token"
    resolved["lmcache.makv_scoutrank_timing"] = {
        "score_time_ms": result.score_time_ms,
        "total_job_time_ms": result.total_job_time_ms,
        "manager_wait_time_ms": result.wait_time_ms,
        "client_wait_time_ms": client_wait_ms,
        "overlap_hidden_time_ms": result.overlap_hidden_time_ms,
    }
    return resolved
