#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generation-level LongBench validation for MaKV precision upgrades.

This is a benchmark driver, not a production policy.  For every manifest
row it runs a cold/direct-high generation, a canonical MaKV hit, sends
explicit matched-count risk signals through the independent Manager, runs an
upgraded hit, and then advances the logical precision window to verify that
the canonical view is restored.

The injected signals test the upgrade mechanism.  They are not CONF labels,
so this driver must not be used to claim CONF selection quality.
"""

from __future__ import annotations

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import argparse
import hashlib
import json
import math
import os
import random
import socket
import statistics
import time

# Third Party
import requests
from transformers import AutoTokenizer

# First Party
from lmcache.v1.storage_backend.makv.format import decode_makv_object
from lmcache.v1.storage_backend.makv.precision_risk import (
    CONF_RISK_SEMANTICS,
    CONF_SCORER_VERSION,
)
from lmcache.v1.storage_backend.makv_remote.protocol import FRAME_HEADER

try:
    # Direct script execution must prefer the sibling LMCache benchmark
    # helpers; the repository also contains vllm/benchmarks as a regular
    # package, which would otherwise shadow this namespace.
    from longbench_makv_cachegen import (
        LongBenchExample,
        extract_answer_text,
        importance,
        load_importance_file,
        official_metric_name,
        official_metric_score,
        prompt_ids,
        prompt_token_hash,
        request_completion,
    )
except ImportError:
    from benchmarks.longbench_makv_cachegen import (
        LongBenchExample,
        extract_answer_text,
        importance,
        load_importance_file,
        official_metric_name,
        official_metric_score,
        prompt_ids,
        prompt_token_hash,
        request_completion,
    )


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "experiments/makv_precision_e2e/longbench_e2e_manifest.jsonl"
)


def _read_exact(connection: socket.socket, size: int) -> bytes:
    """Read exactly size bytes from a Manager response."""
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("MaKV Manager closed an incomplete response")
        data.extend(chunk)
    return bytes(data)


def manager_request(
    host: str,
    port: int,
    operation: str,
    *,
    key: str = "",
    payload: bytes = b"",
    request_fields: Mapping[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[dict[str, Any], bytes]:
    """Send one request to the independent MaKV Manager process."""
    request: dict[str, Any] = {"op": operation}
    if key:
        request["key"] = key
    if request_fields:
        duplicate_fields = request.keys() & request_fields.keys()
        if duplicate_fields:
            raise ValueError(
                "Manager request fields cannot override: "
                + ", ".join(sorted(duplicate_fields))
            )
        request.update(request_fields)
    header = json.dumps(request, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    if len(header) > 64 * 1024:
        raise ValueError("Manager request header is too large")
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(FRAME_HEADER.pack(len(header), len(payload)))
        connection.sendall(header)
        if payload:
            connection.sendall(payload)
        response_header_len, response_payload_len = FRAME_HEADER.unpack(
            _read_exact(connection, FRAME_HEADER.size)
        )
        if response_header_len > 64 * 1024:
            raise ValueError("Manager response header is too large")
        response_header = json.loads(
            _read_exact(connection, response_header_len).decode("utf-8")
        )
        response_payload = _read_exact(connection, response_payload_len)
    if response_header.get("status") != "ok":
        raise RuntimeError(
            str(response_header.get("error", "MaKV Manager request failed"))
        )
    return response_header, response_payload


def _manager_health(host: str, port: int, timeout: float) -> dict[str, Any]:
    """Return Manager health."""
    header, _ = manager_request(host, port, "HEALTH", timeout=timeout)
    return header


def _wait_manager_idle(
    host: str, port: int, timeout: float, poll_s: float = 0.25
) -> dict[str, Any]:
    """Wait until all Manager PUT workers have drained."""
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = _manager_health(host, port, timeout)
        if int(latest.get("queue_size", 0)) == 0 and int(
            latest.get("active_jobs", 0)
        ) == 0:
            return latest
        time.sleep(poll_s)
    raise TimeoutError("MaKV Manager did not become idle before timeout")


def _wait_for_runtime_risk(
    host: str,
    port: int,
    timeout: float,
    baseline: int,
    poll_s: float = 0.25,
) -> tuple[dict[str, Any], int]:
    """Wait until an actual vLLM CONF signal reaches the Manager."""
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = _manager_health(host, port, timeout)
        metrics = latest.get("metrics", {})
        observed = int(metrics.get("makv_remote_risk_signals", 0))
        if observed > baseline:
            return latest, observed - baseline
        time.sleep(poll_s)
    raise TimeoutError(
        "runtime CONF observer produced no Manager risk signal before timeout"
    )


def _read_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read rows emitted by prepare_longbench_precision_e2e.py."""
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "task",
        "sample_id",
        "context",
        "question",
        "answers",
        "prompt_run_id",
        "prompt_token_hash",
        "prompt_token_count",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(f"manifest row is missing fields: {missing}")
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


def _example(row: dict[str, Any]) -> LongBenchExample:
    """Convert one manifest row to the shared LongBench example type."""
    raw_length = row.get("source_length")
    length = int(raw_length) if isinstance(raw_length, int) else None
    return LongBenchExample(
        example_id=str(row["sample_id"]),
        context=str(row["context"]),
        question=str(row["question"]),
        answers=tuple(str(value) for value in row.get("answers", [])),
        task=str(row["task"]),
        length=length,
        all_classes=tuple(str(value) for value in row.get("all_classes", [])),
    )


def _request_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build the narrow namespace consumed by request_completion."""
    return SimpleNamespace(
        mode="makv",
        url=args.url,
        model=args.served_model,
        max_tokens=args.max_tokens,
        generation_seed=args.generation_seed,
        timeout=args.timeout,
        scout_overlap=False,
        require_importance_file=args.require_importance,
        risk_source=getattr(args, "risk_source", "synthetic"),
    )


def _completion(
    session: requests.Session,
    request_args: SimpleNamespace,
    example: LongBenchExample,
    token_ids: list[int],
    importance_values: list[float],
    *,
    stream: bool = True,
    max_tokens: int | None = None,
    risk_token_indices: list[int] | None = None,
    runtime_risk_enabled: bool = True,
) -> dict[str, Any]:
    """Run one completion and attach the official LongBench score."""
    response = request_completion(
        session,
        args=request_args,
        ids=token_ids,
        stream=stream,
        max_tokens=max_tokens,
        importance_values=importance_values,
        risk_token_indices=risk_token_indices,
        runtime_risk_enabled=runtime_risk_enabled,
    )
    answer = extract_answer_text(str(response.get("text", "")))
    response["answer_text"] = answer
    response["official_metric"] = official_metric_name(example.task)
    response["official_score"] = official_metric_score(
        answer, example.answers, example.task, example.all_classes
    )
    return response


def _expected_chunk_hashes(
    token_ids: list[int], chunk_size: int
) -> dict[int, tuple[int, int]]:
    """Reproduce LMCache's pinned builtin hash chain for object discovery."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "PYTHONHASHSEED=0 is required to identify builtin LMCache keys"
        )
    prefix_hash = hash(os.environ["PYTHONHASHSEED"])
    result: dict[int, tuple[int, int]] = {}
    full_length = len(token_ids) - len(token_ids) % chunk_size
    for start in range(0, full_length, chunk_size):
        end = start + chunk_size
        prefix_hash = hash((prefix_hash, tuple(token_ids[start:end]), ()))
        result[prefix_hash] = (start, end)
    return result


def _key_hash(key: str) -> int | None:
    """Extract the chunk hash from a CacheEngineKey string."""
    parts = key.rsplit("@", 2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1], 16)
    except ValueError:
        return None


def _list_matching_keys(
    host: str,
    port: int,
    expected: Mapping[int, Any],
    timeout: float,
) -> list[str]:
    """List only keys matching this prompt's expected chunk hashes."""
    list_header, _ = manager_request(
        host,
        port,
        "LIST",
        request_fields={
            "key_hashes": [format(value, "x") for value in expected]
        },
        timeout=timeout,
    )
    keys = list_header.get("keys", [])
    if not isinstance(keys, list) or not all(
        isinstance(key, str) for key in keys
    ):
        raise ValueError("Manager LIST response has invalid keys")
    return keys


def _discover_objects(
    host: str,
    port: int,
    token_ids: list[int],
    chunk_size: int,
    timeout: float,
) -> list[dict[str, Any]]:
    """Find, decode, and validate this prompt's public MaKV objects."""
    expected = _expected_chunk_hashes(token_ids, chunk_size)
    keys = _list_matching_keys(host, port, expected, timeout)
    objects: list[dict[str, Any]] = []
    for raw_key in keys:
        key = str(raw_key)
        chunk = expected.get(_key_hash(key))
        if chunk is None:
            continue
        get_header, blob = manager_request(
            host, port, "GET", key=key, timeout=timeout
        )
        if not bool(get_header.get("found")):
            continue
        decoded = decode_makv_object(blob)
        plan = decoded.metadata.get("plan")
        if not isinstance(plan, dict):
            raise ValueError(f"MaKV object {key} has no plan")
        actual_range = (
            int(plan.get("chunk_start", -1)),
            int(plan.get("chunk_start", -1)) + int(plan.get("chunk_length", -1)),
        )
        if actual_range != chunk:
            raise ValueError(
                f"MaKV object {key} range {actual_range} does not match {chunk}"
            )
        if int(plan.get("token_count", -1)) != len(token_ids):
            raise ValueError(f"MaKV object {key} token_count mismatch")
        if decoded.object_type != "quantized":
            raise ValueError(f"MaKV object {key} is not quantized")
        if decoded.metadata.get("cache_key") != key:
            raise ValueError(f"MaKV object {key} cache_key metadata mismatch")
        objects.append(
            {
                "key": key,
                "chunk_start": chunk[0],
                "chunk_end": chunk[1],
                "public_hash": hashlib.sha256(blob).hexdigest(),
            }
        )
    objects.sort(key=lambda item: int(item["chunk_start"]))
    if len(objects) != len(expected):
        found = {(item["chunk_start"], item["chunk_end"]) for item in objects}
        missing = [value for value in expected.values() if value not in found]
        raise RuntimeError(
            f"stored MaKV objects are incomplete: found={len(objects)} "
            f"expected={len(expected)} missing={missing[:8]}"
        )
    return objects


def _wait_for_stored_objects(
    host: str,
    port: int,
    token_ids: list[int],
    chunk_size: int,
    timeout: float,
    poll_s: float = 0.25,
) -> None:
    """Wait for all fire-and-forget LMCache PUTs to reach the Manager.

    LMCache submits remote PUT futures without blocking the request that
    generated the KV cache.  Manager worker idleness alone is therefore not
    a sufficient barrier: the connector can still be submitting the final
    batch.  The Redis-backed E2E harness exposes LIST, so use the exact
    builtin hash chain as a durable object barrier before probing a hit.
    """
    expected = _expected_chunk_hashes(token_ids, chunk_size)
    deadline = time.monotonic() + timeout
    found_count = 0
    while time.monotonic() < deadline:
        keys = _list_matching_keys(host, port, expected, timeout)
        found = {
            _key_hash(str(key))
            for key in keys
            if _key_hash(str(key)) in expected
        }
        found_count = len(found)
        if found_count == len(expected):
            return
        time.sleep(poll_s)
    raise TimeoutError(
        "MaKV Manager did not expose all stored objects before the hit probe: "
        f"found={found_count} expected={len(expected)}"
    )


def _delete_objects(
    host: str,
    port: int,
    objects: list[dict[str, Any]],
    timeout: float,
) -> int:
    """Delete one fully evaluated prompt's remote objects."""
    deleted = 0
    for item in objects:
        response, _ = manager_request(
            host,
            port,
            "DELETE",
            key=str(item["key"]),
            timeout=timeout,
        )
        if not bool(response.get("deleted")):
            raise RuntimeError(f"failed to delete evaluated object {item['key']}")
        deleted += 1
    return deleted


def _select_positions(
    token_count: int,
    rate: float,
    seed: int,
    candidate_token_limit: int = 0,
) -> list[int]:
    """Select deterministic matched-count prompt token positions."""
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("upgrade rate must be finite and in [0, 1]")
    if candidate_token_limit < 0:
        raise ValueError("candidate token limit must be non-negative")
    count = min(token_count, int(round(token_count * rate)))
    if candidate_token_limit:
        count = min(count, candidate_token_limit)
    if rate > 0.0:
        count = max(1, count)
    if count == 0:
        return []
    return sorted(random.Random(seed).sample(range(token_count), count))


def _risk_signal(
    step: int,
    token_index: int,
    window_tokens: int,
    risk: float = 1.0,
) -> bytes:
    """Encode an experiment risk signal with an absolute token index."""
    return json.dumps(
        {
            "step": int(step),
            "risk": float(risk),
            "scorer_version": CONF_SCORER_VERSION,
            "semantics": CONF_RISK_SEMANTICS,
            "valid": True,
            "token_index": int(token_index),
            "window_tokens": int(window_tokens),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _expiry_signal(step: int, token_index: int, window_tokens: int) -> bytes:
    """Encode a below-threshold signal that advances logical expiry."""
    return json.dumps(
        {
            "step": int(step),
            "risk": 0.0,
            "scorer_version": CONF_SCORER_VERSION,
            "semantics": CONF_RISK_SEMANTICS,
            "valid": True,
            "token_index": int(token_index),
            "window_tokens": int(window_tokens),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _mean_score(rows: list[dict[str, Any]], field: str) -> float | None:
    """Return a mean official score for one response field."""
    values = [
        float(row[field]["official_score"])
        for row in rows
        if isinstance(row.get(field), dict)
        and isinstance(row[field].get("official_score"), (int, float))
    ]
    return statistics.fmean(values) if values else None


def _load_checkpoint(
    result_path: Path,
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load a completed manifest prefix and fail closed on partial checkpoints."""
    if not result_path.is_file():
        raise ValueError(f"resume checkpoint does not exist: {result_path}")
    completed: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        result_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise ValueError(f"blank checkpoint row at line {line_number}")
        try:
            result = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid checkpoint JSON at line {line_number}"
            ) from exc
        if not isinstance(result, dict):
            raise ValueError(f"checkpoint row {line_number} is not an object")
        index = len(completed)
        if index >= len(manifest_rows):
            raise ValueError("checkpoint contains more rows than the manifest")
        example = _example(manifest_rows[index])
        expected = {
            "sample_index": index,
            "sample_id": example.example_id,
            "task": example.task,
            "prompt_token_hash": str(manifest_rows[index]["prompt_token_hash"]),
        }
        actual = {field: result.get(field) for field in expected}
        if actual != expected:
            raise ValueError(
                f"checkpoint row {line_number} does not match manifest prefix: "
                f"expected={expected} actual={actual}"
            )
        completed.append(result)
    return completed


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the LongBench precision-upgrade experiment."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = _read_manifest(Path(args.manifest), args.max_prompts)
    importance_by_hash = load_importance_file(args.importance_file)
    request_args = _request_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "precision_e2e.jsonl"
    if args.resume:
        result_rows = _load_checkpoint(result_path, rows)
    else:
        result_path.write_text("", encoding="utf-8")
        result_rows = []
    resume_from_samples = len(result_rows)
    if resume_from_samples:
        print(
            f"Resuming from {resume_from_samples}/{len(rows)} completed samples",
            flush=True,
        )
    initial_health = _manager_health(args.manager_host, args.manager_port, args.timeout)
    manager_pid = int(initial_health.get("pid", -1))
    if manager_pid <= 0 or manager_pid == os.getpid():
        raise RuntimeError("Manager health did not identify an independent process")

    with (
        requests.Session() as session,
        result_path.open("a", encoding="utf-8", buffering=1) as result_file,
    ):
        for index in range(resume_from_samples, len(rows)):
            row = rows[index]
            example = _example(row)
            generation = row.get("generation", {})
            ids = prompt_ids(
                tokenizer,
                example,
                str(row["prompt_run_id"]),
                enable_thinking=bool(generation.get("enable_thinking", False)),
            )
            actual_hash = prompt_token_hash(ids)
            if actual_hash != str(row["prompt_token_hash"]):
                raise ValueError(f"prompt hash mismatch for {example.example_id}")
            if len(ids) != int(row["prompt_token_count"]):
                raise ValueError(
                    f"prompt token count mismatch for {example.example_id}"
                )
            importance_values = importance_by_hash.get(actual_hash)
            if importance_values is None:
                if args.require_importance and not args.allow_placeholder_importance:
                    raise ValueError(
                        f"real importance is missing for {example.example_id}; "
                        "pass --importance-file or explicitly use "
                        "--allow-placeholder-importance"
                    )
                importance_values = importance(len(ids))
                importance_source = "placeholder"
            else:
                if len(importance_values) != len(ids):
                    raise ValueError(
                        f"importance length mismatch for {example.example_id}: "
                        f"{len(importance_values)} != {len(ids)}"
                    )
                importance_source = "importance_file"

            expected_tokens = (len(ids) // args.chunk_size) * args.chunk_size
            positions = _select_positions(
                expected_tokens,
                args.upgrade_rate,
                args.seed + index,
                args.candidate_token_limit,
            )
            runtime_positions = (
                positions if args.risk_source == "runtime_conf" else None
            )
            direct_high = _completion(
                session,
                request_args,
                example,
                ids,
                importance_values,
                risk_token_indices=runtime_positions,
                runtime_risk_enabled=False,
            )
            _wait_manager_idle(args.manager_host, args.manager_port, args.timeout)
            _wait_for_stored_objects(
                args.manager_host,
                args.manager_port,
                ids,
                args.chunk_size,
                args.timeout,
            )
            probe = _completion(
                session,
                request_args,
                example,
                ids,
                importance_values,
                stream=False,
                max_tokens=1,
                risk_token_indices=runtime_positions,
                runtime_risk_enabled=False,
            )
            cached_tokens = probe.get("cached_tokens")
            if cached_tokens is None or int(cached_tokens) < expected_tokens:
                raise RuntimeError(
                    f"cache did not become a complete hit for {example.example_id}: "
                    f"cached={cached_tokens} expected={expected_tokens}"
                )
            risk_baseline = _manager_health(
                args.manager_host, args.manager_port, args.timeout
            )
            risk_baseline_count = int(
                risk_baseline.get("metrics", {}).get(
                    "makv_remote_risk_signals", 0
                )
            )
            low = _completion(
                session,
                request_args,
                example,
                ids,
                importance_values,
                risk_token_indices=runtime_positions,
                runtime_risk_enabled=True,
            )
            runtime_risk_count = 0
            if args.risk_source == "runtime_conf":
                _, runtime_risk_count = _wait_for_runtime_risk(
                    args.manager_host,
                    args.manager_port,
                    args.timeout,
                    risk_baseline_count,
                )
            objects = _discover_objects(
                args.manager_host,
                args.manager_port,
                ids,
                args.chunk_size,
                args.timeout,
            )
            canonical_hashes = {
                str(item["key"]): str(item["public_hash"]) for item in objects
            }
            last_step_by_key: dict[str, int] = {}
            risk_responses: list[dict[str, Any]] = []
            if args.risk_source != "runtime_conf":
                for step, token_index in enumerate(positions, start=1):
                    target = next(
                        item
                        for item in objects
                        if int(item["chunk_start"])
                        <= token_index
                        < int(item["chunk_end"])
                    )
                    response, _ = manager_request(
                        args.manager_host,
                        args.manager_port,
                        "PRECISION_RISK",
                        key=str(target["key"]),
                        payload=_risk_signal(
                            step,
                            token_index,
                            args.window_tokens,
                            args.risk_value,
                        ),
                        timeout=args.timeout,
                    )
                    if not bool(response.get("accepted")):
                        raise RuntimeError(f"risk signal was rejected: {response}")
                    if bool(response.get("window_active")):
                        last_step_by_key[str(target["key"])] = step
                    risk_responses.append(response)
            risk_signal_count = (
                runtime_risk_count
                if args.risk_source == "runtime_conf"
                else len(risk_responses)
            )

            upgraded_hashes: dict[str, str] = {}
            for item in objects:
                get_header, blob = manager_request(
                    args.manager_host,
                    args.manager_port,
                    "GET",
                    key=str(item["key"]),
                    timeout=args.timeout,
                )
                if not bool(get_header.get("found")):
                    raise RuntimeError(f"upgraded GET missed {item['key']}")
                upgraded_hashes[str(item["key"])] = hashlib.sha256(blob).hexdigest()
            upgraded = _completion(
                session,
                request_args,
                example,
                ids,
                importance_values,
                risk_token_indices=runtime_positions,
                runtime_risk_enabled=False,
            )

            if args.risk_source == "runtime_conf":
                restored_hashes = {}
                after_expiry = None
            else:
                for key, last_step in last_step_by_key.items():
                    chunk_start = next(
                        int(item["chunk_start"])
                        for item in objects
                        if item["key"] == key
                    )
                    response, _ = manager_request(
                        args.manager_host,
                        args.manager_port,
                        "PRECISION_RISK",
                        key=key,
                        payload=_expiry_signal(
                            last_step + args.window_tokens,
                            chunk_start,
                            args.window_tokens,
                        ),
                        timeout=args.timeout,
                    )
                    if not bool(response.get("window_expired")):
                        raise RuntimeError(
                            f"precision window did not expire for {key}"
                        )

                restored_hashes = {}
                for item in objects:
                    get_header, blob = manager_request(
                        args.manager_host,
                        args.manager_port,
                        "GET",
                        key=str(item["key"]),
                        timeout=args.timeout,
                    )
                    if not bool(get_header.get("found")):
                        raise RuntimeError(
                            f"expired-window GET missed {item['key']}"
                        )
                    restored_hashes[str(item["key"])] = hashlib.sha256(
                        blob
                    ).hexdigest()
                after_expiry = _completion(
                    session,
                    request_args,
                    example,
                    ids,
                    importance_values,
                )
            result = {
                "sample_index": index,
                "task": example.task,
                "sample_id": example.example_id,
                "prompt_tokens": len(ids),
                "prompt_token_hash": actual_hash,
                "importance_source": importance_source,
                "expected_cached_tokens": expected_tokens,
                "probe": probe,
                "direct_high": direct_high,
                "makv_low": low,
                "makv_upgrade": upgraded,
                "makv_after_expiry": after_expiry,
                "risk_signal_count": risk_signal_count,
                "risk_source": args.risk_source,
                "runtime_risk_positions": (
                    positions if args.risk_source == "runtime_conf" else None
                ),
                "upgrade_rate_actual": (
                    len(positions) / expected_tokens if expected_tokens else 0.0
                ),
                "risk_responses": risk_responses,
                "canonical_public_hashes": canonical_hashes,
                "upgraded_public_hashes": upgraded_hashes,
                "restored_public_hashes": restored_hashes,
                "canonical_public_hash_unchanged_after_expiry": (
                    None
                    if args.risk_source == "runtime_conf"
                    else canonical_hashes == restored_hashes
                ),
                "upgraded_view_changed": canonical_hashes != upgraded_hashes,
                "objects": [
                    {
                        "key": item["key"],
                        "chunk_start": item["chunk_start"],
                        "chunk_end": item["chunk_end"],
                    }
                    for item in objects
                ],
            }
            if args.delete_after_sample:
                result["deleted_object_count"] = _delete_objects(
                    args.manager_host,
                    args.manager_port,
                    objects,
                    args.timeout,
                )
            result_rows.append(result)
            result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            result_file.flush()
            print(
                f"[{index + 1}/{len(rows)}] task={example.task} "
                f"id={example.example_id} tokens={len(ids)} "
                f"low={float(low['official_score']):.6f} "
                f"upgrade={float(upgraded['official_score']):.6f} "
                f"risk_tokens={len(positions)}",
                flush=True,
            )

    final_health = _manager_health(args.manager_host, args.manager_port, args.timeout)
    direct_score = _mean_score(result_rows, "direct_high")
    low_score = _mean_score(result_rows, "makv_low")
    upgrade_score = _mean_score(result_rows, "makv_upgrade")
    summary = {
        "status": "success",
        "manifest": str(Path(args.manifest).resolve()),
        "samples": len(result_rows),
        "tasks": sorted({str(row["task"]) for row in result_rows}),
        "precision_scheme": "kv_separate_4tier",
        "bucket_bits": [16, 8, 4, 2],
        "residual_dtype": args.residual_dtype,
        "risk_policy": "full",
        "risk_source": (
            "runtime_conf_logits"
            if args.risk_source == "runtime_conf"
            else "explicit_matched_count_benchmark_signal"
        ),
        "risk_upgrade_threshold": args.risk_upgrade_threshold,
        "risk_signal_value": (
            None if args.risk_source == "runtime_conf" else args.risk_value
        ),
        "runtime_risk_signal_count": sum(
            int(row.get("risk_signal_count", 0))
            for row in result_rows
            if row.get("risk_source") == "runtime_conf"
        ),
        "upgrade_rate": args.upgrade_rate,
        "candidate_token_limit": args.candidate_token_limit,
        "delete_after_sample": args.delete_after_sample,
        "resume_from_samples": resume_from_samples,
        "manager_metrics_scope_samples": len(result_rows) - resume_from_samples,
        "window_tokens": args.window_tokens,
        "direct_high_official_score": direct_score,
        "makv_low_official_score": low_score,
        "makv_upgrade_official_score": upgrade_score,
        "makv_after_expiry_official_score": _mean_score(
            result_rows, "makv_after_expiry"
        ),
        "upgrade_score_delta": (
            upgrade_score - low_score
            if upgrade_score is not None and low_score is not None
            else None
        ),
        "direct_high_to_low_score_gap": (
            direct_score - low_score
            if direct_score is not None and low_score is not None
            else None
        ),
        "canonical_public_hash_unchanged_samples": sum(
            bool(row["canonical_public_hash_unchanged_after_expiry"])
            for row in result_rows
        ),
        "upgraded_view_changed_samples": sum(
            bool(row["upgraded_view_changed"]) for row in result_rows
        ),
        "manager_pid": manager_pid,
        "manager_pid_is_independent": manager_pid != os.getpid(),
        "manager_quantize_calls_before": int(initial_health.get("quantize_calls", 0)),
        "manager_quantize_calls_after": int(final_health.get("quantize_calls", 0)),
        "manager_quantize_calls_current_segment": (
            int(final_health.get("quantize_calls", 0))
            - int(initial_health.get("quantize_calls", 0))
        ),
        "checkpoint_object_count": sum(
            len(row.get("objects", [])) for row in result_rows
        ),
        "manager_health": final_health,
        "client_quantize_calls": 0,
        "client_quantize_calls_assertion": (
            "MaKVSerializer only builds the raw envelope; quantization is "
            "Manager-owned"
        ),
        "production_policy_modified": False,
    }
    (output_dir / "precision_e2e.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    parser.add_argument("--served-model", default="qwen3-8b")
    parser.add_argument("--manager-host", default="127.0.0.1")
    parser.add_argument("--manager-port", type=int, default=65432)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--upgrade-rate", type=float, default=0.10)
    parser.add_argument(
        "--candidate-token-limit",
        type=int,
        default=0,
        help="Cap matched-count risk positions for a bounded pilot; 0 means no cap.",
    )
    parser.add_argument(
        "--delete-after-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete fully evaluated prompt objects to bound remote storage.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and continue an existing precision_e2e.jsonl prefix.",
    )
    parser.add_argument("--window-tokens", type=int, default=16)
    parser.add_argument("--risk-upgrade-threshold", type=float, default=0.8)
    parser.add_argument(
        "--risk-value",
        type=float,
        default=1.0,
        help="Controlled benchmark risk value sent to the Manager.",
    )
    parser.add_argument(
        "--risk-source",
        choices=("synthetic", "runtime_conf"),
        default="synthetic",
        help=(
            "Risk source: synthetic sends controlled Manager requests; "
            "runtime_conf consumes real vLLM decode logits."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--importance-file")
    parser.add_argument("--residual-dtype", default="float16")
    parser.add_argument(
        "--require-importance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require real importance for every prompt (default: true).",
    )
    parser.add_argument(
        "--allow-placeholder-importance",
        action="store_true",
        help="Explicitly permit deterministic smoke-test scores.",
    )
    args = parser.parse_args()
    if args.max_prompts == 0 or args.max_prompts < -1:
        raise SystemExit("--max-prompts must be positive or -1")
    if args.chunk_size <= 0 or args.max_tokens <= 0 or args.window_tokens <= 0:
        raise SystemExit("chunk, generation, and window sizes must be positive")
    if not math.isfinite(args.upgrade_rate) or not 0.0 <= args.upgrade_rate <= 1.0:
        raise SystemExit("--upgrade-rate must be finite and in [0,1]")
    if not math.isfinite(args.risk_upgrade_threshold) or not 0.0 <= (
        args.risk_upgrade_threshold
    ) <= 1.0:
        raise SystemExit("--risk-upgrade-threshold must be finite and in [0,1]")
    if not math.isfinite(args.risk_value) or not 0.0 <= args.risk_value <= 1.0:
        raise SystemExit("--risk-value must be finite and in [0,1]")
    if args.candidate_token_limit < 0:
        raise SystemExit("--candidate-token-limit must be non-negative")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
