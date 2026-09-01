#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure cold and remote-hit TTFT for synthetic prompt lengths.

The client sends token IDs directly to the OpenAI-compatible completion API.
Each length and repetition has a unique prefix, so a previous measurement
cannot satisfy the next measurement through a shared LMCache prefix.
"""

from __future__ import annotations

# Standard
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
import argparse
import csv
import hashlib
import json
import math
import socket
import struct
import statistics
import time

# Third Party
import requests
from transformers import AutoTokenizer


Mode = Literal["naive", "cachegen", "makv"]

SUMMARY_FIELDS = (
    "mode",
    "prompt_tokens",
    "max_tokens",
    "requested_repetitions",
    "successful_repetitions",
    "valid_hit_repetitions",
    "cold_ttft_mean_ms",
    "cold_ttft_median_ms",
    "cold_ttft_p95_ms",
    "hit_ttft_mean_ms",
    "hit_ttft_median_ms",
    "hit_ttft_p95_ms",
    "cold_latency_mean_ms",
    "cold_latency_median_ms",
    "cold_latency_p95_ms",
    "hit_latency_mean_ms",
    "hit_latency_median_ms",
    "hit_latency_p95_ms",
    "cold_cached_tokens_mean",
    "cold_cached_tokens_median",
    "hit_cached_tokens_mean",
    "hit_cached_tokens_median",
    "expected_cached_tokens",
    "cache_hit_rate",
    "status",
    "error",
)

_MANAGER_FRAME_HEADER = struct.Struct("!IQ")


def parse_context_lengths(value: str) -> tuple[int, ...]:
    """Parse comma-separated token lengths, accepting ``k`` and ``m`` suffixes.

    Args:
        value: Comma-separated values such as ``1k,2k,16k``.

    Returns:
        Positive, duplicate-free token lengths in input order.

    Raises:
        ValueError: If a value is malformed or non-positive.
    """
    lengths: list[int] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        multiplier = 1
        if token.endswith("k"):
            multiplier = 1024
            token = token[:-1]
        elif token.endswith("m"):
            multiplier = 1024 * 1024
            token = token[:-1]
        try:
            length = int(token) * multiplier
        except ValueError as exc:
            raise ValueError(f"invalid context length: {item!r}") from exc
        if length <= 0:
            raise ValueError(f"context length must be positive: {item!r}")
        if length not in lengths:
            lengths.append(length)
    if not lengths:
        raise ValueError("at least one context length is required")
    return tuple(lengths)


def build_prompt_ids(tokenizer: Any, token_count: int, case_id: str) -> list[int]:
    """Build a deterministic prompt with exactly ``token_count`` token IDs.

    A case-specific hash is placed at the beginning of every prompt. This
    prevents a shorter test case from populating a prefix that would make a
    later longer case look like a cold miss when it is not.

    Args:
        tokenizer: A Hugging Face tokenizer compatible with the model.
        token_count: Exact number of prompt token IDs to return.
        case_id: Unique benchmark case identifier.

    Returns:
        Exactly ``token_count`` token IDs.

    Raises:
        ValueError: If the tokenizer cannot encode the filler text.
    """
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:24]
    marker = (
        f"Synthetic TTFT benchmark case {case_id} marker {digest}. "
        "This text is intentionally meaningless. "
    )
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Repeat this neutral benchmark text without relying on its meaning. "
    )
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    if not marker_ids or not filler_ids:
        raise ValueError("tokenizer produced no IDs for the benchmark prompt")
    repeated = marker_ids + filler_ids * math.ceil(
        max(0, token_count - len(marker_ids)) / len(filler_ids)
    )
    return list(repeated[:token_count])


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def extract_cached_tokens(body: dict[str, Any]) -> tuple[int | None, str | None]:
    """Extract LMCache cached-token statistics from known response shapes."""
    transfer = body.get("kv_transfer_params")
    if isinstance(transfer, dict):
        stats = transfer.get("cached_token_stats")
        if isinstance(stats, dict):
            value = _nonnegative_int(stats.get("num_lmcache_cached_tokens"))
            if value is not None:
                return value, "lmcache"
        value = _nonnegative_int(transfer.get("num_lmcache_cached_tokens"))
        if value is not None:
            return value, "lmcache"

    usage = body.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            value = _nonnegative_int(details.get("cached_tokens"))
            if value is not None:
                return value, "vllm_usage"
    return None, None


def _read_exact(connection: socket.socket, size: int) -> bytes:
    """Read exactly one manager protocol field from a TCP socket."""
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("MaKV manager closed an incomplete health response")
        data.extend(chunk)
    return bytes(data)


def manager_health(address: str, timeout: float) -> dict[str, Any]:
    """Read the independent manager health state without using vLLM."""
    host, port_text = address.rsplit(":", 1)
    request = json.dumps({"op": "HEALTH", "key": ""}, separators=(",", ":")).encode(
        "utf-8"
    )
    with socket.create_connection(
        (host, int(port_text)), timeout=timeout
    ) as connection:
        connection.sendall(_MANAGER_FRAME_HEADER.pack(len(request), 0) + request)
        raw = _read_exact(connection, _MANAGER_FRAME_HEADER.size)
        header_length, payload_length = _MANAGER_FRAME_HEADER.unpack(raw)
        if header_length <= 0 or header_length > 1 << 20:
            raise ValueError(f"invalid manager health header length: {header_length}")
        if payload_length:
            raise ValueError("manager health response unexpectedly has a payload")
        response = json.loads(_read_exact(connection, header_length).decode("utf-8"))
    if response.get("status") != "ok":
        raise RuntimeError(str(response.get("error", "MaKV manager health failed")))
    return response


def wait_for_manager_idle(
    address: str | None,
    *,
    timeout: float,
    wait_timeout: float,
    wait_interval: float,
) -> None:
    """Wait until asynchronous MaKV PUT work is durably processed."""
    if not address:
        return
    deadline = time.monotonic() + wait_timeout
    last_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_state = manager_health(address, timeout)
        if (
            int(last_state.get("queue_size", 0)) == 0
            and int(last_state.get("active_jobs", 0)) == 0
        ):
            return
        time.sleep(wait_interval)
    raise TimeoutError(f"MaKV manager did not become idle: {last_state}")


def request_completion(
    session: requests.Session,
    *,
    url: str,
    model: str,
    mode: Mode,
    prompt_ids: Sequence[int],
    max_tokens: int,
    timeout: float,
    seed: int,
    stream: bool,
) -> dict[str, Any]:
    """Send one completion request and measure time to the first text token.

    Args:
        session: Reusable HTTP session.
        url: OpenAI-compatible completion endpoint.
        model: Served model name.
        mode: LMCache serialization mode.
        prompt_ids: Exact token IDs to send as the prompt.
        max_tokens: Maximum generated tokens.
        timeout: Requests connect/read timeout in seconds.
        seed: Deterministic generation seed.
        stream: Whether to use streaming for TTFT measurement.

    Returns:
        A dictionary containing latency, TTFT, and cached-token fields.

    Raises:
        requests.RequestException: If the endpoint request fails.
    """
    transfer: dict[str, Any] = {"cached_token_stats": True}
    if mode == "makv":
        # Constant importance is sufficient for a transport/TTFT benchmark and
        # avoids mixing predictor quality with the storage comparison.
        transfer.update(
            {
                "lmcache.makv_importance": [0.5] * len(prompt_ids),
                "lmcache.makv_importance_layout": "token",
            }
        )
    payload: dict[str, Any] = {
        "model": model,
        "prompt": list(prompt_ids),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": stream,
        "kv_transfer_params": transfer,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    response = session.post(url, json=payload, timeout=timeout, stream=stream)
    try:
        if not response.ok:
            detail = response.text[:1000]
            raise requests.HTTPError(
                f"{response.status_code} from {response.url}: {detail}",
                response=response,
            )
        first_token_at: float | None = None
        cached_tokens: int | None = None
        cached_tokens_source: str | None = None

        def update_cached_tokens(body: dict[str, Any]) -> None:
            nonlocal cached_tokens, cached_tokens_source
            value, source = extract_cached_tokens(body)
            if value is None:
                return
            if cached_tokens_source != "lmcache" or source == "lmcache":
                cached_tokens = value
                cached_tokens_source = source

        if stream:
            text_parts: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                body = json.loads(data)
                choices = body.get("choices") or []
                if choices:
                    piece = choices[0].get("text") or ""
                    if piece and first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
                update_cached_tokens(body)
        else:
            body = response.json()
            update_cached_tokens(body)
    finally:
        response.close()

    finished = time.perf_counter()
    return {
        "latency_ms": (finished - started) * 1000,
        "ttft_ms": (
            (first_token_at - started) * 1000 if first_token_at is not None else None
        ),
        "cached_tokens": cached_tokens,
        "cached_tokens_source": cached_tokens_source,
    }


def wait_for_cache_hit(
    session: requests.Session,
    *,
    url: str,
    model: str,
    mode: Mode,
    prompt_ids: Sequence[int],
    expected_cached_tokens: int,
    max_tokens: int,
    timeout: float,
    seed: int,
    wait_timeout: float,
    wait_interval: float,
    manager_health_address: str | None,
    manager_wait_timeout: float,
    manager_wait_interval: float,
) -> dict[str, Any]:
    """Poll the endpoint until the complete prompt is reported as cached."""
    deadline = time.monotonic() + wait_timeout
    last_probe: dict[str, Any] = {
        "cached_tokens": None,
        "cached_tokens_source": None,
        "latency_ms": None,
        "ttft_ms": None,
    }
    while True:
        # A probe is a real inference request. Do not issue another one while
        # the previous cold request's asynchronous PUT is still in flight,
        # otherwise a slow MaKV quantizer can create duplicate full prompts.
        wait_for_manager_idle(
            manager_health_address,
            timeout=timeout,
            wait_timeout=manager_wait_timeout,
            wait_interval=manager_wait_interval,
        )
        last_probe = request_completion(
            session,
            url=url,
            model=model,
            mode=mode,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            timeout=timeout,
            seed=seed,
            stream=False,
        )
        if (
            expected_cached_tokens == 0
            or (last_probe.get("cached_tokens") or 0) >= expected_cached_tokens
        ):
            return last_probe
        if time.monotonic() >= deadline:
            return last_probe
        time.sleep(wait_interval)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _median(values: Sequence[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return statistics.median(filtered) if filtered else None


def _mean(values: Sequence[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return statistics.fmean(filtered) if filtered else None


def _format_number(value: float | int | None) -> str:
    return "" if value is None else f"{value:.6f}"


def append_summary(path: Path, row: dict[str, Any]) -> None:
    """Append one summary row to a CSV file, creating its header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def summarize_samples(
    *,
    mode: Mode,
    prompt_tokens: int,
    max_tokens: int,
    repetitions: int,
    expected_cached_tokens: int,
    samples: Sequence[dict[str, Any]],
    errors: Sequence[str],
) -> dict[str, Any]:
    """Create one CSV-compatible aggregate row for a prompt length."""
    # A cold request is successful even when the asynchronous PUT was not
    # ready before the hit deadline; keep it in the cold aggregate and report
    # the missing hit separately.
    successful = list(samples)
    valid_hits = [sample for sample in successful if sample.get("valid_hit")]
    cold_ttfts = [
        sample["cold"]["ttft_ms"]
        for sample in successful
        if sample["cold"].get("ttft_ms") is not None
    ]
    hit_ttfts = [
        sample["hit"]["ttft_ms"]
        for sample in valid_hits
        if sample["hit"].get("ttft_ms") is not None
    ]
    cold_latencies = [
        sample["cold"]["latency_ms"]
        for sample in successful
        if sample["cold"].get("latency_ms") is not None
    ]
    hit_latencies = [
        sample["hit"]["latency_ms"]
        for sample in valid_hits
        if sample["hit"].get("latency_ms") is not None
    ]
    status = (
        "ok"
        if len(successful) == repetitions and len(valid_hits) == repetitions
        else "partial"
    )
    if not successful:
        status = "error"
    return {
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "requested_repetitions": repetitions,
        "successful_repetitions": len(successful),
        "valid_hit_repetitions": len(valid_hits),
        "cold_ttft_mean_ms": _format_number(_mean(cold_ttfts)),
        "cold_ttft_median_ms": _format_number(_median(cold_ttfts)),
        "cold_ttft_p95_ms": _format_number(_percentile(cold_ttfts, 0.95)),
        "hit_ttft_mean_ms": _format_number(_mean(hit_ttfts)),
        "hit_ttft_median_ms": _format_number(_median(hit_ttfts)),
        "hit_ttft_p95_ms": _format_number(_percentile(hit_ttfts, 0.95)),
        "cold_latency_mean_ms": _format_number(_mean(cold_latencies)),
        "cold_latency_median_ms": _format_number(_median(cold_latencies)),
        "cold_latency_p95_ms": _format_number(_percentile(cold_latencies, 0.95)),
        "hit_latency_mean_ms": _format_number(_mean(hit_latencies)),
        "hit_latency_median_ms": _format_number(_median(hit_latencies)),
        "hit_latency_p95_ms": _format_number(_percentile(hit_latencies, 0.95)),
        "cold_cached_tokens_mean": _format_number(
            _mean([sample["cold"].get("cached_tokens") for sample in successful])
        ),
        "cold_cached_tokens_median": _format_number(
            _median([sample["cold"].get("cached_tokens") for sample in successful])
        ),
        "hit_cached_tokens_mean": _format_number(
            _mean([sample["hit"].get("cached_tokens") for sample in valid_hits])
        ),
        "hit_cached_tokens_median": _format_number(
            _median([sample["hit"].get("cached_tokens") for sample in valid_hits])
        ),
        "expected_cached_tokens": expected_cached_tokens,
        "cache_hit_rate": _format_number(
            len(valid_hits) / len(successful) if successful else None
        ),
        "status": status,
        "error": " | ".join(errors)[:2000],
    }


def main() -> None:
    """Run the synthetic prompt-length TTFT benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("naive", "cachegen", "makv"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--context-lengths",
        default="1k,2k,4k,16k,32k,64k,128k",
        help="Comma-separated prompt lengths in tokens.",
    )
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--cache-wait-timeout", type=float, default=900.0)
    parser.add_argument("--cache-wait-interval", type=float, default=1.0)
    parser.add_argument(
        "--manager-health",
        default=None,
        help="Optional host:port for waiting on the independent MaKV manager.",
    )
    parser.add_argument("--manager-wait-timeout", type=float, default=900.0)
    parser.add_argument("--manager-wait-interval", type=float, default=1.0)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    args = parser.parse_args()

    if args.repetitions <= 0 or args.warmup_requests < 0:
        raise ValueError(
            "repetitions must be positive and warmup-requests non-negative"
        )
    if args.max_tokens <= 0 or args.chunk_size <= 0:
        raise ValueError("max-tokens and chunk-size must be positive")
    if args.max_model_len <= args.max_tokens:
        raise ValueError("max-model-len must be greater than max-tokens")
    lengths = parse_context_lengths(args.context_lengths)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with (
        requests.Session() as session,
        args.raw_jsonl.open("a", encoding="utf-8") as raw,
    ):
        for warmup_index in range(args.warmup_requests):
            warmup_ids = build_prompt_ids(
                tokenizer,
                min(128, args.max_model_len - args.max_tokens),
                f"{args.run_id}:warmup:{warmup_index}",
            )
            request_completion(
                session,
                url=args.url,
                model=args.model,
                mode=args.mode,
                prompt_ids=warmup_ids,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
                seed=args.generation_seed,
                stream=True,
            )

        for prompt_tokens in lengths:
            expected = (prompt_tokens // args.chunk_size) * args.chunk_size
            if prompt_tokens + args.max_tokens > args.max_model_len:
                row = {
                    "mode": args.mode,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": args.max_tokens,
                    "requested_repetitions": args.repetitions,
                    "successful_repetitions": 0,
                    "valid_hit_repetitions": 0,
                    "expected_cached_tokens": expected,
                    "status": "skipped_context_limit",
                    "error": (
                        f"prompt+max_tokens={prompt_tokens + args.max_tokens} "
                        f"> max_model_len={args.max_model_len}"
                    ),
                }
                append_summary(args.summary_csv, row)
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                continue

            samples: list[dict[str, Any]] = []
            errors: list[str] = []
            for repetition in range(args.repetitions):
                case_id = f"{args.run_id}:length={prompt_tokens}:rep={repetition}"
                prompt_ids = build_prompt_ids(tokenizer, prompt_tokens, case_id)
                try:
                    cold = request_completion(
                        session,
                        url=args.url,
                        model=args.model,
                        mode=args.mode,
                        prompt_ids=prompt_ids,
                        max_tokens=args.max_tokens,
                        timeout=args.request_timeout,
                        seed=args.generation_seed,
                        stream=True,
                    )
                    probe = wait_for_cache_hit(
                        session,
                        url=args.url,
                        model=args.model,
                        mode=args.mode,
                        prompt_ids=prompt_ids,
                        expected_cached_tokens=expected,
                        max_tokens=args.max_tokens,
                        timeout=args.request_timeout,
                        seed=args.generation_seed,
                        wait_timeout=args.cache_wait_timeout,
                        wait_interval=args.cache_wait_interval,
                        manager_health_address=args.manager_health,
                        manager_wait_timeout=args.manager_wait_timeout,
                        manager_wait_interval=args.manager_wait_interval,
                    )
                    valid_hit = (
                        expected == 0 or (probe.get("cached_tokens") or 0) >= expected
                    )
                    hit = None
                    if valid_hit:
                        hit = request_completion(
                            session,
                            url=args.url,
                            model=args.model,
                            mode=args.mode,
                            prompt_ids=prompt_ids,
                            max_tokens=args.max_tokens,
                            timeout=args.request_timeout,
                            seed=args.generation_seed,
                            stream=True,
                        )
                        if hit.get("cached_tokens") is None:
                            # vLLM may expose cache statistics only on the
                            # non-streaming probe response. The probe already
                            # established that this request is a complete hit.
                            hit["cached_tokens"] = probe.get("cached_tokens")
                            hit["cached_tokens_source"] = probe.get(
                                "cached_tokens_source"
                            )
                    sample = {
                        "mode": args.mode,
                        "prompt_tokens": prompt_tokens,
                        "repetition": repetition,
                        "cold": cold,
                        "probe": probe,
                        "hit": hit,
                        "valid_hit": valid_hit,
                    }
                    raw.write(json.dumps(sample) + "\n")
                    raw.flush()
                    samples.append(sample)
                    if not valid_hit or hit is None:
                        errors.append(
                            f"rep={repetition}: cache not ready; "
                            f"cached={probe.get('cached_tokens')} expected={expected}"
                        )
                except Exception as exc:  # Keep later lengths measurable.
                    error = f"rep={repetition}: {type(exc).__name__}: {exc}"
                    errors.append(error)
                    raw.write(
                        json.dumps(
                            {
                                "mode": args.mode,
                                "prompt_tokens": prompt_tokens,
                                "repetition": repetition,
                                "status": "error",
                                "error": error,
                            }
                        )
                        + "\n"
                    )
                    raw.flush()

            row = summarize_samples(
                mode=args.mode,
                prompt_tokens=prompt_tokens,
                max_tokens=args.max_tokens,
                repetitions=args.repetitions,
                expected_cached_tokens=expected,
                samples=samples,
                errors=errors,
            )
            append_summary(args.summary_csv, row)
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
