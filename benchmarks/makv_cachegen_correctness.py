#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate cold-vs-remote-hit generation correctness for three serde modes.

Run this script once against each separately started vLLM server, using the
same ``--run-id`` and sample selection. The ``run`` command writes one JSON
object per example and a ``.summary.json`` file. GSM8K JSONL and MATH
``test/train-*.parquet`` directories are supported; MATH Parquet loading is
lazy and requires the optional ``pyarrow`` package.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time

# Third Party
import requests
from transformers import AutoTokenizer

DEFAULT_GSM8K_TEST = Path(
    "/media/home/iic/mahaoyuan/datas/gsm8k-openai/grade_school_math/data/test.jsonl"
)
DEFAULT_GSM8K_TRAIN = Path(
    "/media/home/iic/mahaoyuan/datas/gsm8k-openai/grade_school_math/data/train.jsonl"
)
DEFAULT_MATH_ROOT = Path("/media/home/iic/mahaoyuan/datas/math")

NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[1-9]\d*)?%?")
HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer is)\s*[:=]?\s*([^\n]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    gold_text: str


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _last_boxed(text: str) -> Optional[str]:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    begin = start + len(marker)
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index]
    return None


def _normalize_number(candidate: str) -> Optional[str]:
    match = NUMBER_RE.search(candidate.replace(" ", ""))
    if match is None:
        return None
    value = match.group(0).replace("$", "").replace(",", "")
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            decimal = Decimal(numerator) / Decimal(denominator)
        else:
            decimal = Decimal(value)
    except (InvalidOperation, ZeroDivisionError):
        return None
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return f"{normalized}%" if percent else normalized


def extract_answer(text: str, dataset: str) -> Optional[str]:
    """Extract a stable answer for GSM8K or a normalized MATH exact match."""
    boxed = _last_boxed(text)
    if dataset == "math" and boxed is not None:
        return re.sub(r"\s+", "", boxed).replace("\\,", "")

    candidate_groups = [
        HASH_ANSWER_RE.findall(text),
        [boxed] if boxed is not None else [],
        FINAL_ANSWER_RE.findall(text),
        NUMBER_RE.findall(text),
    ]
    for candidates in candidate_groups:
        for candidate in reversed(candidates):
            normalized = _normalize_number(candidate)
            if normalized is not None:
                return normalized
    if dataset == "math":
        compact = re.sub(r"\s+", "", text).replace("\\,", "")
        return compact or None
    return None


def _parquet_paths(path: Path, split: str | None) -> list[Path]:
    """Resolve one Parquet file or a subject directory for a dataset split."""
    if path.is_dir():
        pattern = f"{split}-*.parquet" if split else "*.parquet"
        return sorted(path.rglob(pattern))
    if path.suffix.lower() == ".parquet":
        return [path]
    return []


def _load_parquet_examples(paths: list[Path], dataset: str) -> list[Example]:
    """Load MATH-style rows from Parquet without importing pyarrow at module load."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "MATH Parquet input requires pyarrow. Install it in the active "
            "environment with `uv pip install pyarrow`."
        ) from error

    examples: list[Example] = []
    for parquet_path in paths:
        table = parquet.read_table(parquet_path)
        for row_index, item in enumerate(table.to_pylist()):
            if dataset == "gsm8k":
                question, answer = item["question"], item["answer"]
            else:
                question = item.get("problem") or item.get("question")
                answer = item.get("solution") or item.get("answer")
                if question is None or answer is None:
                    raise ValueError("MATH Parquet needs problem/solution fields")
            example_id = f"{parquet_path.parent.name}:{parquet_path.stem}:{row_index}"
            examples.append(Example(example_id, str(question), str(answer)))
    return examples


def load_examples(
    path: Path, dataset: str, *, split: str | None = None
) -> list[Example]:
    """Load GSM8K JSONL or MATH Parquet files from a file or subject directory."""
    parquet_paths = _parquet_paths(path, split)
    if parquet_paths:
        return _load_parquet_examples(parquet_paths, dataset)
    if path.suffix.lower() == ".parquet":
        raise ValueError(f"No Parquet files found at {path}")
    examples = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            if dataset == "gsm8k":
                question, answer = item["question"], item["answer"]
            else:
                question = item.get("problem") or item.get("question")
                answer = item.get("solution") or item.get("answer")
                if question is None or answer is None:
                    raise ValueError("MATH JSONL needs problem/solution fields")
            examples.append(Example(str(index), str(question), str(answer)))
    return examples


def select_examples(
    examples: list[Example], *, limit: int, offset: int, seed: int
) -> list[Example]:
    selected = examples[offset:]
    if limit >= len(selected):
        return selected
    indices = list(range(len(selected)))
    random.Random(seed).shuffle(indices)
    return [selected[index] for index in sorted(indices[:limit])]


def build_few_shot_prefix(examples: Iterable[Example], dataset: str) -> str:
    sections = [
        "Solve the final problem carefully. End with exactly `#### ANSWER`. "
        "Do not copy an answer from an example unless it is correct for the "
        "final problem.\n"
    ]
    for example in examples:
        answer = extract_answer(example.gold_text, dataset)
        sections.append(
            f"Example problem:\n{example.question}\n"
            f"Example final answer:\n#### {answer}\n"
        )
    return "\n".join(sections)


def make_prompt_ids(
    tokenizer: Any,
    example: Example,
    prefix: str,
    *,
    run_id: str,
    min_prompt_tokens: int,
) -> list[int]:
    marker = hashlib.sha256(f"{run_id}:{example.example_id}".encode()).hexdigest()
    prompt = (
        f"Evaluation case identifier: {marker}.\n{prefix}\n"
        f"Final problem:\n{example.question}\n"
        "Give the reasoning, then end with `#### ANSWER`."
    )
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("correctness evaluation only supports batch size 1")
        ids = ids[0]
    if len(ids) < min_prompt_tokens:
        raise ValueError(
            f"few-shot prompt has {len(ids)} tokens, below "
            f"--min-prompt-tokens={min_prompt_tokens}; increase --few-shot"
        )
    return list(ids)


def _importance(token_count: int) -> list[float]:
    # Stable request-global scores. Replace this function with ScoutRank output
    # when evaluating predictor quality rather than MaKV restore correctness.
    return [float((index * 37) % 1009) / 1009.0 for index in range(token_count)]


def prompt_token_hash(prompt_ids: list[int]) -> str:
    """Hash the exact prompt token IDs used by the vLLM request."""
    encoded = ",".join(str(int(value)) for value in prompt_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_importance_file(path: Optional[str]) -> dict[str, list[float]]:
    """Load prompt-hash keyed ScoutRank vectors from a JSON artifact."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = payload.get("scores", payload) if isinstance(payload, dict) else None
    if not isinstance(scores, dict):
        raise ValueError("importance file must contain a hash->score-list map")
    result: dict[str, list[float]] = {}
    for key, values in scores.items():
        if not isinstance(values, list):
            raise ValueError(f"importance scores for {key!r} must be a list")
        result[str(key)] = [float(value) for value in values]
    return result


def _nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def extract_cached_tokens(body: dict[str, Any]) -> tuple[int | None, str | None]:
    """Extract LMCache hit tokens from nested, direct, or vLLM usage fields."""
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


def request_completion(
    session: requests.Session,
    *,
    url: str,
    model: str,
    mode: str,
    prompt_ids: list[int],
    importance: list[float] | None,
    max_tokens: int,
    timeout: float,
    seed: int,
) -> dict[str, Any]:
    transfer_params: dict[str, Any] = {"cached_token_stats": True}
    if mode == "makv":
        transfer_params.update(
            {
                "lmcache.makv_importance": importance
                if importance is not None
                else _importance(len(prompt_ids)),
                "lmcache.makv_importance_layout": "token",
            }
        )
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "kv_transfer_params": transfer_params,
    }
    started = time.perf_counter()
    response = session.post(url, json=payload, timeout=timeout, stream=True)
    response.raise_for_status()
    text_parts: list[str] = []
    first_token_at: Optional[float] = None
    usage = None
    cached_tokens: int | None = None
    cached_tokens_source: str | None = None
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
        if body.get("usage") is not None:
            usage = body["usage"]
        value, source = extract_cached_tokens(body)
        if value is not None and (
            cached_tokens_source != "lmcache" or source == "lmcache"
        ):
            cached_tokens = value
            cached_tokens_source = source
    finished = time.perf_counter()
    return {
        "text": "".join(text_parts),
        "latency_ms": (finished - started) * 1000,
        "ttft_ms": (first_token_at - started) * 1000
        if first_token_at is not None
        else None,
        "cached_tokens": cached_tokens,
        "cached_tokens_source": cached_tokens_source,
        "usage": usage,
    }


def wait_for_complete_hit(
    session: requests.Session,
    *,
    args: argparse.Namespace,
    prompt_ids: list[int],
    importance: list[float] | None,
) -> dict[str, Any]:
    expected = (len(prompt_ids) // args.chunk_size) * args.chunk_size
    last: dict[str, Any] = {"cached_tokens": None}
    for attempt in range(args.hit_wait_retries):
        time.sleep(args.hit_wait_seconds)
        transfer_params: dict[str, Any] = {"cached_token_stats": True}
        if args.mode == "makv":
            transfer_params.update(
                {
                    "lmcache.makv_importance": importance
                    if importance is not None
                    else _importance(len(prompt_ids)),
                    "lmcache.makv_importance_layout": "token",
                }
            )
        response = session.post(
            args.url,
            json={
                "model": args.model,
                "prompt": prompt_ids,
                "max_tokens": 1,
                "temperature": 0.0,
                "seed": args.generation_seed,
                "stream": False,
                "kv_transfer_params": transfer_params,
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        body = response.json()
        cached_tokens, source = extract_cached_tokens(body)
        last = {"cached_tokens": cached_tokens, "cached_tokens_source": source}
        last["attempt"] = attempt + 1
        last["expected_cached_tokens"] = expected
        if cached_tokens is not None and cached_tokens >= expected:
            last["verification"] = "response_cache_stats"
            last["ready"] = True
            return last
    last["ready"] = False
    last["expected_cached_tokens"] = expected
    return last


def score_response(response: dict[str, Any], gold: Optional[str], dataset: str) -> None:
    response["answer"] = extract_answer(response["text"], dataset)
    response["correct"] = response["answer"] is not None and response["answer"] == gold


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["valid"]]
    cold_correct = sum(bool(record["cold"]["correct"]) for record in valid)
    hit_correct = sum(bool(record["hits"][0]["correct"]) for record in valid)
    answer_agreement = sum(
        record["cold"]["answer"] == record["hits"][0]["answer"] for record in valid
    )
    text_agreement = sum(
        record["cold"]["text"] == record["hits"][0]["text"] for record in valid
    )
    degraded = sum(
        bool(record["cold"]["correct"]) and not record["hits"][0]["correct"]
        for record in valid
    )
    denominator = len(valid)
    cold_ttfts = [
        record["cold"]["ttft_ms"]
        for record in valid
        if record["cold"]["ttft_ms"] is not None
    ]
    hit_ttfts = [
        record["hits"][0]["ttft_ms"]
        for record in valid
        if record["hits"][0]["ttft_ms"] is not None
    ]
    cold_latencies = [
        record["cold"]["latency_ms"]
        for record in valid
        if record["cold"]["latency_ms"] is not None
    ]
    hit_latencies = [
        record["hits"][0]["latency_ms"]
        for record in valid
        if record["hits"][0]["latency_ms"] is not None
    ]
    return {
        "total": len(records),
        "valid_complete_hits": denominator,
        "invalid_cache_runs": len(records) - denominator,
        "cold_accuracy": cold_correct / denominator if denominator else None,
        "hit_accuracy": hit_correct / denominator if denominator else None,
        "cold_hit_answer_agreement": answer_agreement / denominator
        if denominator
        else None,
        "cold_hit_text_exact_match": text_agreement / denominator
        if denominator
        else None,
        "cold_correct_to_hit_wrong": degraded,
        "cold_correct_retention": (cold_correct - degraded) / cold_correct
        if cold_correct
        else None,
        "cold_latency_mean_ms": (
            statistics.fmean(cold_latencies) if cold_latencies else None
        ),
        "cold_latency_median_ms": (
            statistics.median(cold_latencies) if cold_latencies else None
        ),
        "cold_latency_p95_ms": _p95(cold_latencies),
        "hit_latency_mean_ms": (
            statistics.fmean(hit_latencies) if hit_latencies else None
        ),
        "hit_latency_median_ms": (
            statistics.median(hit_latencies) if hit_latencies else None
        ),
        "hit_latency_p95_ms": _p95(hit_latencies),
        "cold_ttft_sample_count": len(cold_ttfts),
        "hit_ttft_sample_count": len(hit_ttfts),
        "cold_ttft_mean_ms": statistics.fmean(cold_ttfts) if cold_ttfts else None,
        "cold_ttft_median_ms": statistics.median(cold_ttfts) if cold_ttfts else None,
        "cold_ttft_p95_ms": _p95(cold_ttfts),
        "hit_ttft_mean_ms": statistics.fmean(hit_ttfts) if hit_ttfts else None,
        "hit_ttft_median_ms": statistics.median(hit_ttfts) if hit_ttfts else None,
        "hit_ttft_p95_ms": _p95(hit_ttfts),
    }


def _redis_payload_bytes(client: Any) -> tuple[int, int]:
    keys = list(client.scan_iter(match="*"))
    total = 0
    for key in keys:
        if client.type(key) == b"string":
            total += int(client.strlen(key))
    return len(keys), total


def _file_payload_bytes(path: Optional[str]) -> tuple[int, int]:
    if not path:
        return 0, 0
    files = [item for item in Path(path).rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _redis_client(url: Optional[str]) -> Any:
    if not url:
        return None
    try:
        import redis
    except ImportError as error:
        raise RuntimeError("--redis-url requires the redis Python package") from error
    return redis.Redis.from_url(url)


def run_evaluation(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path)
    examples = select_examples(
        load_examples(dataset_path, args.dataset, split="test"),
        limit=args.limit,
        offset=args.offset,
        seed=args.sample_seed,
    )
    train_examples = load_examples(
        Path(args.few_shot_path), args.dataset, split="train"
    )
    prefix = build_few_shot_prefix(train_examples[: args.few_shot], args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    importance_by_hash = load_importance_file(args.importance_file)
    if (
        args.mode == "makv"
        and args.require_importance_file
        and not args.importance_file
    ):
        raise ValueError("MaKV evaluation requires --importance-file")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix(".summary.json")
    redis_client = _redis_client(args.redis_url)
    if args.flush_redis_before:
        if redis_client is None:
            raise ValueError("--flush-redis-before requires --redis-url")
        redis_client.flushdb()

    records: list[dict[str, Any]] = []
    try:
        with requests.Session() as session, output.open("w", encoding="utf-8") as log:
            for ordinal, example in enumerate(examples):
                prompt_ids = make_prompt_ids(
                    tokenizer,
                    example,
                    prefix,
                    run_id=args.run_id,
                    min_prompt_tokens=args.min_prompt_tokens,
                )
                importance = None
                importance_source = None
                if args.mode == "makv":
                    importance = importance_by_hash.get(prompt_token_hash(prompt_ids))
                    if importance is None:
                        if args.require_importance_file:
                            raise ValueError(
                                "importance file has no vector for prompt hash "
                                f"{prompt_token_hash(prompt_ids)}"
                            )
                        importance = _importance(len(prompt_ids))
                        importance_source = "placeholder"
                    else:
                        importance_source = "scoutrank"
                    if len(importance) != len(prompt_ids):
                        raise ValueError(
                            "importance length does not match prompt token count: "
                            f"{len(importance)} != {len(prompt_ids)}"
                        )
                gold = extract_answer(example.gold_text, args.dataset)
                cold = request_completion(
                    session,
                    url=args.url,
                    model=args.model,
                    mode=args.mode,
                    prompt_ids=prompt_ids,
                    importance=importance,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    seed=args.generation_seed,
                )
                score_response(cold, gold, args.dataset)
                probe = wait_for_complete_hit(
                    session,
                    args=args,
                    prompt_ids=prompt_ids,
                    importance=importance,
                )
                hits = []
                for _ in range(args.hit_repeats):
                    hit = request_completion(
                        session,
                        url=args.url,
                        model=args.model,
                        mode=args.mode,
                        prompt_ids=prompt_ids,
                        importance=importance,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        seed=args.generation_seed,
                    )
                    score_response(hit, gold, args.dataset)
                    hits.append(hit)
                expected = probe["expected_cached_tokens"]
                valid = probe["ready"]
                record = {
                    "mode": args.mode,
                    "run_id": args.run_id,
                    "ordinal": ordinal,
                    "example_id": example.example_id,
                    "question": example.question,
                    "prompt_tokens": len(prompt_ids),
                    "importance_source": importance_source,
                    "gold": gold,
                    "cold": cold,
                    "probe": probe,
                    "hits": hits,
                    "valid": valid,
                }
                records.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(
                    f"[{ordinal + 1}/{len(examples)}] id={example.example_id} "
                    f"cold={cold['answer']} hit={hits[0]['answer']} gold={gold} "
                    f"cached={hits[0]['cached_tokens']}/{expected} valid={valid}",
                    flush=True,
                )
        summary = {
            "mode": args.mode,
            "run_id": args.run_id,
            "dataset": args.dataset,
            "dataset_path": str(dataset_path),
            "output": str(output),
            "importance_file": args.importance_file,
            "importance_vectors_available": len(importance_by_hash),
            **summarize_records(records),
        }
        if redis_client is not None:
            object_count, stored_bytes = _redis_payload_bytes(redis_client)
        else:
            object_count, stored_bytes = _file_payload_bytes(args.makv_storage_dir)
        cached_tokens = sum(
            int(record["probe"]["expected_cached_tokens"]) for record in records
        )
        raw_bytes_per_token = (
            args.model_layers
            * 2
            * args.model_kv_heads
            * args.model_head_dim
            * args.model_dtype_bytes
        )
        raw_kv_bytes = cached_tokens * raw_bytes_per_token
        summary.update(
            {
                "remote_object_count": object_count,
                "remote_stored_bytes": stored_bytes,
                "raw_kv_bytes": raw_kv_bytes,
                "compression_ratio": raw_kv_bytes / stored_bytes
                if stored_bytes
                else None,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        if args.flush_redis_after:
            if redis_client is None:
                raise ValueError("--flush-redis-after requires --redis-url")
            before = redis_client.dbsize()
            redis_client.flushdb()
            print(f"Redis cleanup: before={before}, after={redis_client.dbsize()}")


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[str(record["example_id"])] = record
    return records


def compare_results(args: argparse.Namespace) -> None:
    cachegen = _read_jsonl(Path(args.cachegen))
    makv = _read_jsonl(Path(args.makv))
    common = sorted(
        cachegen.keys() & makv.keys(),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if not common:
        raise ValueError("CacheGen and MaKV outputs have no common example IDs")
    rows = []
    for example_id in common:
        left, right = cachegen[example_id], makv[example_id]
        if left["run_id"] != right["run_id"]:
            raise ValueError(f"run_id mismatch for example {example_id}")
        rows.append(
            {
                "example_id": example_id,
                "gold": left["gold"],
                "cachegen_valid": left["valid"],
                "makv_valid": right["valid"],
                "cachegen_cold": left["cold"]["answer"],
                "makv_cold": right["cold"]["answer"],
                "cachegen_hit": left["hits"][0]["answer"],
                "makv_hit": right["hits"][0]["answer"],
                "cachegen_hit_correct": left["hits"][0]["correct"],
                "makv_hit_correct": right["hits"][0]["correct"],
            }
        )
    valid = [row for row in rows if row["cachegen_valid"] and row["makv_valid"]]
    summary = {
        "common_examples": len(rows),
        "valid_in_both": len(valid),
        "cold_answer_agreement_between_runs": sum(
            row["cachegen_cold"] == row["makv_cold"] for row in valid
        )
        / len(valid)
        if valid
        else None,
        "cachegen_makv_hit_answer_agreement": sum(
            row["cachegen_hit"] == row["makv_hit"] for row in valid
        )
        / len(valid)
        if valid
        else None,
        "cachegen_hit_accuracy": sum(row["cachegen_hit_correct"] for row in valid)
        / len(valid)
        if valid
        else None,
        "makv_hit_accuracy": sum(row["makv_hit_correct"] for row in valid) / len(valid)
        if valid
        else None,
        "disagreements": [
            row
            for row in valid
            if row["cachegen_hit"] != row["makv_hit"]
            or row["cachegen_hit_correct"] != row["makv_hit_correct"]
        ],
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run cold-vs-hit evaluation")
    run.add_argument(
        "--mode", choices=("naive", "cachegen", "makv"), required=True
    )
    run.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    run.add_argument("--model", default="qwen3-8b")
    run.add_argument("--tokenizer", default="/media/home/iic/mahaoyuan/models/Qwen3-8B")
    run.add_argument("--dataset", choices=("gsm8k", "math"), default="gsm8k")
    run.add_argument("--dataset-path", default=str(DEFAULT_GSM8K_TEST))
    run.add_argument("--few-shot-path", default=str(DEFAULT_GSM8K_TRAIN))
    run.add_argument("--few-shot", type=int, default=8)
    run.add_argument("--limit", type=int, default=32)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--sample-seed", type=int, default=20260811)
    run.add_argument("--generation-seed", type=int, default=123)
    run.add_argument("--run-id", required=True)
    run.add_argument("--min-prompt-tokens", type=int, default=512)
    run.add_argument("--chunk-size", type=int, default=256)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--hit-repeats", type=int, default=1)
    run.add_argument("--hit-wait-seconds", type=float, default=1.0)
    run.add_argument("--hit-wait-retries", type=int, default=10)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--output", required=True)
    run.add_argument("--importance-file")
    run.add_argument("--require-importance-file", action="store_true")
    run.add_argument("--redis-url")
    run.add_argument("--makv-storage-dir")
    run.add_argument("--model-layers", type=int, default=36)
    run.add_argument("--model-kv-heads", type=int, default=8)
    run.add_argument("--model-head-dim", type=int, default=128)
    run.add_argument("--model-dtype-bytes", type=int, default=2)
    run.add_argument("--flush-redis-before", action="store_true")
    run.add_argument("--flush-redis-after", action="store_true")
    run.set_defaults(function=run_evaluation)

    compare = subparsers.add_parser("compare", help="compare two run outputs")
    compare.add_argument("--cachegen", required=True)
    compare.add_argument("--makv", required=True)
    compare.add_argument("--output")
    compare.set_defaults(function=compare_results)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
