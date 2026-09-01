# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the optional MaKV PrecisionRiskObserver call.

This benchmark times only the per-token observer boundary on pre-generated
logits.  It does not run a model, modify a KV cache, select precision, or
represent end-to-end model TPOT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import math
import time

import torch

from lmcache.v1.storage_backend.makv.precision_risk import (
    CONF_SCORER_VERSION,
    compute_precision_risk_signal,
)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _make_logits(
    *,
    tokens: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (tokens, vocab_size),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def _run_condition(
    logits: torch.Tensor,
    *,
    enabled: bool,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    token_count = int(logits.shape[0])
    warmup_count = min(warmup, token_count)
    with torch.inference_mode():
        for step in range(warmup_count):
            if enabled:
                compute_precision_risk_signal(logits[step], step=step)
        _synchronize(device)

        latencies_ms: list[float] = []
        started = time.perf_counter()
        valid_count = 0
        for step in range(token_count):
            iteration_started = time.perf_counter()
            if enabled:
                signal = compute_precision_risk_signal(logits[step], step=step)
                valid_count += int(signal.valid)
            # Use the same synchronization boundary for both conditions so the
            # disabled baseline includes no hidden asynchronous work advantage.
            _synchronize(device)
            latencies_ms.append((time.perf_counter() - iteration_started) * 1000.0)
        elapsed_s = time.perf_counter() - started

    return {
        "scorer_enabled": enabled,
        "tokens": token_count,
        "warmup_tokens": warmup_count,
        "scorer_latency_mean_ms": (
            sum(latencies_ms) / len(latencies_ms) if enabled else 0.0
        ),
        "scorer_latency_p95_ms": _p95(latencies_ms) if enabled else 0.0,
        "observer_iteration_mean_ms": sum(latencies_ms) / len(latencies_ms),
        "observer_iteration_p95_ms": _p95(latencies_ms),
        "tpot_ms_per_token": elapsed_s * 1000.0 / token_count,
        "tokens_per_s": token_count / elapsed_s if elapsed_s > 0.0 else 0.0,
        "valid_signals": valid_count,
    }


def run_precision_risk_benchmark(
    *,
    device: str = "auto",
    tokens: int = 128,
    vocab_size: int = 151936,
    warmup: int = 16,
    seed: int = 17,
) -> dict[str, Any]:
    """Run disabled/enabled observer microbenchmarks on identical logits."""
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    resolved_device = _resolve_device(device)
    logits = _make_logits(
        tokens=tokens,
        vocab_size=vocab_size,
        device=resolved_device,
        seed=seed,
    )
    disabled = _run_condition(
        logits,
        enabled=False,
        warmup=warmup,
        device=resolved_device,
    )
    enabled = _run_condition(
        logits,
        enabled=True,
        warmup=warmup,
        device=resolved_device,
    )
    return {
        "benchmark": "makv_precision_risk_observer",
        "scope": "observer_only_pre_generated_logits",
        "scorer_version": CONF_SCORER_VERSION,
        "device": str(resolved_device),
        "tokens": tokens,
        "vocab_size": vocab_size,
        "warmup_tokens": warmup,
        "seed": seed,
        "disabled": disabled,
        "enabled": enabled,
        "enabled_minus_disabled_tpot_ms_per_token": (
            enabled["tpot_ms_per_token"] - disabled["tpot_ms_per_token"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_precision_risk_benchmark(
        device=args.device,
        tokens=args.tokens,
        vocab_size=args.vocab_size,
        warmup=args.warmup,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
