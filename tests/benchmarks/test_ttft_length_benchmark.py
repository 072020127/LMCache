# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the synthetic prompt-length TTFT benchmark helpers."""

from __future__ import annotations

# First Party
from benchmarks.ttft_length_benchmark import (
    build_prompt_ids,
    parse_context_lengths,
    summarize_samples,
)


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        base = sum(ord(character) for character in text) % 100000
        return [base + index for index, _ in enumerate(text.split())]


def test_parse_context_lengths_accepts_binary_suffixes() -> None:
    assert parse_context_lengths("1k,2k,4k,16k,32k,64k,128k") == (
        1024,
        2048,
        4096,
        16384,
        32768,
        65536,
        131072,
    )


def test_build_prompt_ids_returns_exact_length_and_unique_prefixes() -> None:
    tokenizer = _FakeTokenizer()
    first = build_prompt_ids(tokenizer, 1024, "run:length=1024:rep=0")
    second = build_prompt_ids(tokenizer, 2048, "run:length=2048:rep=0")

    assert len(first) == 1024
    assert len(second) == 2048
    assert first[:10] != second[:10]


def test_summarize_samples_reports_mean_median_and_p95() -> None:
    samples = [
        {
            "valid_hit": True,
            "cold": {"ttft_ms": 10.0, "latency_ms": 20.0, "cached_tokens": 100},
            "hit": {"ttft_ms": 5.0, "latency_ms": 12.0, "cached_tokens": 90},
        },
        {
            "valid_hit": True,
            "cold": {"ttft_ms": 20.0, "latency_ms": 30.0, "cached_tokens": 110},
            "hit": {"ttft_ms": 10.0, "latency_ms": 14.0, "cached_tokens": 100},
        },
        {
            "valid_hit": True,
            "cold": {"ttft_ms": 100.0, "latency_ms": 120.0, "cached_tokens": 120},
            "hit": {"ttft_ms": 50.0, "latency_ms": 60.0, "cached_tokens": 110},
        },
    ]
    row = summarize_samples(
        mode="naive",
        prompt_tokens=1024,
        max_tokens=1,
        repetitions=3,
        expected_cached_tokens=1024,
        samples=samples,
        errors=(),
    )

    assert row["cold_ttft_mean_ms"] == "43.333333"
    assert row["cold_ttft_median_ms"] == "20.000000"
    assert row["cold_ttft_p95_ms"] == "100.000000"
    assert row["hit_ttft_mean_ms"] == "21.666667"
    assert row["hit_latency_mean_ms"] == "28.666667"
    assert row["hit_latency_p95_ms"] == "60.000000"
    assert row["cold_cached_tokens_mean"] == "110.000000"
