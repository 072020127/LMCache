# SPDX-License-Identifier: Apache-2.0

# Standard
import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[2] / "benchmarks" / "makv_cachegen_correctness.py"
SPEC = importlib.util.spec_from_file_location("makv_cachegen_correctness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_make_prompt_ids_accepts_batch_encoding_mapping():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 1
            return {"input_ids": list(range(512)), "attention_mask": [1] * 512}

    example = MODULE.Example("0", "What is 1 + 1?", "#### 2")
    ids = MODULE.make_prompt_ids(
        Tokenizer(),
        example,
        "example",
        run_id="test",
        min_prompt_tokens=512,
    )
    assert ids == list(range(512))


def test_extract_gsm8k_answer_prefers_explicit_final_answer():
    assert MODULE.extract_answer("work 12, then #### 18", "gsm8k") == "18"
    assert MODULE.extract_answer("Final answer: $70,000", "gsm8k") == "70000"
    assert MODULE.extract_answer("therefore \\boxed{-7}", "gsm8k") == "-7"


def test_extract_math_boxed_answer_handles_nested_braces():
    assert MODULE.extract_answer(r"result is \boxed{\frac{1}{2}}", "math") == (
        r"\frac{1}{2}"
    )


def test_summarize_records_reports_correctness_degradation():
    records = [
        {
            "valid": True,
            "cold": {
                "correct": True,
                "answer": "18",
                "text": "a",
                "latency_ms": 2,
                "ttft_ms": 1,
            },
            "hits": [
                {
                    "correct": False,
                    "answer": "19",
                    "text": "b",
                    "latency_ms": 1,
                    "ttft_ms": 0.5,
                }
            ],
        },
        {
            "valid": True,
            "cold": {
                "correct": True,
                "answer": "3",
                "text": "c",
                "latency_ms": 4,
                "ttft_ms": 2,
            },
            "hits": [
                {
                    "correct": True,
                    "answer": "3",
                    "text": "c",
                    "latency_ms": 3,
                    "ttft_ms": 1,
                }
            ],
        },
        {
            "valid": False,
            "cold": {
                "correct": True,
                "answer": "1",
                "text": "x",
                "latency_ms": 1,
                "ttft_ms": 1,
            },
            "hits": [
                {
                    "correct": False,
                    "answer": "2",
                    "text": "y",
                    "latency_ms": 1,
                    "ttft_ms": 1,
                }
            ],
        },
    ]
    summary = MODULE.summarize_records(records)
    assert summary["valid_complete_hits"] == 2
    assert summary["invalid_cache_runs"] == 1
    assert summary["cold_accuracy"] == 1.0
    assert summary["hit_accuracy"] == 0.5
    assert summary["cold_correct_to_hit_wrong"] == 1
    assert summary["cold_correct_retention"] == 0.5
    assert summary["cold_ttft_mean_ms"] == 1.5
    assert summary["cold_ttft_median_ms"] == 1.5
    assert summary["cold_ttft_p95_ms"] == 2
    assert summary["hit_ttft_mean_ms"] == 0.75
    assert summary["hit_latency_mean_ms"] == 2
    assert summary["hit_latency_p95_ms"] == 3
