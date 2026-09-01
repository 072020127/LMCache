# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[2] / "benchmarks" / "longbench_makv_cachegen.py"
SPEC = importlib.util.spec_from_file_location("longbench_makv_cachegen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_longbench_answer_hit_is_normalized():
    assert MODULE.score("The answer is Canberra.", ("Canberra",), "triviaqa")
    assert MODULE.score("C2", ("c2",), "trec")
    assert not MODULE.score("The answer is Paris.", ("Canberra",), "triviaqa")


def test_extract_answer_text_drops_qwen_thinking_trace():
    assert MODULE.extract_answer_text("<think>work</think> Canberra") == "Canberra"
    assert MODULE.extract_answer_text("<think>unfinished reasoning") == ""
    assert MODULE.extract_answer_text("Canberra") == "Canberra"


def test_prompt_ids_passes_answer_only_to_chat_template():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return {"input_ids": [1, 2, 3]}

    example = MODULE.LongBenchExample(
        "id", "context", "question", ("answer",), "hotpotqa"
    )
    assert MODULE.prompt_ids(Tokenizer(), example, "run") == [1, 2, 3]
    assert calls[0]["enable_thinking"] is False


def test_importance_file_is_keyed_by_prompt_hash(tmp_path):
    path = tmp_path / "importance.json"
    path.write_text('{"scores":{"abc":[0.1,0.2]}}', encoding="utf-8")
    assert MODULE.load_importance_file(str(path)) == {"abc": [0.1, 0.2]}


def test_load_importance_timing_reads_per_prompt_metadata(tmp_path):
    path = tmp_path / "importance.json"
    path.write_text(
        '{"scores":{"abc":[0.1,0.2]},'
        '"metadata":{"abc":{"scoutrank_time_ms":12.5}}}',
        encoding="utf-8",
    )
    assert MODULE.load_importance_timing(str(path)) == {"abc": 12.5}


def test_longbench_summarization_has_no_fake_exact_accuracy():
    assert MODULE.score("a summary", ("reference",), "gov_report") is None


def test_longbench_loader_reads_official_fields(tmp_path):
    path = tmp_path / "hotpotqa.jsonl"
    path.write_text(
        '{"_id":"x","context":"doc","input":"question",'
        '"answers":["answer"],"length":17,"all_classes":["A","B"]}\n',
        encoding="utf-8",
    )
    rows = MODULE.load_examples(path, "hotpotqa", 1, 0)
    assert rows[0].example_id == "x"
    assert rows[0].context == "doc"
    assert rows[0].answers == ("answer",)
    assert rows[0].length == 17
    assert rows[0].all_classes == ("A", "B")


def test_extract_cached_tokens_accepts_all_response_shapes():
    assert MODULE.extract_cached_tokens(
        {
            "kv_transfer_params": {
                "cached_token_stats": {"num_lmcache_cached_tokens": 256}
            }
        }
    ) == (256, "lmcache")
    assert MODULE.extract_cached_tokens(
        {"kv_transfer_params": {"num_lmcache_cached_tokens": 128}}
    ) == (128, "lmcache")
    assert MODULE.extract_cached_tokens(
        {"usage": {"prompt_tokens_details": {"cached_tokens": 64}}}
    ) == (64, "vllm_usage")
    assert MODULE.extract_cached_tokens(
        {"kv_transfer_params": {"num_lmcache_cached_tokens": -1}}
    ) == (None, None)
    assert MODULE.extract_cached_tokens({}) == (None, None)


def test_makv_manager_timing_summary_reports_quantizer_share():
    summary = MODULE.makv_manager_timing_summary(
        {
            "metrics": {
                "makv_remote_put_requests": 3,
                "makv_raw_input_bytes": 1200,
                "makv_stored_bytes": 400,
                "makv_remote_put_total_time_ms": 100.0,
                "makv_remote_put_decode_time_ms": 5.0,
                "makv_remote_quantize_time_ms": 60.0,
                "makv_remote_quantize_kernel_time_ms": 50.0,
                "makv_remote_encode_validate_time_ms": 20.0,
                "makv_remote_storage_put_time_ms": 15.0,
                "makv_remote_get_requests": 4,
                "makv_remote_get_total_time_ms": 12.0,
                "makv_remote_get_storage_time_ms": 4.0,
                "makv_remote_get_validate_time_ms": 2.0,
            }
        }
    )
    assert summary["put_requests"] == 3
    assert summary["compression_ratio"] == 3.0
    assert summary["remote_quantize_core_share_of_put_pct"] == 50.0


def test_summarize_exposes_official_scores_without_removing_legacy_scores():
    records = [
        {
            "valid": True,
            "answers": ["Canberra"],
            "all_classes": [],
            "prompt_tokens": 4,
            "scoutrank_time_ms": 2.0,
            "cold": {
                "correct": True,
                "official_score": 2.0 / 3.0,
                "text": "Canberra city",
                "ttft_ms": 1.0,
                "ttft_with_scoutrank_ms": 3.0,
                "latency_ms": 2.0,
            },
            "hit": {
                "correct": True,
                "official_score": 1.0,
                "text": "Canberra",
                "ttft_ms": 0.5,
                "ttft_with_scoutrank_ms": 2.5,
                "latency_ms": 1.0,
            },
        }
    ]
    args = argparse.Namespace(
        mode="cachegen",
        task="hotpotqa",
        evaluator="official",
        model_layers=1,
        model_kv_heads=1,
        model_head_dim=1,
        model_dtype_bytes=2,
        redis_url=None,
        storage_dir=None,
    )
    summary = MODULE.summarize(records, args)
    assert summary["cold_accuracy"] == 2.0 / 3.0
    assert summary["hit_accuracy"] == 1.0
    assert summary["legacy_cold_accuracy"] == 1.0
    assert summary["official_metric"] == "qa_f1"
    assert summary["hit_official_score_percent"] == 100.0
    assert summary["cold_ttft_mean_ms"] == 1.0
    assert summary["cold_ttft_p95_ms"] == 1.0
    assert summary["hit_ttft_mean_ms"] == 0.5
    assert summary["scoutrank_time_mean_ms"] == 2.0
    assert summary["cold_ttft_with_scoutrank_mean_ms"] == 3.0
    assert summary["hit_ttft_with_scoutrank_mean_ms"] == 2.5
    assert round(summary["scoutrank_share_of_cold_ttft_pct"], 6) == 66.666667
    assert summary["hit_latency_mean_ms"] == 1.0
    assert summary["hit_latency_p95_ms"] == 1.0
