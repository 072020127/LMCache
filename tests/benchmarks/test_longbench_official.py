# SPDX-License-Identifier: Apache-2.0

# Standard
from benchmarks.longbench_official.adapter import official_score, score_records


def test_hotpotqa_uses_official_token_f1():
    score = official_score("Canberra city", ("Canberra",), "hotpotqa")
    assert score is not None
    assert abs(score - (2.0 / 3.0)) < 1e-9


def test_classification_uses_all_classes_and_first_line():
    assert official_score(
        "C2\nadditional explanation", ("C2",), "trec", ("C1", "C2", "C3")
    ) == 1.0


def test_e_task_uses_the_base_task_metric():
    assert official_score("Canberra", ("Canberra",), "hotpotqa_e") == 1.0


def test_score_records_only_scores_complete_hits_by_default():
    records = [
        {
            "example_id": "complete",
            "answers": ["Canberra"],
            "valid": True,
            "cold": {"text": "Canberra city"},
            "hit": {"text": "Canberra"},
        },
        {
            "example_id": "incomplete",
            "answers": ["Canberra"],
            "valid": False,
            "cold": {"text": "Paris"},
            "hit": {"text": "Paris"},
        },
    ]
    result = score_records(records, "hotpotqa")
    assert result["valid_complete_hits"] == 1
    assert result["cold_scored_samples"] == 1
    assert result["hit_scored_samples"] == 1
    assert result["cold_score_percent"] == 66.67
    assert result["hit_score_percent"] == 100.0
