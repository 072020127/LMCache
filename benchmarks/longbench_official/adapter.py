# SPDX-License-Identifier: Apache-2.0
"""Adapt the official LongBench metrics to LMCache benchmark JSONL files.

The upstream evaluator expects one flat prediction file per task.  The
LMCache runner stores cold and cache-hit generations in one record so that
latency and cache validity can be analyzed together.  This module keeps the
upstream metric implementations unchanged and only translates between those
two record formats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

try:
    from .metrics import (
        classification_score,
        code_sim_score,
        count_score,
        qa_f1_score,
        qa_f1_zh_score,
        retrieval_score,
        retrieval_zh_score,
        rouge_score,
        rouge_zh_score,
    )
except ImportError:  # Allow ``python adapter.py`` from this directory.
    from metrics import (  # type: ignore[no-redef]
        classification_score,
        code_sim_score,
        count_score,
        qa_f1_score,
        qa_f1_zh_score,
        retrieval_score,
        retrieval_zh_score,
        rouge_score,
        rouge_zh_score,
    )


OFFICIAL_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"

Metric = Callable[..., float]

# This table intentionally mirrors LongBench/LongBench/eval.py.  The ``_e``
# suffix is handled by ``base_task`` because LongBench-E uses the same metric
# with an additional context-length breakdown.
DATASET2METRIC: dict[str, Metric] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

METRIC_NAMES: dict[Metric, str] = {
    qa_f1_score: "qa_f1",
    qa_f1_zh_score: "qa_f1_zh",
    rouge_score: "rouge_l",
    rouge_zh_score: "rouge_l_zh",
    classification_score: "classification_accuracy",
    retrieval_score: "retrieval_accuracy",
    retrieval_zh_score: "retrieval_accuracy_zh",
    count_score: "count_accuracy",
    code_sim_score: "edit_similarity",
}

_FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def base_task(task: str) -> str:
    """Return the non-E LongBench task name for ``task``."""
    normalized = task.strip().lower()
    return normalized[:-2] if normalized.endswith("_e") else normalized


def _metric_for(task: str) -> tuple[str, Metric]:
    dataset = base_task(task)
    try:
        return dataset, DATASET2METRIC[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET2METRIC))
        raise ValueError(
            f"Unsupported LongBench task {task!r}; supported tasks: {supported}"
        ) from exc


def metric_name(task: str) -> str:
    """Return the official metric name used for a LongBench task."""
    _, metric = _metric_for(task)
    return METRIC_NAMES[metric]


def official_score(
    prediction: str,
    ground_truths: Sequence[str],
    task: str,
    all_classes: Sequence[str] | None = None,
) -> float | None:
    """Score one prediction using the upstream LongBench implementation.

    The return value is in the upstream internal range ``[0, 1]``.  The
    official command-line evaluator multiplies the mean by 100 and rounds to
    two decimal places; :func:`score_records` exposes both representations.
    """
    dataset, metric = _metric_for(task)
    if not ground_truths:
        return None

    if dataset in _FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]

    classes = list(all_classes or [])
    if metric is classification_score and not classes:
        raise ValueError(
            f"LongBench task {task!r} requires its all_classes metadata"
        )

    best = 0.0
    for ground_truth in ground_truths:
        best = max(
            best,
            float(metric(prediction, ground_truth, all_classes=classes)),
        )
    return best


def _metadata_by_id(dataset_path: Path | None) -> dict[str, dict[str, Any]]:
    if dataset_path is None:
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            example_id = str(item.get("_id", index))
            metadata[example_id] = {
                "all_classes": item.get("all_classes") or [],
                "length": item.get("length"),
            }
    return metadata


def _record_score(
    record: dict[str, Any],
    key: str,
    task: str,
    metadata: dict[str, Any] | None = None,
) -> float | None:
    block = record.get(key)
    if not isinstance(block, dict):
        return None
    existing = block.get("official_score")
    if isinstance(existing, (int, float)) and not isinstance(existing, bool):
        return float(existing)

    answers = record.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, Sequence):
        answers = []
    classes = record.get("all_classes")
    if not classes and metadata:
        classes = metadata.get("all_classes")
    text = block.get("text", "")
    return official_score(
        str(text),
        tuple(str(answer) for answer in answers),
        task,
        tuple(str(value) for value in (classes or [])),
    )


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    materialized = [value for value in values if value is not None]
    return mean(materialized) if materialized else None


def _percent(value: float | None) -> float | None:
    return round(100.0 * value, 2) if value is not None else None


def score_records(
    records: Sequence[dict[str, Any]],
    task: str,
    *,
    dataset_path: Path | None = None,
    only_valid: bool = True,
) -> dict[str, Any]:
    """Score cold and hit generations in the LMCache JSONL schema."""
    dataset, _ = _metric_for(task)
    metadata = _metadata_by_id(dataset_path)
    selected = [
        record
        for record in records
        if not only_valid or bool(record.get("valid", False))
    ]

    cold_scores: list[float] = []
    hit_scores: list[float] = []
    length_scores: dict[str, dict[str, list[float]]] = {
        "0-4k": {"cold": [], "hit": []},
        "4-8k": {"cold": [], "hit": []},
        "8k+": {"cold": [], "hit": []},
    }
    for record in selected:
        record_metadata = metadata.get(str(record.get("example_id", "")), {})
        cold = _record_score(record, "cold", task, record_metadata)
        hit = _record_score(record, "hit", task, record_metadata)
        if cold is not None:
            cold_scores.append(cold)
        if hit is not None:
            hit_scores.append(hit)

        length = record.get("length", record_metadata.get("length"))
        if isinstance(length, (int, float)):
            bucket = "0-4k" if length < 4000 else "4-8k" if length < 8000 else "8k+"
            if cold is not None:
                length_scores[bucket]["cold"].append(cold)
            if hit is not None:
                length_scores[bucket]["hit"].append(hit)

    cold_score = _mean_or_none(cold_scores)
    hit_score = _mean_or_none(hit_scores)
    result: dict[str, Any] = {
        "evaluator": "longbench_official",
        "official_commit": OFFICIAL_COMMIT,
        "task": task,
        "base_task": dataset,
        "metric": metric_name(task),
        "metric_scale": "0-100",
        "total": len(records),
        "valid_complete_hits": len(selected),
        "cold_scored_samples": len(cold_scores),
        "hit_scored_samples": len(hit_scores),
        "cold_score": cold_score,
        "hit_score": hit_score,
        "cold_score_percent": _percent(cold_score),
        "hit_score_percent": _percent(hit_score),
        "score_delta_percent": (
            round(100.0 * (hit_score - cold_score), 2)
            if cold_score is not None and hit_score is not None
            else None
        ),
    }
    if task.strip().lower().endswith("_e") and any(
        bucket["cold"] or bucket["hit"] for bucket in length_scores.values()
    ):
        result["length_bucket_scores_percent"] = {
            bucket: {
                "cold": _percent(_mean_or_none(values["cold"])),
                "hit": _percent(_mean_or_none(values["hit"])),
            }
            for bucket, values in length_scores.items()
        }
    return result


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    """Run official scoring on an LMCache benchmark JSONL file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include records without a complete cache hit in the score.",
    )
    args = parser.parse_args()
    result = score_records(
        _load_records(args.input),
        args.task,
        dataset_path=args.dataset_path,
        only_valid=not args.include_invalid,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
