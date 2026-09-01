#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Precompute real ScoutRank token importance for GSM8K or MATH prompts."""

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import argparse
import gc
import json

# Third Party
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# First Party
from experiments.scoutrank_transfer.observer import (
    ProductionMaKVErrorObserver,
    VectorizedMaKVErrorObserver,
)
from makv_scoutrank import ScoutForwardAdapter, ScoutRankConfig, ScoutRankScorer

try:
    from benchmarks.makv_cachegen_correctness import (
        build_few_shot_prefix,
        load_examples,
        make_prompt_ids,
        prompt_token_hash,
        select_examples,
    )
except ImportError:
    from makv_cachegen_correctness import (
        build_few_shot_prefix,
        load_examples,
        make_prompt_ids,
        prompt_token_hash,
        select_examples,
    )


def _load_model(path: str, device: torch.device, dtype: torch.dtype) -> Any:
    """Load the frozen ScoutRank model on the requested device."""
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device).eval()


def _parse_anchor_layers(
    value: str | None, mode: str, exit_layer: int | None = None
) -> tuple[int, ...]:
    """Parse the anchor policy, using the cheaper two-layer fast default."""
    if value:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        if not layers:
            raise ValueError("--anchor-layers must contain at least one layer")
        return layers
    if exit_layer is not None:
        layers = tuple(layer for layer in (7, 14, 21, 28) if layer <= exit_layer)
        if not layers:
            raise ValueError(
                "--exit-layer must be at least 7 when --anchor-layers is omitted"
            )
        return layers
    if mode == "fast":
        return (14, 28)
    return (7, 14, 21, 28)


def _build_scout_runtime(
    model: Any,
    *,
    mode: str,
    observer_backend: str,
    observer_token_chunk_size: int,
    anchor_layers: tuple[int, ...],
    exit_layer: int | None,
) -> tuple[ScoutForwardAdapter, ScoutRankScorer]:
    """Build reusable ScoutRank state for all prompts in one artifact."""
    config = ScoutRankConfig(
        mode=mode,
        anchor_layers=anchor_layers,
        exit_layer=exit_layer,
        block_size=32,
        include_token_scores=False,
        target_layers=36,
        target_kv_heads=8,
        target_head_dim=128,
        output_attentions=False,
        use_cache=False,
    )
    if observer_backend == "vectorized":
        observer = VectorizedMaKVErrorObserver(observer_token_chunk_size)
    elif observer_backend == "production":
        observer = ProductionMaKVErrorObserver()
    else:
        raise ValueError(f"unsupported observer backend: {observer_backend}")
    return ScoutForwardAdapter(model, config, observer), ScoutRankScorer(config)


def _score_prompt(
    ids: list[int],
    *,
    device: torch.device,
    adapter: ScoutForwardAdapter,
    scorer: ScoutRankScorer,
) -> list[float]:
    """Run one prompt through reusable ScoutRank state."""
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    forward = adapter.forward_once(
        input_ids, compute_nll=adapter.cfg.mode != "fast"
    )
    if forward.summary is None:
        raise RuntimeError("ScoutRank forward did not produce a summary")
    summary = forward.summary
    scores = scorer.token_importance_from_summaries(
        token_ids=summary.token_ids,
        valid_token_mask=summary.valid_token_mask,
        self_information_nll=summary.self_information_nll,
        representation_drift=summary.representation_drift,
        local_novelty=summary.local_novelty,
        task_relevance=summary.task_relevance,
        k_errors=summary.k_errors,
        v_errors=summary.v_errors,
    )
    if scores.numel() != len(ids):
        raise RuntimeError("ScoutRank token score count does not match prompt")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("ScoutRank produced a non-finite token score")
    return scores.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3-0.6B ScoutRank model")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--few-shot-path", required=True)
    parser.add_argument("--few-shot", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260811)
    parser.add_argument("--prompt-run-id", required=True)
    parser.add_argument("--min-prompt-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--mode", choices=("fast", "balanced"), default="balanced")
    parser.add_argument(
        "--observer-backend",
        choices=("vectorized", "production"),
        default="vectorized",
        help="Use GPU vectorized production-math errors or the legacy CPU round trip.",
    )
    parser.add_argument(
        "--observer-token-chunk-size",
        type=int,
        default=4096,
        help="Token chunk size for vectorized observer temporary tensors.",
    )
    parser.add_argument(
        "--anchor-layers",
        default=None,
        help=(
            "Comma-separated 1-based Scout model anchor layers; "
            "fast defaults to 14,28."
        ),
    )
    parser.add_argument(
        "--exit-layer",
        type=int,
        default=None,
        help="Optional Qwen3 early-exit depth; anchors must not exceed it.",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    examples = select_examples(
        load_examples(Path(args.dataset_path), args.dataset, split="test"),
        limit=args.limit,
        offset=args.offset,
        seed=args.sample_seed,
    )
    train_examples = load_examples(
        Path(args.few_shot_path), args.dataset, split="train"
    )
    prefix = build_few_shot_prefix(train_examples[: args.few_shot], args.dataset)
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = _load_model(args.model, device, dtype)
    anchor_layers = _parse_anchor_layers(
        args.anchor_layers, args.mode, args.exit_layer
    )
    adapter, scorer = _build_scout_runtime(
        model,
        mode=args.mode,
        observer_backend=args.observer_backend,
        observer_token_chunk_size=args.observer_token_chunk_size,
        anchor_layers=anchor_layers,
        exit_layer=args.exit_layer,
    )
    scores: dict[str, list[float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    try:
        for index, example in enumerate(examples, start=1):
            ids = make_prompt_ids(
                tokenizer,
                example,
                prefix,
                run_id=args.prompt_run_id,
                min_prompt_tokens=args.min_prompt_tokens,
            )
            key = prompt_token_hash(ids)
            values = _score_prompt(
                ids, device=device, adapter=adapter, scorer=scorer
            )
            scores[key] = values
            metadata[key] = {
                "example_id": example.example_id,
                "token_count": len(ids),
                "importance_layout": "token",
                "score_semantics": "ScoutRank damage_22; higher means more important",
                "scoring_mode": args.mode,
                "observer_backend": args.observer_backend,
                "anchor_layers": list(anchor_layers),
                "exit_layer": args.exit_layer,
            }
            print(
                f"[{index}/{len(examples)}] id={example.example_id} "
                f"tokens={len(ids)} score_range=({min(values):.6g},{max(values):.6g})",
                flush=True,
            )
    finally:
        del adapter, scorer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "prompt_run_id": args.prompt_run_id,
        "scout_model": args.model,
        "few_shot": args.few_shot,
        "sample_seed": args.sample_seed,
        "importance_layout": "token",
        "score_semantics": "ScoutRank damage_22; higher means more important",
        "scoring_mode": args.mode,
        "observer_backend": args.observer_backend,
        "observer_token_chunk_size": args.observer_token_chunk_size,
        "anchor_layers": list(anchor_layers),
        "exit_layer": args.exit_layer,
        "metadata": metadata,
        "scores": scores,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(scores)} importance vectors to {output}")


if __name__ == "__main__":
    main()
