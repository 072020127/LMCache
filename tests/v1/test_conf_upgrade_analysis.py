# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.scoutrank_transfer.conf_controller import BlockOracle
from experiments.scoutrank_transfer.conf_upgrade_analysis import (
    _ranked_selection_row,
    build_upgrade_blocks,
    evaluate_selection,
)


COSTS = {
    "K2V2": 10,
    "K4V2": 20,
    "K8V4": 30,
    "BF16": 40,
}


def _oracle(block_id: int, *, conf: float, margin: float, kl: float) -> BlockOracle:
    key = ("sample", 1024, 1024, block_id)
    values = {
        "K2V2": (1.0, 0.4),
        "K4V2": (kl, kl / 2.0),
        "K8V4": (0.1, 0.05),
        "BF16": (0.0, 0.0),
        "MIXED": (0.2, 0.1),
    }
    return BlockOracle(
        key=key,
        sample=key[0],
        requested_context_length=key[1],
        context_length=key[2],
        block_id=block_id,
        token_count=32,
        scores={
            "conf_makv": conf,
            "margin_only": margin,
            "margin_p1": margin,
        },
        tokens_by_precision={
            precision: tuple(
                {
                    "kl_divergence": row[0],
                    "js_divergence": row[1],
                }
                for _ in range(32)
            )
            for precision, row in values.items()
        },
        alignment_hashes={
            "prefix_alignment_hash": "prefix",
            "suffix_alignment_hash": "suffix",
            "target_alignment_hash": "target",
        },
    )


def test_upgrade_benefit_uses_real_block_damage_and_excludes_bf16():
    oracles = [_oracle(0, conf=0.9, margin=0.1, kl=0.8)]
    records = build_upgrade_blocks(oracles, {oracles[0].key: "K4V2"}, COSTS)

    record = records[0]
    assert record["eligible_for_upgrade"] is True
    assert record["damage_base_kl"] == pytest.approx(25.6)
    assert record["damage_upgrade_kl"] == pytest.approx(3.2)
    assert record["upgrade_benefit_kl"] == pytest.approx(22.4)
    assert record["added_bytes_if_upgraded"] == 10


def test_same_rate_margin_ranking_selects_exactly_conf_count():
    first = _oracle(0, conf=0.9, margin=0.1, kl=0.8)
    second = _oracle(1, conf=0.1, margin=0.9, kl=0.2)
    base_plan = {first.key: "K4V2", second.key: "K4V2"}
    records = build_upgrade_blocks([first, second], base_plan, COSTS)

    row = _ranked_selection_row(records, COSTS, "margin_only", 1)

    assert row["same_rate_exact"] is True
    assert row["upgrade_block_count"] == 1
    assert row["selected_block_keys"] == [["sample", 1024, 1024, 1]]


def test_dangerous_block_definition_is_top_twenty_percent():
    oracles = [
        _oracle(index, conf=index / 10.0, margin=index / 10.0, kl=0.2 + index / 10.0)
        for index in range(10)
    ]
    records = build_upgrade_blocks(
        oracles,
        {oracle.key: "K4V2" for oracle in oracles},
        COSTS,
    )

    assert sum(record["dangerous_block"] for record in records) == 2


def test_evaluate_selection_rejects_non_upgradeable_block():
    oracle = _oracle(0, conf=0.9, margin=0.9, kl=0.8)
    records = build_upgrade_blocks([oracle], {oracle.key: "BF16"}, COSTS)

    with pytest.raises(ValueError, match="non-upgradeable"):
        evaluate_selection(
            records,
            [oracle.key],
            COSTS,
            method="conf",
        )
