"""The DAPD training objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Completion = Literal["reference", "rollout"]
Condition = Literal["none", "reference", "rollout"]
Anchor = Literal["base", "snapshot"]


@dataclass(frozen=True)
class Pair:
    name: str
    completion: Completion
    left_condition: Condition
    right_condition: Condition
    anchor: Anchor
    weight: float


# Main DAPD coefficients.
DAPD_WEIGHTS = {
    "entangled_rollout": 2.0 / 15.0,
    "inference_reference": 2.0 / 15.0,
    "privileged_rollout": 2.0 / 15.0,
    "entangled_reference": 2.0 / 5.0,
    "inference_rollout": 2.0 / 5.0,
    "privileged_reference": 4.0 / 5.0,
}


PAIRS = (
    Pair(
        "entangled_rollout",
        "rollout",
        "reference",
        "none",
        "snapshot",
        DAPD_WEIGHTS["entangled_rollout"],
    ),
    Pair(
        "inference_reference",
        "reference",
        "none",
        "reference",
        "base",
        DAPD_WEIGHTS["inference_reference"],
    ),
    Pair(
        "privileged_rollout",
        "rollout",
        "reference",
        "rollout",
        "base",
        DAPD_WEIGHTS["privileged_rollout"],
    ),
    Pair(
        "entangled_reference",
        "reference",
        "rollout",
        "none",
        "snapshot",
        DAPD_WEIGHTS["entangled_reference"],
    ),
    Pair(
        "inference_rollout",
        "rollout",
        "none",
        "rollout",
        "base",
        DAPD_WEIGHTS["inference_rollout"],
    ),
    Pair(
        "privileged_reference",
        "reference",
        "rollout",
        "reference",
        "base",
        DAPD_WEIGHTS["privileged_reference"],
    ),
)


def validate_objective() -> None:
    names = [pair.name for pair in PAIRS]
    expected = [
        "entangled_rollout",
        "inference_reference",
        "privileged_rollout",
        "entangled_reference",
        "inference_rollout",
        "privileged_reference",
    ]
    if names != expected:
        raise RuntimeError(f"unexpected DAPD pair order: {names}")
    if abs(sum(DAPD_WEIGHTS.values()) - 2.0) > 1e-12:
        raise RuntimeError("DAPD KL weights must sum to 2")
    snapshot_names = {pair.name for pair in PAIRS if pair.anchor == "snapshot"}
    if snapshot_names != {"entangled_reference", "entangled_rollout"}:
        raise RuntimeError(
            "moving snapshots must anchor the two Entangled Distillation objectives"
        )


validate_objective()
