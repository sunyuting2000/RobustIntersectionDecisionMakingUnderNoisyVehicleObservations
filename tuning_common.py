"""Development-seed restrictions and safety-first parameter-selection helpers."""
from __future__ import annotations
from typing import Mapping
from experiment_config import PROJECT_DIRECTORY, REALISM_CANDIDATES, FROZEN_DEVELOPMENT_REALISM_CANDIDATE_NAME

DEVELOPMENT_SEEDS = frozenset(range(50))

SAFETY_EPSILON = 1e-12

def frozen_realism_candidate():
    """Return the single frozen nominal development sensing configuration."""
    matches = [
        candidate
        for candidate in REALISM_CANDIDATES
        if candidate.name == FROZEN_DEVELOPMENT_REALISM_CANDIDATE_NAME
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen development realism candidate is ambiguous")
    return matches[0]

def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)

def safety_exclusion_reasons(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return predeclared reasons that make a development candidate unsafe."""
    reasons: list[str] = []
    if float(row["collision_rate"]) > SAFETY_EPSILON:
        reasons.append("ego collision")
    if float(row["simultaneous_zone_occupancy_rate"]) > SAFETY_EPSILON:
        reasons.append("simultaneous conflict-zone occupancy")
    minimum_pet = _optional_float(row.get("minimum_pet"))
    if minimum_pet is not None and minimum_pet <= SAFETY_EPSILON:
        reasons.append("PET=0")
    return tuple(reasons)

def is_safety_eligible(row: Mapping[str, object]) -> bool:
    """Require zero collisions, zero overlap occupancy, and no zero PET."""
    return not safety_exclusion_reasons(row)
