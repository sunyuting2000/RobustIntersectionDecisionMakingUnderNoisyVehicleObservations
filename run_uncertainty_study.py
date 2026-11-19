"""Resumable coarse-to-refined Controller 3 parameter selection at four joint sensing profiles."""

from __future__ import annotations

import argparse

import csv

import json

from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass

from decimal import Decimal

from math import inf, isfinite

from pathlib import Path

from statistics import mean

from time import perf_counter

from typing import Iterable, Mapping

from fixed_gap_controller import ControllerState, ego_has_cleared_conflict_zone

from experiment_config import EVALUATION_DRIVER_SHIFTS, EVALUATION_EGO_HARD_BRAKING_THRESHOLD, EVALUATION_HAZARD_ENVIRONMENT_SEED, EVALUATION_HAZARD_VARIANTS, EVALUATION_TRAFFIC_SPAWN_PROBABILITIES

from experiment_config import AdaptiveCandidate, COARSE_BASE_CLEARANCES, COARSE_UNCERTAINTY_MULTIPLIERS

from experiment_config import REALISM_CANDIDATES, UNCERTAINTY_AWARE

from scenarios import _hazard_parameters, _make_hazard_environment, _make_stochastic_environment

from safety_metrics import PET_DEFINITION, GroundTruthSafetyTracker, SafetyMetricConfig, capture_policy_step_safety_metrics, observed_conflict_zone_ttc

from state_estimation import make_shared_kalman_observation_provider

from trajectory_prediction import DEFAULT_PREDICTION_CONFIG

from uncertainty_aware_predictive_controller import DEFAULT_CONFLICT_WINDOW_HALF_WIDTH, UncertaintyAwarePredictiveConfig, UncertaintyAwarePredictiveController

from vehicle_observations import ObservationLatencyConfig

PROJECT_DIRECTORY = Path(__file__).resolve().parent

RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"

PROTOCOL_VERSION = "controller3_noise_parameter_study_v1"

NOISE_LEVEL_NAMES = ("exact", "low", "medium_q1p5", "high")

NOISE_BY_NAME = {
    candidate.name: candidate
    for candidate in REALISM_CANDIDATES
    if candidate.name in NOISE_LEVEL_NAMES
}

if tuple(name for name in NOISE_LEVEL_NAMES if name in NOISE_BY_NAME) != (
    NOISE_LEVEL_NAMES
):
    raise RuntimeError("the four declared observation levels are unavailable")

SAFETY_EPSILON = 1e-12

BASE_REFINEMENT_RADIUS = Decimal("0.2")

BASE_REFINEMENT_STEP = Decimal("0.1")

MULTIPLIER_REFINEMENT_RADIUS = Decimal("0.01")

MULTIPLIER_REFINEMENT_STEP = Decimal("0.005")

@dataclass(frozen=True)
class NoiseStudyTask:
    """One paired Controller 3 episode for one parameter candidate."""

    task_id: str
    study_signature: str
    stage: str
    kind: str
    controller: str
    noise_level: str
    base_clearance: float
    uncertainty_multiplier: float
    seed: int
    spawn_probability: float | None
    driver_shift: str | None
    sensor_shift: str
    hazard_variant: str | None

def _candidate_id(candidate: AdaptiveCandidate) -> str:
    return (
        f"b0_{candidate.base_clearance:.3f}_"
        f"k_{candidate.uncertainty_multiplier:.3f}"
    )

def _study_signature(*, seed_start: int, episodes_per_cell: int) -> str:
    return (
        f"{PROTOCOL_VERSION}|seed_start={seed_start}|"
        f"episodes_per_cell={episodes_per_cell}|"
        f"conflict_window_half_width="
        f"{DEFAULT_CONFLICT_WINDOW_HALF_WIDTH:.3f}"
    )

def _build_tasks(
    *,
    stage: str,
    noise_level: str,
    candidates: Iterable[AdaptiveCandidate],
    seed_start: int,
    episodes_per_cell: int,
    study_signature: str,
) -> tuple[NoiseStudyTask, ...]:
    tasks: list[NoiseStudyTask] = []
    for candidate in candidates:
        candidate_id = _candidate_id(candidate)
        for driver_shift in EVALUATION_DRIVER_SHIFTS:
            for spawn_probability in EVALUATION_TRAFFIC_SPAWN_PROBABILITIES:
                for seed in range(seed_start, seed_start + episodes_per_cell):
                    task_id = (
                        f"{stage}|{noise_level}|{candidate_id}|stochastic|"
                        f"{driver_shift.name}|{spawn_probability:.2f}|{seed}"
                    )
                    tasks.append(
                        NoiseStudyTask(
                            task_id=task_id,
                            study_signature=study_signature,
                            stage=stage,
                            kind="stochastic",
                            controller=UNCERTAINTY_AWARE,
                            noise_level=noise_level,
                            base_clearance=candidate.base_clearance,
                            uncertainty_multiplier=(
                                candidate.uncertainty_multiplier
                            ),
                            seed=seed,
                            spawn_probability=spawn_probability,
                            driver_shift=driver_shift.name,
                            sensor_shift=noise_level,
                            hazard_variant=None,
                        )
                    )
        for hazard in EVALUATION_HAZARD_VARIANTS:
            task_id = (
                f"{stage}|{noise_level}|{candidate_id}|hazard|{hazard.name}"
            )
            tasks.append(
                NoiseStudyTask(
                    task_id=task_id,
                    study_signature=study_signature,
                    stage=stage,
                    kind="hazard",
                    controller=UNCERTAINTY_AWARE,
                    noise_level=noise_level,
                    base_clearance=candidate.base_clearance,
                    uncertainty_multiplier=candidate.uncertainty_multiplier,
                    seed=EVALUATION_HAZARD_ENVIRONMENT_SEED,
                    spawn_probability=None,
                    driver_shift=None,
                    sensor_shift=noise_level,
                    hazard_variant=hazard.name,
                )
            )
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise AssertionError("noise-study task IDs are not unique")
    return tuple(tasks)

def _make_provider(task: NoiseStudyTask):
    candidate = NOISE_BY_NAME[task.noise_level]
    noise_seed = task.seed
    if task.kind == "hazard":
        hazard_names = tuple(
            hazard.name for hazard in EVALUATION_HAZARD_VARIANTS
        )
        noise_seed = (
            EVALUATION_HAZARD_ENVIRONMENT_SEED
            + hazard_names.index(str(task.hazard_variant))
        )
    return make_shared_kalman_observation_provider(
        noise=candidate.noise,
        latency=ObservationLatencyConfig(candidate.latency),
        acceleration_process_std=candidate.acceleration_process_std,
        seed=noise_seed,
    )

def _make_controller(task: NoiseStudyTask, provider):
    return UncertaintyAwarePredictiveController(
        UncertaintyAwarePredictiveConfig(
            base_clearance=task.base_clearance,
            uncertainty_multiplier=task.uncertainty_multiplier,
            prediction=DEFAULT_PREDICTION_CONFIG,
            conflict_window_half_width=(
                DEFAULT_CONFLICT_WINDOW_HALF_WIDTH
            ),
        ),
        observation_provider=provider,
    )

RAW_FIELDNAMES = (
    "task_id",
    "study_signature",
    "stage",
    "kind",
    "controller",
    "noise_level",
    "candidate_id",
    "base_clearance",
    "uncertainty_multiplier",
    "conflict_window_half_width",
    "seed",
    "spawn_probability",
    "driver_shift",
    "hazard_variant",
    "hazard_parameters",
    "correlation",
    "position_bias_innovation_std",
    "position_white_std",
    "speed_bias_innovation_std",
    "speed_white_std",
    "configured_latency",
    "acceleration_process_std",
    "success",
    "collision",
    "background_collision",
    "timeout",
    "episode_time",
    "completion_time",
    "waiting_time",
    "commit_time",
    "clear_time",
    "minimum_predicted_clearance_at_commit",
    "maximum_required_clearance_at_commit",
    "minimum_clearance_slack_at_commit",
    "maximum_conflict_window_position_std_at_commit",
    "minimum_actual_ttc",
    "minimum_actual_ttc_during_commit",
    "minimum_pet",
    "simultaneous_zone_occupancy",
    "ego_hard_braking_events",
    "actual_ttc_at_ego_hard_braking",
    "perceived_ttc_at_ego_hard_braking",
    "mean_decision_time_ms",
    "maximum_decision_time_ms",
    "pet_definition",
)

def run_task(task: NoiseStudyTask) -> dict[str, object]:
    """Run one candidate episode and return its outcome and safety metrics."""
    env, initial_vehicles = (
        _make_stochastic_environment(task)
        if task.kind == "stochastic"
        else _make_hazard_environment(task)
    )
    provider = _make_provider(task)
    controller = _make_controller(task, provider)
    tracker = GroundTruthSafetyTracker(
        env,
        config=SafetyMetricConfig(
            hard_braking_threshold=EVALUATION_EGO_HARD_BRAKING_THRESHOLD
        ),
    )
    waiting_time = 0.0
    commit_time: float | None = None
    clear_time: float | None = None
    minimum_predicted_clearance: float | None = None
    maximum_required_clearance: float | None = None
    minimum_clearance_slack: float | None = None
    maximum_conflict_sigma: float | None = None
    decision_times_ms: list[float] = []
    all_background_vehicles: set[object] = set(initial_vehicles)

    try:
        env.road.record_history = True
        controller.reset()
        controller.state = (
            ControllerState.APPROACH
            if task.kind == "stochastic"
            else ControllerState.WAIT
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            decision_start = perf_counter()
            decision = controller.decide(env)
            decision_times_ms.append(
                (perf_counter() - decision_start) * 1000.0
            )
            if decision.state is ControllerState.WAIT:
                waiting_time += 1.0 / float(env.config["policy_frequency"])
            if decision.state is ControllerState.COMMIT and commit_time is None:
                commit_time = float(env.time)
                minimum_predicted_clearance = getattr(
                    decision,
                    "minimum_predicted_clearance",
                    None,
                )
                maximum_required_clearance = getattr(
                    decision,
                    "maximum_required_clearance",
                    None,
                )
                minimum_clearance_slack = getattr(
                    decision,
                    "minimum_clearance_slack",
                    None,
                )
                profiles = getattr(decision, "adaptive_clearances", ())
                if profiles:
                    maximum_conflict_sigma = max(
                        profile.maximum_position_std for profile in profiles
                    )

            if controller.last_observation is None:
                raise RuntimeError("controller did not retain its observation")
            step_start_time = float(env.time)
            perceived_ttc = observed_conflict_zone_ttc(
                env,
                controller.last_observation,
            )
            background_vehicles = tracker.associate_background_vehicles(env)
            all_background_vehicles.update(
                vehicle for _, vehicle in background_vehicles
            )
            tracker.observe(
                absolute_time=step_start_time,
                ego=env.vehicle,
                background_vehicles=background_vehicles,
                ego_committed=decision.state is ControllerState.COMMIT,
                env=env,
                perceived_ttc=perceived_ttc,
                controller_state=decision.state.value,
                action_name=decision.action_name,
            )
            _, _, terminated, truncated, _ = env.step(decision.action)
            capture_policy_step_safety_metrics(
                env=env,
                tracker=tracker,
                step_start_time=step_start_time,
                background_vehicles=background_vehicles,
                ego_committed=decision.state is ControllerState.COMMIT,
                perceived_ttc_at_step_start=perceived_ttc,
                controller_state=decision.state.value,
                action_name=decision.action_name,
            )
            if ego_has_cleared_conflict_zone(env.vehicle) and clear_time is None:
                clear_time = float(env.time)

        collision = bool(env.vehicle.crashed)
        success = bool(env.has_arrived(env.vehicle)) and not collision
        timeout = bool(truncated) and not collision and not success
        safety = tracker.result()
        episode_time = float(env.time)
        noise = NOISE_BY_NAME[task.noise_level]
        return dict(
            zip(
                RAW_FIELDNAMES,
                (
                    task.task_id,
                    task.study_signature,
                    task.stage,
                    task.kind,
                    task.controller,
                    task.noise_level,
                    _candidate_id(
                        AdaptiveCandidate(
                            task.base_clearance,
                            task.uncertainty_multiplier,
                        )
                    ),
                    task.base_clearance,
                    task.uncertainty_multiplier,
                    DEFAULT_CONFLICT_WINDOW_HALF_WIDTH,
                    task.seed,
                    task.spawn_probability,
                    task.driver_shift,
                    task.hazard_variant,
                    json.dumps(
                        _hazard_parameters(task)
                        if task.kind == "hazard"
                        else {},
                        sort_keys=True,
                    ),
                    noise.correlation,
                    noise.position_bias_innovation_std,
                    noise.position_white_std,
                    noise.speed_bias_innovation_std,
                    noise.speed_white_std,
                    noise.latency,
                    noise.acceleration_process_std,
                    success,
                    collision,
                    any(
                        bool(getattr(vehicle, "crashed", False))
                        for vehicle in all_background_vehicles
                    ),
                    timeout,
                    episode_time,
                    episode_time if success else None,
                    waiting_time,
                    commit_time,
                    clear_time,
                    minimum_predicted_clearance,
                    maximum_required_clearance,
                    minimum_clearance_slack,
                    maximum_conflict_sigma,
                    safety.minimum_actual_ttc,
                    safety.minimum_actual_ttc_during_commit,
                    safety.minimum_pet,
                    safety.simultaneous_zone_occupancy,
                    safety.ego_hard_braking_events,
                    safety.minimum_actual_ttc_at_ego_hard_braking,
                    safety.minimum_perceived_ttc_at_ego_hard_braking,
                    mean(decision_times_ms),
                    max(decision_times_ms),
                    PET_DEFINITION,
                ),
                strict=True,
            )
        )
    finally:
        env.close()

def _append_checkpoint(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()

def _load_checkpoint(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if rows and tuple(rows[0]) != RAW_FIELDNAMES:
        raise ValueError("noise-study checkpoint schema does not match")
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("noise-study checkpoint has duplicate task IDs")
    return rows

def _write_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def _as_bool(value: object) -> bool:
    return value is True or str(value).lower() == "true"

def _as_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    parsed = float(value)
    return parsed if isfinite(parsed) else None

def _optional_mean(
    rows: Iterable[Mapping[str, object]],
    field: str,
) -> float | None:
    values = [_as_float(row[field]) for row in rows]
    present = [value for value in values if value is not None]
    return mean(present) if present else None

def _optional_minimum(
    rows: Iterable[Mapping[str, object]],
    field: str,
) -> float | None:
    values = [_as_float(row[field]) for row in rows]
    present = [value for value in values if value is not None]
    return min(present) if present else None

SUMMARY_FIELDNAMES = (
    "study_signature",
    "stage",
    "noise_level",
    "candidate_id",
    "base_clearance",
    "uncertainty_multiplier",
    "conflict_window_half_width",
    "correlation",
    "position_bias_innovation_std",
    "position_white_std",
    "speed_bias_innovation_std",
    "speed_white_std",
    "configured_latency",
    "acceleration_process_std",
    "episodes",
    "stochastic_episodes",
    "hazard_episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "stochastic_success_rate",
    "stochastic_collision_rate",
    "stochastic_timeout_rate",
    "hazard_success_rate",
    "hazard_collision_rate",
    "hazard_timeout_rate",
    "mean_stochastic_waiting_time",
    "mean_stochastic_successful_completion_time",
    "minimum_actual_ttc",
    "minimum_actual_ttc_during_commit",
    "minimum_pet",
    "simultaneous_zone_occupancy_rate",
    "mean_stochastic_ego_hard_braking_events",
    "mean_maximum_required_clearance_at_commit",
    "mean_conflict_window_position_std_at_commit",
    "mean_decision_time_ms",
    "zero_observed_safety_events",
    "observed_safety_events",
)

def _rate(rows: list[Mapping[str, object]], field: str) -> float:
    return sum(_as_bool(row[field]) for row in rows) / len(rows)

def _summarize_candidate(
    rows: list[Mapping[str, object]],
    *,
    stage: str,
    noise_level: str,
    candidate: AdaptiveCandidate,
) -> dict[str, object]:
    candidate_id = _candidate_id(candidate)
    selected = [
        row
        for row in rows
        if row["stage"] == stage
        and row["noise_level"] == noise_level
        and row["candidate_id"] == candidate_id
    ]
    if not selected:
        raise RuntimeError(
            f"no rows found for {stage}/{noise_level}/{candidate_id}"
        )
    stochastic = [row for row in selected if row["kind"] == "stochastic"]
    hazards = [row for row in selected if row["kind"] == "hazard"]
    if not stochastic or len(hazards) != len(EVALUATION_HAZARD_VARIANTS):
        raise RuntimeError(
            f"incomplete candidate rows for {stage}/{noise_level}/{candidate_id}"
        )

    completion_times = [
        _as_float(row["completion_time"])
        for row in stochastic
        if _as_bool(row["success"])
    ]
    completion_times = [
        value for value in completion_times if value is not None
    ]
    collision_rate = _rate(selected, "collision")
    simultaneous_rate = _rate(
        selected,
        "simultaneous_zone_occupancy",
    )
    minimum_pet = _optional_minimum(selected, "minimum_pet")
    reasons: list[str] = []
    if collision_rate > SAFETY_EPSILON:
        reasons.append("ego collision")
    if simultaneous_rate > SAFETY_EPSILON:
        reasons.append("simultaneous conflict-zone occupancy")
    if minimum_pet is not None and minimum_pet <= SAFETY_EPSILON:
        reasons.append("PET=0")
    if not all(_as_bool(row["success"]) for row in hazards):
        reasons.append("hazard not completed successfully")

    first = selected[0]
    values = (
        first["study_signature"],
        stage,
        noise_level,
        candidate_id,
        candidate.base_clearance,
        candidate.uncertainty_multiplier,
        DEFAULT_CONFLICT_WINDOW_HALF_WIDTH,
        first["correlation"],
        first["position_bias_innovation_std"],
        first["position_white_std"],
        first["speed_bias_innovation_std"],
        first["speed_white_std"],
        first["configured_latency"],
        first["acceleration_process_std"],
        len(selected),
        len(stochastic),
        len(hazards),
        _rate(selected, "success"),
        collision_rate,
        _rate(selected, "timeout"),
        _rate(stochastic, "success"),
        _rate(stochastic, "collision"),
        _rate(stochastic, "timeout"),
        _rate(hazards, "success"),
        _rate(hazards, "collision"),
        _rate(hazards, "timeout"),
        _optional_mean(stochastic, "waiting_time"),
        mean(completion_times) if completion_times else None,
        _optional_minimum(selected, "minimum_actual_ttc"),
        _optional_minimum(selected, "minimum_actual_ttc_during_commit"),
        minimum_pet,
        simultaneous_rate,
        _optional_mean(stochastic, "ego_hard_braking_events"),
        _optional_mean(
            stochastic,
            "maximum_required_clearance_at_commit",
        ),
        _optional_mean(
            stochastic,
            "maximum_conflict_window_position_std_at_commit",
        ),
        _optional_mean(selected, "mean_decision_time_ms"),
        not reasons,
        "; ".join(reasons),
    )
    return dict(zip(SUMMARY_FIELDNAMES, values, strict=True))

def _ranking_key(row: Mapping[str, object]) -> tuple[float, ...]:
    completion = _as_float(
        row["mean_stochastic_successful_completion_time"]
    )
    required = _as_float(
        row["mean_maximum_required_clearance_at_commit"]
    )
    return (
        float(row["stochastic_collision_rate"]),
        float(row["simultaneous_zone_occupancy_rate"]),
        float(row["hazard_collision_rate"]),
        float(row["hazard_timeout_rate"]),
        float(row["stochastic_timeout_rate"]),
        completion if completion is not None else inf,
        float(row["mean_stochastic_waiting_time"]),
        required if required is not None else inf,
        float(row["base_clearance"]),
        float(row["uncertainty_multiplier"]),
    )

def _select_candidate(
    summaries: Iterable[Mapping[str, object]],
) -> Mapping[str, object] | None:
    rows = list(summaries)
    return min(rows, key=_ranking_key) if rows else None

def _run_missing_tasks(
    tasks: tuple[NoiseStudyTask, ...],
    *,
    checkpoint_path: Path,
    checkpoint_rows: list[dict[str, object]],
    completed_ids: set[str],
    workers: int,
    progress_every: int,
) -> None:
    remaining = [task for task in tasks if task.task_id not in completed_ids]
    if not remaining:
        return
    print(
        f"Running {len(remaining)} missing episodes "
        f"({tasks[0].stage}, {tasks[0].noise_level})...",
        flush=True,
    )
    if workers == 1:
        for index, task in enumerate(remaining, start=1):
            row = run_task(task)
            _append_checkpoint(checkpoint_path, row)
            checkpoint_rows.append(row)
            completed_ids.add(task.task_id)
            if index % progress_every == 0 or index == len(remaining):
                print(
                    f"  checkpointed {index}/{len(remaining)} missing",
                    flush=True,
                )
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_task, task): task for task in remaining
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            _append_checkpoint(checkpoint_path, row)
            checkpoint_rows.append(row)
            completed_ids.add(str(row["task_id"]))
            if index % progress_every == 0 or index == len(remaining):
                print(
                    f"  checkpointed {index}/{len(remaining)} missing",
                    flush=True,
                )

def _summaries_for(
    rows: list[Mapping[str, object]],
    *,
    stage: str,
    noise_level: str,
    candidates: Iterable[AdaptiveCandidate],
) -> list[dict[str, object]]:
    return [
        _summarize_candidate(
            rows,
            stage=stage,
            noise_level=noise_level,
            candidate=candidate,
        )
        for candidate in candidates
    ]

def _refinement_axis(
    centre: float,
    *,
    radius: Decimal,
    step: Decimal,
) -> tuple[float, ...]:
    decimal_centre = Decimal(str(centre))
    lower = max(Decimal("0"), decimal_centre - radius)
    upper = decimal_centre + radius
    count = int((upper - lower) / step)
    return tuple(float(lower + step * index) for index in range(count + 1))

def _local_refinement_candidates(
    centre: AdaptiveCandidate,
) -> tuple[AdaptiveCandidate, ...]:
    """Refine around a leader even when it lies on the coarse boundary."""
    base_values = _refinement_axis(
        centre.base_clearance,
        radius=BASE_REFINEMENT_RADIUS,
        step=BASE_REFINEMENT_STEP,
    )
    multiplier_values = _refinement_axis(
        centre.uncertainty_multiplier,
        radius=MULTIPLIER_REFINEMENT_RADIUS,
        step=MULTIPLIER_REFINEMENT_STEP,
    )
    return tuple(
        AdaptiveCandidate(base, multiplier)
        for base in base_values
        for multiplier in multiplier_values
    )

def _coarse_search(
    noise_level: str,
    *,
    seed_start: int,
    episodes_per_cell: int,
    study_signature: str,
    checkpoint_path: Path,
    checkpoint_rows: list[dict[str, object]],
    completed_ids: set[str],
    declared_ids: set[str],
    workers: int,
    progress_every: int,
) -> tuple[
    tuple[AdaptiveCandidate, ...],
    list[dict[str, object]],
    Mapping[str, object],
]:
    candidates = tuple(
        AdaptiveCandidate(base, multiplier)
        for base in COARSE_BASE_CLEARANCES
        for multiplier in COARSE_UNCERTAINTY_MULTIPLIERS
    )
    tasks = _build_tasks(
        stage="coarse",
        noise_level=noise_level,
        candidates=candidates,
        seed_start=seed_start,
        episodes_per_cell=episodes_per_cell,
        study_signature=study_signature,
    )
    declared_ids.update(task.task_id for task in tasks)
    _run_missing_tasks(
        tasks,
        checkpoint_path=checkpoint_path,
        checkpoint_rows=checkpoint_rows,
        completed_ids=completed_ids,
        workers=workers,
        progress_every=progress_every,
    )
    summaries = _summaries_for(
        checkpoint_rows,
        stage="coarse",
        noise_level=noise_level,
        candidates=candidates,
    )
    leader = _select_candidate(summaries)
    if leader is None:
        raise RuntimeError(f"no completed coarse candidates for {noise_level}")
    return candidates, summaries, leader

def _refined_search(
    noise_level: str,
    *,
    coarse_candidates: tuple[AdaptiveCandidate, ...],
    coarse_leader: Mapping[str, object],
    seed_start: int,
    episodes_per_cell: int,
    study_signature: str,
    checkpoint_path: Path,
    checkpoint_rows: list[dict[str, object]],
    completed_ids: set[str],
    declared_ids: set[str],
    workers: int,
    progress_every: int,
) -> tuple[
    tuple[AdaptiveCandidate, ...],
    list[dict[str, object]],
    Mapping[str, object],
]:
    centre = AdaptiveCandidate(
        float(coarse_leader["base_clearance"]),
        float(coarse_leader["uncertainty_multiplier"]),
    )
    del coarse_candidates
    candidates = _local_refinement_candidates(centre)
    tasks = _build_tasks(
        stage="refined",
        noise_level=noise_level,
        candidates=candidates,
        seed_start=seed_start,
        episodes_per_cell=episodes_per_cell,
        study_signature=study_signature,
    )
    declared_ids.update(task.task_id for task in tasks)
    _run_missing_tasks(
        tasks,
        checkpoint_path=checkpoint_path,
        checkpoint_rows=checkpoint_rows,
        completed_ids=completed_ids,
        workers=workers,
        progress_every=progress_every,
    )
    summaries = _summaries_for(
        checkpoint_rows,
        stage="refined",
        noise_level=noise_level,
        candidates=candidates,
    )
    leader = _select_candidate(summaries)
    if leader is None:
        raise RuntimeError(f"no completed refined candidates for {noise_level}")
    return candidates, summaries, leader

SELECTED_FIELDNAMES = (
    "noise_level",
    "selected_base_clearance",
    "selected_uncertainty_multiplier",
    "coarse_base_clearance",
    "coarse_uncertainty_multiplier",
    "selected_stochastic_success_rate",
    "selected_stochastic_collision_rate",
    "selected_stochastic_timeout_rate",
    "selected_hazard_success_rate",
    "selected_mean_stochastic_waiting_time",
    "selected_mean_stochastic_successful_completion_time",
    "selected_minimum_pet",
    "selected_simultaneous_zone_occupancy_rate",
    "selected_mean_required_clearance_at_commit",
    "selected_mean_conflict_window_position_std_at_commit",
    "selected_mean_decision_time_ms",
    "refined_b0_at_local_boundary",
    "refined_k_at_local_boundary",
    "correlation",
    "position_bias_innovation_std",
    "position_white_std",
    "speed_bias_innovation_std",
    "speed_white_std",
    "configured_latency",
    "acceleration_process_std",
    "conflict_window_half_width",
    "study_signature",
)

def _selected_row(
    *,
    noise_level: str,
    coarse_leader: Mapping[str, object],
    refined_leader: Mapping[str, object],
    refined_candidates: tuple[AdaptiveCandidate, ...],
) -> dict[str, object]:
    b0 = float(refined_leader["base_clearance"])
    k = float(refined_leader["uncertainty_multiplier"])
    base_values = tuple(candidate.base_clearance for candidate in refined_candidates)
    multiplier_values = tuple(
        candidate.uncertainty_multiplier for candidate in refined_candidates
    )
    values = (
        noise_level,
        b0,
        k,
        coarse_leader["base_clearance"],
        coarse_leader["uncertainty_multiplier"],
        refined_leader["stochastic_success_rate"],
        refined_leader["stochastic_collision_rate"],
        refined_leader["stochastic_timeout_rate"],
        refined_leader["hazard_success_rate"],
        refined_leader["mean_stochastic_waiting_time"],
        refined_leader["mean_stochastic_successful_completion_time"],
        refined_leader["minimum_pet"],
        refined_leader["simultaneous_zone_occupancy_rate"],
        refined_leader["mean_maximum_required_clearance_at_commit"],
        refined_leader["mean_conflict_window_position_std_at_commit"],
        refined_leader["mean_decision_time_ms"],
        abs(b0 - min(base_values)) <= SAFETY_EPSILON
        or abs(b0 - max(base_values)) <= SAFETY_EPSILON,
        abs(k - min(multiplier_values)) <= SAFETY_EPSILON
        or abs(k - max(multiplier_values)) <= SAFETY_EPSILON,
        refined_leader["correlation"],
        refined_leader["position_bias_innovation_std"],
        refined_leader["position_white_std"],
        refined_leader["speed_bias_innovation_std"],
        refined_leader["speed_white_std"],
        refined_leader["configured_latency"],
        refined_leader["acceleration_process_std"],
        DEFAULT_CONFLICT_WINDOW_HALF_WIDTH,
        refined_leader["study_signature"],
    )
    return dict(zip(SELECTED_FIELDNAMES, values, strict=True))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable Controller 3 parameter study at four noise "
            "levels."
        )
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--episodes-per-cell", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument(
        "--output-prefix",
        default="controller3_noise_study",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.progress_every <= 0:
        raise ValueError("workers and progress interval must be positive")
    if args.episodes_per_cell <= 0 or args.seed_start < 0:
        raise ValueError("episode count must be positive and seed non-negative")
    if Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("output prefix must be a filename prefix, not a path")

    signature = _study_signature(
        seed_start=args.seed_start,
        episodes_per_cell=args.episodes_per_cell,
    )
    prefix = args.output_prefix
    checkpoint_path = RESULTS_DIRECTORY / f"{prefix}_checkpoint.csv"
    raw_path = RESULTS_DIRECTORY / f"{prefix}_raw.csv"
    summary_path = RESULTS_DIRECTORY / f"{prefix}_candidate_summary.csv"
    selected_path = RESULTS_DIRECTORY / f"{prefix}_selected_parameters.csv"

    checkpoint_rows: list[dict[str, object]] = list(
        _load_checkpoint(checkpoint_path)
    )
    incompatible = [
        row
        for row in checkpoint_rows
        if row["study_signature"] != signature
    ]
    if incompatible:
        raise ValueError(
            "existing checkpoint was created with different seeds, episode "
            "count, protocol version, or uncertainty window"
        )
    completed_ids = {str(row["task_id"]) for row in checkpoint_rows}
    declared_ids: set[str] = set()
    summaries_by_key: dict[
        tuple[str, str, str], dict[str, object]
    ] = {}
    selected_rows: list[dict[str, object]] = []

    print("Controller 3 observation-noise parameter study", flush=True)
    print(
        "Noise levels: exact, low, medium_q1p5, high; "
        f"checkpointed episodes: {len(completed_ids)}",
        flush=True,
    )
    print(
        "Only (b_0, k) is searched. Controllers 1 and 2 are not run.",
        flush=True,
    )

    start = perf_counter()
    for noise_level in NOISE_LEVEL_NAMES:
        print(f"\n=== {noise_level} ===", flush=True)
        coarse_candidates, coarse_summaries, coarse_leader = _coarse_search(
            noise_level,
            seed_start=args.seed_start,
            episodes_per_cell=args.episodes_per_cell,
            study_signature=signature,
            checkpoint_path=checkpoint_path,
            checkpoint_rows=checkpoint_rows,
            completed_ids=completed_ids,
            declared_ids=declared_ids,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        for row in coarse_summaries:
            summaries_by_key[(
                "coarse",
                noise_level,
                str(row["candidate_id"]),
            )] = row
        refined_candidates, refined_summaries, refined_leader = (
            _refined_search(
                noise_level,
                coarse_candidates=coarse_candidates,
                coarse_leader=coarse_leader,
                seed_start=args.seed_start,
                episodes_per_cell=args.episodes_per_cell,
                study_signature=signature,
                checkpoint_path=checkpoint_path,
                checkpoint_rows=checkpoint_rows,
                completed_ids=completed_ids,
                declared_ids=declared_ids,
                workers=args.workers,
                progress_every=args.progress_every,
            )
        )
        for row in refined_summaries:
            summaries_by_key[(
                "refined",
                noise_level,
                str(row["candidate_id"]),
            )] = row
        selected = _selected_row(
            noise_level=noise_level,
            coarse_leader=coarse_leader,
            refined_leader=refined_leader,
            refined_candidates=refined_candidates,
        )
        selected_rows.append(selected)
        print(
            f"Selected {noise_level}: b_0="
            f"{float(selected['selected_base_clearance']):.3f} m, k="
            f"{float(selected['selected_uncertainty_multiplier']):.3f}",
            flush=True,
        )

    checkpoint_ids = {str(row["task_id"]) for row in checkpoint_rows}
    supplemental = checkpoint_ids - declared_ids
    missing = declared_ids - checkpoint_ids
    if missing:
        raise AssertionError(
            f"study ended with {len(missing)} declared episodes missing"
        )

    raw_rows = sorted(
        (
            row
            for row in checkpoint_rows
            if str(row["task_id"]) in declared_ids
        ),
        key=lambda row: str(row["task_id"]),
    )
    summary_rows = sorted(
        summaries_by_key.values(),
        key=lambda row: (
            NOISE_LEVEL_NAMES.index(str(row["noise_level"])),
            str(row["stage"]),
            float(row["base_clearance"]),
            float(row["uncertainty_multiplier"]),
        ),
    )
    _write_rows(raw_path, raw_rows, RAW_FIELDNAMES)
    _write_rows(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    _write_rows(selected_path, selected_rows, SELECTED_FIELDNAMES)

    elapsed = perf_counter() - start
    if supplemental:
        print(
            f"\nRetained {len(supplemental)} checkpoint episode(s) outside "
            "the declared grid; excluded these episodes from the current outputs.",
            flush=True,
        )
    print(
        f"\nCompleted {len(raw_rows)} Controller 3 episodes in "
        f"{elapsed:.1f} s.",
        flush=True,
    )
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Raw results: {raw_path}", flush=True)
    print(f"Candidate summaries: {summary_path}", flush=True)
    print(f"Selected parameter trend: {selected_path}", flush=True)

if __name__ == "__main__":
    main()
