"""Evaluate the two frozen fixed controllers with checkpointed outputs."""

from __future__ import annotations
from scenarios import _make_stochastic_environment, _make_hazard_environment, _hazard_parameters, SENSOR_SHIFT_BY_NAME, HAZARD_BY_NAME
from experiment_config import PROTOCOL_VERSION, FIXED_GAP, PREDICTIVE

import argparse

import csv

import json

from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass

from math import isfinite

from pathlib import Path

from statistics import mean

from time import perf_counter

from typing import Iterable, Mapping

from fixed_gap_controller import ControllerState, FixedGapConfig, FixedGapController, ego_has_cleared_conflict_zone

from experiment_config import EVALUATION_DRIVER_SHIFTS, EVALUATION_EGO_HARD_BRAKING_THRESHOLD, EVALUATION_HAZARD_ENVIRONMENT_SEED, EVALUATION_HAZARD_VARIANTS, EVALUATION_KALMAN_ACCELERATION_PROCESS_STD, EVALUATION_SEEDS, EVALUATION_SENSOR_SHIFTS, EVALUATION_TRAFFIC_SPAWN_PROBABILITIES



from predictive_fixed_margin_controller import FROZEN_REALISTIC_FIXED_CLEARANCE, PredictiveFixedMarginConfig, PredictiveFixedMarginController

from safety_metrics import PET_DEFINITION, GroundTruthSafetyTracker, SafetyMetricConfig, capture_policy_step_safety_metrics, observed_conflict_zone_ttc, serialize_ego_hard_braking_events

from state_estimation import make_shared_kalman_observation_provider

from trajectory_prediction import DEFAULT_PREDICTION_CONFIG




from vehicle_observations import CorrelatedGaussianNoiseConfig, ObservationLatencyConfig

PROJECT_DIRECTORY = Path(__file__).resolve().parent

CONTROLLERS = (FIXED_GAP, PREDICTIVE)

FROZEN_CROSSING_TIME = 5.0

FROZEN_GAP_MARGIN = 0.0

@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    kind: str
    controller: str
    seed: int
    spawn_probability: float | None
    driver_shift: str | None
    sensor_shift: str
    hazard_variant: str | None

def build_tasks(*, controllers=CONTROLLERS, seeds=EVALUATION_SEEDS) -> tuple[EvaluationTask, ...]:
    tasks: list[EvaluationTask] = []
    for controller in controllers:
        for driver_shift in EVALUATION_DRIVER_SHIFTS:
            for sensor_shift in EVALUATION_SENSOR_SHIFTS:
                for spawn_probability in EVALUATION_TRAFFIC_SPAWN_PROBABILITIES:
                    for seed in seeds:
                        task_id = (
                            f"stochastic|{controller}|{driver_shift.name}|"
                            f"{sensor_shift.name}|{spawn_probability:.2f}|{seed}"
                        )
                        tasks.append(
                            EvaluationTask(
                                task_id=task_id,
                                kind="stochastic",
                                controller=controller,
                                seed=seed,
                                spawn_probability=spawn_probability,
                                driver_shift=driver_shift.name,
                                sensor_shift=sensor_shift.name,
                                hazard_variant=None,
                            )
                        )
        for variant in EVALUATION_HAZARD_VARIANTS:
            tasks.append(
                EvaluationTask(
                    task_id=f"hazard|{controller}|{variant.name}",
                    kind="hazard",
                    controller=controller,
                    seed=EVALUATION_HAZARD_ENVIRONMENT_SEED,
                    spawn_probability=None,
                    driver_shift=None,
                    sensor_shift=variant.sensor_shift,
                    hazard_variant=variant.name,
                )
            )
    return tuple(tasks)

def _make_provider(task: EvaluationTask):
    sensor = SENSOR_SHIFT_BY_NAME[task.sensor_shift]
    noise_seed = task.seed
    if task.kind == "hazard":
        variant = HAZARD_BY_NAME[str(task.hazard_variant)]
        noise_seed = (
            variant.noise_seed
            if variant.noise_seed is not None
            else EVALUATION_HAZARD_ENVIRONMENT_SEED
        )
    return make_shared_kalman_observation_provider(
        noise=CorrelatedGaussianNoiseConfig(
            correlation=sensor.correlation,
            position_bias_innovation_std=sensor.position_bias_innovation_std,
            position_white_std=sensor.position_white_std,
            speed_bias_innovation_std=sensor.speed_bias_innovation_std,
            speed_white_std=sensor.speed_white_std,
        ),
        latency=ObservationLatencyConfig(sensor.latency),
        acceleration_process_std=EVALUATION_KALMAN_ACCELERATION_PROCESS_STD,
        seed=noise_seed,
    )

def _make_controller(task: EvaluationTask, provider):
    if task.controller == FIXED_GAP:
        return FixedGapController(
            FixedGapConfig(
                crossing_time=FROZEN_CROSSING_TIME,
                safety_margin=FROZEN_GAP_MARGIN,
            ),
            observation_provider=provider,
        )
    if task.controller == PREDICTIVE:
        return PredictiveFixedMarginController(
            PredictiveFixedMarginConfig(
                clearance=FROZEN_REALISTIC_FIXED_CLEARANCE,
                prediction=DEFAULT_PREDICTION_CONFIG,
            ),
            observation_provider=provider,
        )
    raise ValueError(f"unknown controller: {task.controller}")

def initial_controller_state(task: EvaluationTask) -> ControllerState:
    """Return the state matching the task's physical ego initialization.

    Stochastic episodes retain the normal moving approach created by the
    environment, so they must enter through the controller's APPROACH logic.
    Hazard episodes explicitly place a stationary ego at the stop line and
    therefore begin in WAIT.
    """
    if task.kind == "stochastic":
        return ControllerState.APPROACH
    if task.kind == "hazard":
        return ControllerState.WAIT
    raise ValueError(f"unknown evaluation task kind: {task.kind}")

RAW_FIELDNAMES = (
    "task_id", "kind", "controller", "seed", "spawn_probability",
    "driver_shift", "sensor_shift", "hazard_variant", "hazard_parameters",
    "success", "collision", "background_collision", "timeout",
    "episode_time", "completion_time", "waiting_time", "commit_time",
    "clear_time", "minimum_predicted_clearance_at_commit",
    "required_clearance_at_commit", "clearance_slack_at_commit",
    "maximum_position_std_at_commit", "minimum_actual_ttc",
    "minimum_actual_ttc_during_commit", "minimum_pet",
    "simultaneous_zone_occupancy", "ego_hard_braking_events",
    "ego_hard_braking_event_log", "actual_ttc_at_ego_hard_braking",
    "perceived_ttc_at_ego_hard_braking", "mean_decision_time_ms",
    "maximum_decision_time_ms", "pet_definition", "protocol_version",
)

def run_task(task: EvaluationTask) -> dict[str, object]:
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
    required_clearance: float | None = None
    clearance_slack: float | None = None
    maximum_sigma: float | None = None
    decision_times_ms: list[float] = []
    all_background_vehicles: set[object] = set(initial_vehicles)

    try:
        env.road.record_history = True
        controller.reset()
        controller.state = initial_controller_state(task)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            decision_start = perf_counter()
            decision = controller.decide(env)
            decision_times_ms.append((perf_counter() - decision_start) * 1000)
            if decision.state is ControllerState.WAIT:
                waiting_time += 1.0 / float(env.config["policy_frequency"])
            if decision.state is ControllerState.COMMIT and commit_time is None:
                commit_time = float(env.time)
                minimum_predicted_clearance = getattr(
                    decision, "minimum_predicted_clearance", None
                )
                if task.controller == FIXED_GAP:
                    required_clearance = None
                elif task.controller == PREDICTIVE:
                    required_clearance = FROZEN_REALISTIC_FIXED_CLEARANCE
                else:
                    required_clearance = getattr(
                        decision, "maximum_required_clearance", None
                    )
                    profiles = getattr(decision, "adaptive_clearances", ())
                    if profiles:
                        maximum_sigma = max(
                            profile.maximum_position_std for profile in profiles
                        )
                if required_clearance is not None and minimum_predicted_clearance is not None:
                    clearance_slack = minimum_predicted_clearance - required_clearance

            if controller.last_observation is None:
                raise RuntimeError("controller did not retain its observation")
            step_start_time = float(env.time)
            perceived_ttc = observed_conflict_zone_ttc(
                env, controller.last_observation
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
        events = safety.ego_hard_braking_event_records
        episode_time = float(env.time)
        return dict(
            zip(
                RAW_FIELDNAMES,
                (
                    task.task_id, task.kind, task.controller, task.seed,
                    task.spawn_probability, task.driver_shift, task.sensor_shift,
                    task.hazard_variant,
                    json.dumps(
                        _hazard_parameters(task) if task.kind == "hazard" else {},
                        sort_keys=True,
                    ),
                    success, collision,
                    any(bool(getattr(vehicle, "crashed", False)) for vehicle in all_background_vehicles),
                    timeout, episode_time, episode_time if success else None,
                    waiting_time, commit_time, clear_time,
                    minimum_predicted_clearance, required_clearance,
                    clearance_slack, maximum_sigma, safety.minimum_actual_ttc,
                    safety.minimum_actual_ttc_during_commit, safety.minimum_pet,
                    safety.simultaneous_zone_occupancy,
                    safety.ego_hard_braking_events,
                    serialize_ego_hard_braking_events(events),
                    safety.minimum_actual_ttc_at_ego_hard_braking,
                    safety.minimum_perceived_ttc_at_ego_hard_braking,
                    mean(decision_times_ms), max(decision_times_ms),
                    PET_DEFINITION, PROTOCOL_VERSION,
                ),
                strict=True,
            )
        )
    finally:
        env.close()

def _write_rows(path: Path, rows: Iterable[Mapping[str, object]], fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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
        raise ValueError("Baseline checkpoint schema does not match the runner")
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Baseline checkpoint contains duplicate task IDs")
    return rows

def _as_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    result = float(value)
    return result if isfinite(result) else None

def _as_bool(value: object) -> bool:
    return value is True or str(value).lower() == "true"

def _mean_optional(rows, key: str) -> float | None:
    values = [_as_float(row[key]) for row in rows]
    present = [value for value in values if value is not None]
    return mean(present) if present else None

SUMMARY_FIELDNAMES = (
    "kind", "controller", "driver_shift", "sensor_shift",
    "spawn_probability", "hazard_variant", "episodes", "success_rate",
    "collision_rate", "background_collision_rate", "timeout_rate",
    "mean_waiting_time", "mean_successful_completion_time",
    "minimum_actual_ttc", "minimum_actual_ttc_during_commit", "minimum_pet",
    "simultaneous_zone_occupancy_rate", "mean_ego_hard_braking_events",
    "mean_actual_ttc_at_ego_hard_braking",
    "mean_perceived_ttc_at_ego_hard_braking", "mean_decision_time_ms",
    "maximum_decision_time_ms", "pet_definition", "protocol_version",
)

def summarize(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            row["kind"], row["controller"], row["driver_shift"],
            row["sensor_shift"], row["spawn_probability"], row["hazard_variant"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        successful_completion = [
            _as_float(row["completion_time"])
            for row in group
            if _as_bool(row["success"])
        ]
        successful_completion = [v for v in successful_completion if v is not None]
        minimum = lambda field: min(
            (value for value in (_as_float(row[field]) for row in group) if value is not None),
            default=None,
        )
        values = (
            *key, len(group),
            sum(_as_bool(row["success"]) for row in group) / len(group),
            sum(_as_bool(row["collision"]) for row in group) / len(group),
            sum(_as_bool(row["background_collision"]) for row in group) / len(group),
            sum(_as_bool(row["timeout"]) for row in group) / len(group),
            _mean_optional(group, "waiting_time"),
            mean(successful_completion) if successful_completion else None,
            minimum("minimum_actual_ttc"),
            minimum("minimum_actual_ttc_during_commit"),
            minimum("minimum_pet"),
            sum(_as_bool(row["simultaneous_zone_occupancy"]) for row in group) / len(group),
            _mean_optional(group, "ego_hard_braking_events"),
            _mean_optional(group, "actual_ttc_at_ego_hard_braking"),
            _mean_optional(group, "perceived_ttc_at_ego_hard_braking"),
            _mean_optional(group, "mean_decision_time_ms"),
            max(float(row["maximum_decision_time_ms"]) for row in group),
            PET_DEFINITION, PROTOCOL_VERSION,
        )
        summaries.append(dict(zip(SUMMARY_FIELDNAMES, values, strict=True)))
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the two frozen fixed controllers.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--episodes-per-cell", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--output-directory", type=Path, default=PROJECT_DIRECTORY / "results")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.workers, args.progress_every, args.episodes_per_cell) <= 0 or args.seed_start < 0:
        raise ValueError("workers, progress interval, and episodes must be positive; seed must be non-negative")
    if len(set(args.controllers)) != len(args.controllers):
        raise ValueError("controllers must not contain duplicates")
    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes_per_cell))
    if set(seeds) & set(range(50)):
        raise ValueError("evaluation seeds must be disjoint from development seeds 0-49")
    output = args.output_directory.expanduser().resolve()
    raw_path, summary_path, checkpoint_path = (output / f"baselines_{suffix}.csv" for suffix in ("raw", "summary", "checkpoint"))
    for path in (raw_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Completed output already exists: {path}; use another output directory")
    tasks = build_tasks(controllers=tuple(args.controllers), seeds=seeds)
    declared_ids = {task.task_id for task in tasks}
    if len(declared_ids) != len(tasks):
        raise AssertionError("duplicated evaluation task IDs")
    checkpoint_rows = _load_checkpoint(checkpoint_path)
    completed_ids = {row["task_id"] for row in checkpoint_rows}
    if len(completed_ids) != len(checkpoint_rows) or not completed_ids.issubset(declared_ids):
        raise ValueError("checkpoint contains duplicated or undeclared task IDs")
    if any(row.get("protocol_version") != PROTOCOL_VERSION for row in checkpoint_rows):
        raise ValueError("checkpoint was generated with another protocol")
    remaining = [task for task in tasks if task.task_id not in completed_ids]
    print(f"Baseline evaluation: {len(tasks)} episodes; {len(remaining)} remaining", flush=True)
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_task, task) for task in remaining]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            _append_checkpoint(checkpoint_path, row)
            checkpoint_rows.append(row)
            completed = len(completed_ids) + index
            if completed % args.progress_every == 0 or completed == len(tasks):
                print(f"Checkpointed {completed}/{len(tasks)}", flush=True)
    rows = sorted(checkpoint_rows, key=lambda row: str(row["task_id"]))
    _write_rows(raw_path, rows, RAW_FIELDNAMES)
    _write_rows(summary_path, summarize(rows), SUMMARY_FIELDNAMES)
    print(f"Completed {len(rows)} episodes in {perf_counter()-started:.1f} s", flush=True)
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
