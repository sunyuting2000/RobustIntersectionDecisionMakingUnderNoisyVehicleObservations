"""Select and record auditable report-video replays from completed experiment CSVs.

Use --list to inspect selections; the full report results provide up to 68 clips.
Completed clips are checkpointed and validated against their source rows."""

from __future__ import annotations

import argparse

import csv

import hashlib


from dataclasses import dataclass

from math import isclose

from pathlib import Path

from statistics import mean

from time import perf_counter

from typing import Iterable, Mapping

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

PROJECT_DIRECTORY = SCRIPT_DIRECTORY



from gymnasium.wrappers import RecordVideo

from fixed_gap_controller import ControllerState, FastDriverVehicle, FixedGapConfig, FixedGapController, NormalDriverVehicle, SlowDriverVehicle, ego_has_cleared_conflict_zone, place_ego_at_stop_line

from experiment_config import EVALUATION_EGO_HARD_BRAKING_THRESHOLD, EVALUATION_HAZARD_DURATION, EVALUATION_HAZARD_ENVIRONMENT_SEED, EVALUATION_HAZARD_VARIANTS

from experiment_config import FIXED_GAP, PREDICTIVE, UNCERTAINTY_AWARE

from predictive_fixed_margin_controller import FROZEN_REALISTIC_FIXED_CLEARANCE, PredictiveFixedMarginConfig, PredictiveFixedMarginController

from video_rendering import DiagnosticOverlay, ReplayDiagnostics, _display, _make_environment, _sensing_configuration, _update_diagnostics

from scenarios import HAZARD_BY_NAME

from safety_metrics import PET_DEFINITION, GroundTruthSafetyTracker, SafetyMetricConfig, capture_policy_step_safety_metrics, observed_conflict_zone_ttc

from state_estimation import make_shared_kalman_observation_provider

from trajectory_prediction import DEFAULT_PREDICTION_CONFIG

from uncertainty_aware_predictive_controller import DEFAULT_CONFLICT_WINDOW_HALF_WIDTH, UncertaintyAwarePredictiveConfig, UncertaintyAwarePredictiveController

from scenarios import STOP_LINE_CLEARANCE, make_empty_environment

from scenarios import _add_profile_vehicle

from vehicle_observations import ObservationLatencyConfig

RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"

FINAL_RAW_PATH = RESULTS_DIRECTORY / "baselines_raw.csv"

C3_RAW_PATH = RESULTS_DIRECTORY / "controller3_noise_study_raw.csv"

C3_SELECTED_PATH = (
    RESULTS_DIRECTORY / "controller3_noise_study_selected_parameters.csv"
)

DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "videos" / "experiment_evidence"

CHECKPOINT_NAME = "recording_checkpoint.csv"

FROZEN_CROSSING_TIME = 5.0

FROZEN_GAP_MARGIN = 0.0

C3_NOISE_LEVELS = ("exact", "low", "medium_q1p5", "high")

OUTCOMES = ("success", "collision", "timeout")

COLLECTIONS = (
    "all",
    "c1_c2_stochastic",
    "c1_c2_hazards",
    "c3_stochastic",
    "c3_hazards",
)

@dataclass(frozen=True)
class EvidenceSpec:
    """One replay selected from a completed raw-result row."""

    case_id: str
    collection: str
    source_file: Path
    source_sha256: str
    source_task_id: str
    kind: str
    controller: str
    sensing_profile: str
    noise_seed: int
    seed: int
    spawn_probability: float | None
    driver_shift: str | None
    hazard_variant: str | None
    base_clearance: float | None
    uncertainty_multiplier: float | None
    expected_outcome: str
    expected_episode_time: float
    expected_waiting_time: float
    expected_required_clearance: float | None
    expected_minimum_pet: float | None

@dataclass(frozen=True)
class AvailabilityRow:
    collection: str
    controller: str
    sensing_profile: str
    outcome: str
    available_source_rows: int
    selected_rows: int

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)

def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}

def _outcome(row: Mapping[str, str]) -> str:
    if _truth(row.get("collision")):
        return "collision"
    if _truth(row.get("success")):
        return "success"
    if _truth(row.get("timeout")):
        return "timeout"
    return "other"

def _safe_name(value: object) -> str:
    return (
        str(value)
        .replace("|", "-")
        .replace(".", "p")
        .replace("/", "-")
        .replace(" ", "-")
    )

def _row_sort_key(row: Mapping[str, str]) -> tuple[object, ...]:
    return (
        row.get("seed", ""),
        row.get("driver_shift", ""),
        row.get("sensor_shift", row.get("noise_level", "")),
        row.get("spawn_probability", ""),
        row.get("task_id", ""),
    )

def _choose_diverse(
    rows: Iterable[dict[str, str]],
    count: int,
) -> tuple[dict[str, str], ...]:
    """Select stable examples while preferring distinct conditions/seeds."""
    remaining = sorted(rows, key=_row_sort_key)
    if len(remaining) <= count:
        return tuple(remaining)
    chosen = [remaining.pop(0)]
    while remaining and len(chosen) < count:
        def diversity(candidate: Mapping[str, str]) -> tuple[object, ...]:
            feature_names = (
                "seed",
                "driver_shift",
                "sensor_shift",
                "noise_level",
                "spawn_probability",
            )
            distinct = sum(
                all(candidate.get(name) != prior.get(name) for prior in chosen)
                for name in feature_names
            )
            wait = _float(candidate.get("waiting_time")) or 0.0
            wait_spread = max(
                abs(wait - (_float(prior.get("waiting_time")) or 0.0))
                for prior in chosen
            )
            return (distinct, wait_spread, _row_sort_key(candidate))

        selected = max(remaining, key=diversity)
        chosen.append(selected)
        remaining.remove(selected)
    return tuple(chosen)

def _expected_required_clearance(
    row: Mapping[str, str],
    *,
    source: str,
) -> float | None:
    column = (
        "maximum_required_clearance_at_commit"
        if source == "c3"
        else "required_clearance_at_commit"
    )
    return _float(row.get(column))

def _final_noise_seed(row: Mapping[str, str]) -> int:
    if row["kind"] != "hazard":
        return int(row["seed"])
    variant = HAZARD_BY_NAME[row["hazard_variant"]]
    return int(
        variant.noise_seed
        if variant.noise_seed is not None
        else EVALUATION_HAZARD_ENVIRONMENT_SEED
    )

def _spec_from_final_row(
    row: dict[str, str],
    *,
    source_hash: str,
    collection: str,
    ordinal: int | None = None,
) -> EvidenceSpec:
    outcome = _outcome(row)
    if row["kind"] == "hazard":
        case_id = (
            f"{row['controller']}-hazard-{row['hazard_variant']}"
        )
    else:
        case_id = (
            f"{row['controller']}-stochastic-{outcome}-{ordinal:02d}-"
            f"seed-{row['seed']}-{row['driver_shift']}-"
            f"{row['sensor_shift']}-spawn-{_safe_name(row['spawn_probability'])}"
        )
    return EvidenceSpec(
        case_id=case_id,
        collection=collection,
        source_file=FINAL_RAW_PATH,
        source_sha256=source_hash,
        source_task_id=row["task_id"],
        kind=row["kind"],
        controller=row["controller"],
        sensing_profile=row["sensor_shift"],
        noise_seed=_final_noise_seed(row),
        seed=int(row["seed"]),
        spawn_probability=_float(row.get("spawn_probability")),
        driver_shift=row.get("driver_shift") or None,
        hazard_variant=row.get("hazard_variant") or None,
        base_clearance=None,
        uncertainty_multiplier=None,
        expected_outcome=outcome,
        expected_episode_time=float(row["episode_time"]),
        expected_waiting_time=float(row["waiting_time"]),
        expected_required_clearance=_expected_required_clearance(
            row,
            source="final",
        ),
        expected_minimum_pet=_float(row.get("minimum_pet")),
    )

def _build_final_specs(
    *,
    max_per_outcome: int,
) -> tuple[tuple[EvidenceSpec, ...], tuple[AvailabilityRow, ...]]:
    rows = _read_csv(FINAL_RAW_PATH)
    source_hash = _sha256(FINAL_RAW_PATH)
    specs: list[EvidenceSpec] = []
    availability: list[AvailabilityRow] = []
    for controller in (FIXED_GAP, PREDICTIVE):
        controller_rows = [
            row for row in rows if row["controller"] == controller
        ]
        stochastic = [
            row for row in controller_rows if row["kind"] == "stochastic"
        ]
        for outcome in OUTCOMES:
            candidates = [row for row in stochastic if _outcome(row) == outcome]
            selected = _choose_diverse(candidates, max_per_outcome)
            availability.append(
                AvailabilityRow(
                    collection="c1_c2_stochastic",
                    controller=controller,
                    sensing_profile="all_final_profiles",
                    outcome=outcome,
                    available_source_rows=len(candidates),
                    selected_rows=len(selected),
                )
            )
            for ordinal, row in enumerate(selected, start=1):
                specs.append(
                    _spec_from_final_row(
                        row,
                        source_hash=source_hash,
                        collection="c1_c2_stochastic",
                        ordinal=ordinal,
                    )
                )
        hazards = sorted(
            (row for row in controller_rows if row["kind"] == "hazard"),
            key=lambda row: row["hazard_variant"],
        )
        for row in hazards:
            specs.append(
                _spec_from_final_row(
                    row,
                    source_hash=source_hash,
                    collection="c1_c2_hazards",
                )
            )
    return tuple(specs), tuple(availability)

def _selected_c3_rows(
    raw_rows: list[dict[str, str]],
    selected_row: Mapping[str, str],
) -> list[dict[str, str]]:
    noise = selected_row["noise_level"]
    selected_b0 = float(selected_row["selected_base_clearance"])
    selected_k = float(selected_row["selected_uncertainty_multiplier"])
    rows = [
        row
        for row in raw_rows
        if row["noise_level"] == noise
        and isclose(float(row["base_clearance"]), selected_b0, abs_tol=1e-12)
        and isclose(
            float(row["uncertainty_multiplier"]),
            selected_k,
            abs_tol=1e-12,
        )
    ]
    refined = [row for row in rows if row["stage"] == "refined"]
    if refined:
        rows = refined
    if len(rows) != 306:
        raise RuntimeError(
            f"selected Controller 3 row set for {noise} has {len(rows)} "
            "episodes, expected 306"
        )
    return rows

def _spec_from_c3_row(
    row: dict[str, str],
    *,
    source_hash: str,
    collection: str,
    ordinal: int | None = None,
) -> EvidenceSpec:
    outcome = _outcome(row)
    noise = row["noise_level"]
    if row["kind"] == "hazard":
        case_id = f"controller3-{noise}-hazard-{row['hazard_variant']}"
        hazard_names = tuple(variant.name for variant in EVALUATION_HAZARD_VARIANTS)
        noise_seed = (
            EVALUATION_HAZARD_ENVIRONMENT_SEED
            + hazard_names.index(row["hazard_variant"])
        )
    else:
        case_id = (
            f"controller3-{noise}-stochastic-{outcome}-{ordinal:02d}-"
            f"seed-{row['seed']}-{row['driver_shift']}-"
            f"spawn-{_safe_name(row['spawn_probability'])}"
        )
        noise_seed = int(row["seed"])
    return EvidenceSpec(
        case_id=case_id,
        collection=collection,
        source_file=C3_RAW_PATH,
        source_sha256=source_hash,
        source_task_id=row["task_id"],
        kind=row["kind"],
        controller=UNCERTAINTY_AWARE,
        sensing_profile=noise,
        noise_seed=noise_seed,
        seed=int(row["seed"]),
        spawn_probability=_float(row.get("spawn_probability")),
        driver_shift=row.get("driver_shift") or None,
        hazard_variant=row.get("hazard_variant") or None,
        base_clearance=float(row["base_clearance"]),
        uncertainty_multiplier=float(row["uncertainty_multiplier"]),
        expected_outcome=outcome,
        expected_episode_time=float(row["episode_time"]),
        expected_waiting_time=float(row["waiting_time"]),
        expected_required_clearance=_expected_required_clearance(
            row,
            source="c3",
        ),
        expected_minimum_pet=_float(row.get("minimum_pet")),
    )

def _build_c3_specs(
    *,
    max_per_outcome: int,
) -> tuple[tuple[EvidenceSpec, ...], tuple[AvailabilityRow, ...]]:
    raw_rows = _read_csv(C3_RAW_PATH)
    selected_rows = {
        row["noise_level"]: row for row in _read_csv(C3_SELECTED_PATH)
    }
    source_hash = _sha256(C3_RAW_PATH)
    specs: list[EvidenceSpec] = []
    availability: list[AvailabilityRow] = []
    for noise in C3_NOISE_LEVELS:
        rows = _selected_c3_rows(raw_rows, selected_rows[noise])
        stochastic = [row for row in rows if row["kind"] == "stochastic"]
        for outcome in OUTCOMES:
            candidates = [row for row in stochastic if _outcome(row) == outcome]
            selected = _choose_diverse(candidates, max_per_outcome)
            availability.append(
                AvailabilityRow(
                    collection="c3_stochastic",
                    controller=UNCERTAINTY_AWARE,
                    sensing_profile=noise,
                    outcome=outcome,
                    available_source_rows=len(candidates),
                    selected_rows=len(selected),
                )
            )
            for ordinal, row in enumerate(selected, start=1):
                specs.append(
                    _spec_from_c3_row(
                        row,
                        source_hash=source_hash,
                        collection="c3_stochastic",
                        ordinal=ordinal,
                    )
                )
        hazards = sorted(
            (row for row in rows if row["kind"] == "hazard"),
            key=lambda row: row["hazard_variant"],
        )
        if len(hazards) != 6:
            raise RuntimeError(
                f"selected Controller 3 {noise} set has {len(hazards)} "
                "hazards, expected 6"
            )
        for row in hazards:
            specs.append(
                _spec_from_c3_row(
                    row,
                    source_hash=source_hash,
                    collection="c3_hazards",
                )
            )
    return tuple(specs), tuple(availability)

def build_specs(
    *,
    max_per_outcome: int,
) -> tuple[tuple[EvidenceSpec, ...], tuple[AvailabilityRow, ...]]:
    final_specs, final_availability = _build_final_specs(
        max_per_outcome=max_per_outcome
    )
    c3_specs, c3_availability = _build_c3_specs(
        max_per_outcome=max_per_outcome
    )
    specs = (*final_specs, *c3_specs)
    case_ids = [spec.case_id for spec in specs]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("evidence recording case IDs are not unique")
    return tuple(specs), (*final_availability, *c3_availability)

def _make_hazard_environment():
    return make_empty_environment(
        duration=EVALUATION_HAZARD_DURATION,
        seed=EVALUATION_HAZARD_ENVIRONMENT_SEED,
        render_mode="rgb_array",
    )

def _populate_hazard(spec: EvidenceSpec, env) -> tuple[object, ...]:
    if spec.hazard_variant is None:
        raise ValueError("hazard evidence spec has no variant")
    values = dict(HAZARD_BY_NAME[spec.hazard_variant].truth_parameters)
    place_ego_at_stop_line(env, STOP_LINE_CLEARANCE)
    if spec.hazard_variant in {
        "left_boundary_ttc_shift",
        "right_boundary_speed_shift",
    }:
        return (
            _add_profile_vehicle(
                env,
                vehicle_type=FastDriverVehicle,
                origin=int(values["origin"]),
                front_distance=float(values["front_distance"]),
                speed=float(values["speed"]),
            ),
        )
    if spec.hazard_variant in {
        "shifted_slow_leader_fast_follower",
        "shifted_leader_induced_braking",
    }:
        return (
            _add_profile_vehicle(
                env,
                vehicle_type=SlowDriverVehicle,
                origin=int(values["origin"]),
                front_distance=float(values["leader_front_distance"]),
                speed=float(values["leader_speed"]),
            ),
            _add_profile_vehicle(
                env,
                vehicle_type=FastDriverVehicle,
                origin=int(values["origin"]),
                front_distance=float(values["follower_front_distance"]),
                speed=float(values["follower_speed"]),
            ),
        )
    if spec.hazard_variant == "shifted_two_direction_arrival":
        return (
            _add_profile_vehicle(
                env,
                vehicle_type=FastDriverVehicle,
                origin=1,
                front_distance=float(values["left_front_distance"]),
                speed=float(values["left_speed"]),
            ),
            _add_profile_vehicle(
                env,
                vehicle_type=FastDriverVehicle,
                origin=3,
                front_distance=float(values["right_front_distance"]),
                speed=float(values["right_speed"]),
            ),
        )
    if spec.hazard_variant == "mirrored_biased_latency_boundary":
        return (
            _add_profile_vehicle(
                env,
                vehicle_type=NormalDriverVehicle,
                origin=int(values["origin"]),
                front_distance=float(values["front_distance"]),
                speed=float(values["speed"]),
            ),
        )
    raise ValueError(f"unknown hazard variant: {spec.hazard_variant}")

def _make_provider(spec: EvidenceSpec):
    noise, latency, acceleration_std, _ = _sensing_configuration(
        spec.sensing_profile
    )
    return make_shared_kalman_observation_provider(
        noise=noise,
        latency=ObservationLatencyConfig(latency),
        acceleration_process_std=acceleration_std,
        seed=spec.noise_seed,
    )

def _make_controller(spec: EvidenceSpec, provider):
    if spec.controller == FIXED_GAP:
        return FixedGapController(
            FixedGapConfig(
                crossing_time=FROZEN_CROSSING_TIME,
                safety_margin=FROZEN_GAP_MARGIN,
            ),
            observation_provider=provider,
        )
    if spec.controller == PREDICTIVE:
        return PredictiveFixedMarginController(
            PredictiveFixedMarginConfig(
                clearance=FROZEN_REALISTIC_FIXED_CLEARANCE,
                prediction=DEFAULT_PREDICTION_CONFIG,
            ),
            observation_provider=provider,
        )
    if spec.controller == UNCERTAINTY_AWARE:
        if (
            spec.base_clearance is None
            or spec.uncertainty_multiplier is None
        ):
            raise ValueError("Controller 3 evidence spec is missing b_0 or k")
        return UncertaintyAwarePredictiveController(
            UncertaintyAwarePredictiveConfig(
                base_clearance=spec.base_clearance,
                uncertainty_multiplier=spec.uncertainty_multiplier,
                prediction=DEFAULT_PREDICTION_CONFIG,
                conflict_window_half_width=(
                    DEFAULT_CONFLICT_WINDOW_HALF_WIDTH
                ),
            ),
            observation_provider=provider,
        )
    raise ValueError(f"unknown controller: {spec.controller}")

def _controller_line(spec: EvidenceSpec) -> str:
    if spec.controller == FIXED_GAP:
        return "Controller 1 | critical gap=5.00 s"
    if spec.controller == PREDICTIVE:
        return "Controller 2 | fixed clearance=1.50 m"
    return (
        f"Controller 3 | b0={spec.base_clearance:.2f} m | "
        f"k={spec.uncertainty_multiplier:.3f}"
    )

def _overlay_lines(
    spec: EvidenceSpec,
    decision: object,
    diagnostics: ReplayDiagnostics,
    *,
    simulation_time: float,
    waiting_time: float,
) -> tuple[str, ...]:
    state = getattr(decision, "state", None)
    state_text = getattr(state, "value", state)
    lines = [
        f"EXPERIMENT REPLAY | {spec.case_id}",
        _controller_line(spec),
        f"kind={spec.kind} | sensing={spec.sensing_profile} | seed={spec.seed}",
        f"t={simulation_time:.0f} s | state={state_text} | "
        f"action={getattr(decision, 'action_name', 'n/a')} | "
        f"wait={waiting_time:.0f} s",
    ]
    if spec.controller == FIXED_GAP:
        lines.extend(
            (
                f"Available gap={_display(diagnostics.available_gap)} s | "
                "critical=5.00 s",
                f"Reviewed outcome={spec.expected_outcome}",
            )
        )
    else:
        sigma = (
            "not used"
            if spec.controller == PREDICTIVE
            else f"{_display(diagnostics.maximum_conflict_sigma)} m"
        )
        lines.extend(
            (
                f"Conflict-window sigma={sigma}",
                f"Required={_display(diagnostics.required_clearance)} m | "
                f"predicted min={_display(diagnostics.minimum_predicted_clearance)} m",
                f"Clearance slack={_display(diagnostics.clearance_slack)} m | "
                f"reviewed outcome={spec.expected_outcome}",
            )
        )
    return tuple(lines)

def _video_path(spec: EvidenceSpec, output_directory: Path) -> Path:
    return output_directory / f"{spec.case_id}-episode-0.mp4"

def _validate_replay(
    spec: EvidenceSpec,
    *,
    outcome: str,
    episode_time: float,
    waiting_time: float,
    required_clearance: float | None,
    minimum_pet: float | None,
) -> None:
    if outcome != spec.expected_outcome:
        raise RuntimeError(
            f"{spec.case_id}: outcome {outcome} != source "
            f"{spec.expected_outcome}"
        )
    for label, actual, expected, tolerance in (
        ("episode_time", episode_time, spec.expected_episode_time, 1e-9),
        ("waiting_time", waiting_time, spec.expected_waiting_time, 1e-9),
    ):
        if not isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f"{spec.case_id}: {label} {actual} != source {expected}"
            )
    if spec.expected_required_clearance is None:
        if required_clearance is not None:
            raise RuntimeError(
                f"{spec.case_id}: unexpected committed clearance "
                f"{required_clearance}"
            )
    elif required_clearance is None or not isclose(
        required_clearance,
        spec.expected_required_clearance,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            f"{spec.case_id}: required clearance {required_clearance!r} != "
            f"source {spec.expected_required_clearance}"
        )
    if spec.expected_minimum_pet is None:
        if minimum_pet is not None:
            raise RuntimeError(
                f"{spec.case_id}: unexpected PET {minimum_pet}"
            )
    elif minimum_pet is None or not isclose(
        minimum_pet,
        spec.expected_minimum_pet,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            f"{spec.case_id}: PET {minimum_pet!r} != source "
            f"{spec.expected_minimum_pet}"
        )

def record_spec(
    spec: EvidenceSpec,
    *,
    output_directory: Path,
) -> dict[str, object]:
    """Record and validate one evidence replay."""
    if spec.kind == "stochastic":
        if spec.spawn_probability is None or spec.driver_shift is None:
            raise ValueError("stochastic evidence spec is incomplete")
        base_env = _make_environment(spec)  # type: ignore[arg-type]
    else:
        base_env = _make_hazard_environment()

    provider = _make_provider(spec)
    controller = _make_controller(spec, provider)
    overlay_env = DiagnosticOverlay(base_env)
    recorded_env = RecordVideo(
        overlay_env,
        video_folder=str(output_directory),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=spec.case_id,
        disable_logger=True,
    )
    base_env.set_record_video_wrapper(recorded_env)
    diagnostics = ReplayDiagnostics()
    waiting_time = 0.0
    commit_time: float | None = None
    clear_time: float | None = None
    commit_minimum_clearance: float | None = None
    commit_required_clearance: float | None = None
    commit_clearance_slack: float | None = None
    commit_maximum_sigma: float | None = None
    decision_times_ms: list[float] = []
    initial_vehicles: tuple[object, ...] = ()
    all_background_vehicles: set[object] = set()

    try:
        recorded_env.reset(seed=spec.seed)
        if spec.kind == "hazard":
            initial_vehicles = _populate_hazard(spec, base_env)
            all_background_vehicles.update(initial_vehicles)
        base_env.road.record_history = True
        controller.reset()
        controller.state = (
            ControllerState.APPROACH
            if spec.kind == "stochastic"
            else ControllerState.WAIT
        )
        tracker = GroundTruthSafetyTracker(
            base_env,
            config=SafetyMetricConfig(
                hard_braking_threshold=EVALUATION_EGO_HARD_BRAKING_THRESHOLD
            ),
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            decision_start = perf_counter()
            decision = controller.decide(base_env)
            decision_times_ms.append(
                (perf_counter() - decision_start) * 1000.0
            )
            if decision.state is ControllerState.WAIT:
                waiting_time += 1.0 / float(
                    base_env.config["policy_frequency"]
                )
            _update_diagnostics(spec, decision, diagnostics)  # type: ignore[arg-type]
            if decision.state is ControllerState.COMMIT and commit_time is None:
                commit_time = float(base_env.time)
                commit_minimum_clearance = (
                    diagnostics.minimum_predicted_clearance
                )
                commit_required_clearance = diagnostics.required_clearance
                commit_clearance_slack = diagnostics.clearance_slack
                commit_maximum_sigma = diagnostics.maximum_conflict_sigma

            overlay_env.set_lines(
                _overlay_lines(
                    spec,
                    decision,
                    diagnostics,
                    simulation_time=float(base_env.time),
                    waiting_time=waiting_time,
                )
            )
            if controller.last_observation is None:
                raise RuntimeError("controller did not retain its observation")
            step_start_time = float(base_env.time)
            perceived_ttc = observed_conflict_zone_ttc(
                base_env,
                controller.last_observation,
            )
            background_vehicles = tracker.associate_background_vehicles(base_env)
            all_background_vehicles.update(
                vehicle for _, vehicle in background_vehicles
            )
            tracker.observe(
                absolute_time=step_start_time,
                ego=base_env.vehicle,
                background_vehicles=background_vehicles,
                ego_committed=decision.state is ControllerState.COMMIT,
                env=base_env,
                perceived_ttc=perceived_ttc,
                controller_state=decision.state.value,
                action_name=decision.action_name,
            )
            _, _, terminated, truncated, _ = recorded_env.step(decision.action)
            capture_policy_step_safety_metrics(
                env=base_env,
                tracker=tracker,
                step_start_time=step_start_time,
                background_vehicles=background_vehicles,
                ego_committed=decision.state is ControllerState.COMMIT,
                perceived_ttc_at_step_start=perceived_ttc,
                controller_state=decision.state.value,
                action_name=decision.action_name,
            )
            if (
                ego_has_cleared_conflict_zone(base_env.vehicle)
                and clear_time is None
            ):
                clear_time = float(base_env.time)

        collision = bool(base_env.vehicle.crashed)
        success = bool(base_env.has_arrived(base_env.vehicle)) and not collision
        timeout = bool(truncated) and not collision and not success
        outcome = (
            "collision"
            if collision
            else "success"
            if success
            else "timeout"
            if timeout
            else "other"
        )
        episode_time = float(base_env.time)
        safety = tracker.result()
        background_collision = any(
            bool(getattr(vehicle, "crashed", False))
            for vehicle in all_background_vehicles
        )
    finally:
        recorded_env.close()

    _validate_replay(
        spec,
        outcome=outcome,
        episode_time=episode_time,
        waiting_time=waiting_time,
        required_clearance=commit_required_clearance,
        minimum_pet=safety.minimum_pet,
    )
    video_path = _video_path(spec, output_directory)
    if not video_path.exists():
        raise RuntimeError(f"expected video was not created: {video_path}")
    return {
        "case_id": spec.case_id,
        "collection": spec.collection,
        "source_file": str(spec.source_file),
        "source_sha256": spec.source_sha256,
        "source_task_id": spec.source_task_id,
        "kind": spec.kind,
        "controller": spec.controller,
        "sensing_profile": spec.sensing_profile,
        "noise_seed": spec.noise_seed,
        "seed": spec.seed,
        "spawn_probability": spec.spawn_probability,
        "driver_shift": spec.driver_shift,
        "hazard_variant": spec.hazard_variant,
        "base_clearance": spec.base_clearance,
        "uncertainty_multiplier": spec.uncertainty_multiplier,
        "expected_outcome": spec.expected_outcome,
        "replayed_outcome": outcome,
        "success": success,
        "collision": collision,
        "background_collision": background_collision,
        "timeout": timeout,
        "episode_time": episode_time,
        "waiting_time": waiting_time,
        "commit_time": commit_time,
        "clear_time": clear_time,
        "minimum_predicted_clearance_at_commit": commit_minimum_clearance,
        "required_clearance_at_commit": commit_required_clearance,
        "clearance_slack_at_commit": commit_clearance_slack,
        "maximum_conflict_window_position_std_at_commit": (
            commit_maximum_sigma
        ),
        "minimum_pet": safety.minimum_pet,
        "simultaneous_zone_occupancy": (
            safety.simultaneous_zone_occupancy
        ),
        "ego_hard_braking_events": safety.ego_hard_braking_events,
        "mean_decision_time_ms": mean(decision_times_ms),
        "maximum_decision_time_ms": max(decision_times_ms),
        "pet_definition": PET_DEFINITION,
        "video_path": str(video_path),
        "validation": "exact source outcome/episode/wait/PET; clearance tol=0.01m",
        "artifact_role": "post_hoc_reproducibility_replay",
    }

def _write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        if path.exists():
            path.unlink()
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)

def _read_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    return {row["case_id"]: row for row in rows}

def _spec_row(spec: EvidenceSpec) -> dict[str, object]:
    return {
        "case_id": spec.case_id,
        "collection": spec.collection,
        "source_file": str(spec.source_file),
        "source_sha256": spec.source_sha256,
        "source_task_id": spec.source_task_id,
        "kind": spec.kind,
        "controller": spec.controller,
        "sensing_profile": spec.sensing_profile,
        "seed": spec.seed,
        "spawn_probability": spec.spawn_probability,
        "driver_shift": spec.driver_shift,
        "hazard_variant": spec.hazard_variant,
        "base_clearance": spec.base_clearance,
        "uncertainty_multiplier": spec.uncertainty_multiplier,
        "expected_outcome": spec.expected_outcome,
        "expected_episode_time": spec.expected_episode_time,
        "expected_waiting_time": spec.expected_waiting_time,
        "expected_required_clearance": spec.expected_required_clearance,
        "expected_minimum_pet": spec.expected_minimum_pet,
    }

def _filter_specs(
    specs: tuple[EvidenceSpec, ...],
    *,
    collection: str,
    cases: tuple[str, ...],
) -> tuple[EvidenceSpec, ...]:
    if cases:
        by_id = {spec.case_id: spec for spec in specs}
        unknown = [case for case in cases if case not in by_id]
        if unknown:
            raise ValueError(f"unknown evidence case IDs: {', '.join(unknown)}")
        return tuple(by_id[case] for case in dict.fromkeys(cases))
    if collection == "all":
        return specs
    return tuple(spec for spec in specs if spec.collection == collection)

def _print_selection(
    specs: tuple[EvidenceSpec, ...],
    availability: tuple[AvailabilityRow, ...],
) -> None:
    print("Outcome availability and deterministic selection:")
    for row in availability:
        print(
            f"  {row.controller:30s} {row.sensing_profile:22s} "
            f"{row.outcome:9s}: available={row.available_source_rows:3d}, "
            f"selected={row.selected_rows}"
        )
    print(f"\nSelected videos: {len(specs)}")
    for spec in specs:
        detail = (
            spec.hazard_variant
            if spec.kind == "hazard"
            else spec.expected_outcome
        )
        print(
            f"  {spec.case_id}: {spec.controller}, {spec.sensing_profile}, "
            f"{detail}, source task={spec.source_task_id}"
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record auditable replays selected from completed CSV rows."
    )
    parser.add_argument(
        "--baseline-results", type=Path, default=FINAL_RAW_PATH,
        help="Completed fixed-controller raw CSV.",
    )
    parser.add_argument(
        "--study-results", type=Path, default=C3_RAW_PATH,
        help="Completed Controller 3 parameter-study raw CSV.",
    )
    parser.add_argument(
        "--selected-parameters", type=Path, default=C3_SELECTED_PATH,
        help="Selected Controller 3 parameters CSV.",
    )
    parser.add_argument(
        "--collection",
        choices=COLLECTIONS,
        default="all",
        help="Evidence subset to record (default: all).",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Record one generated case ID; repeat to select several.",
    )
    parser.add_argument(
        "--max-per-outcome",
        type=int,
        default=2,
        help="Maximum stochastic examples per outcome/profile (default: 2).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIRECTORY}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only selected videos and their checkpoint rows.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print availability and selected source task IDs without replaying.",
    )
    return parser.parse_args()

def main() -> None:
    global FINAL_RAW_PATH, C3_RAW_PATH, C3_SELECTED_PATH
    args = parse_args()
    FINAL_RAW_PATH = args.baseline_results.expanduser().resolve()
    C3_RAW_PATH = args.study_results.expanduser().resolve()
    C3_SELECTED_PATH = args.selected_parameters.expanduser().resolve()
    if args.max_per_outcome <= 0:
        raise ValueError("--max-per-outcome must be positive")
    all_specs, availability = build_specs(
        max_per_outcome=args.max_per_outcome
    )
    specs = _filter_specs(
        all_specs,
        collection=args.collection,
        cases=tuple(args.case),
    )
    if args.list:
        _print_selection(specs, availability)
        return

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / CHECKPOINT_NAME
    checkpoint = _read_checkpoint(checkpoint_path)

    if args.overwrite:
        for spec in specs:
            video_path = _video_path(spec, output_directory)
            if video_path.exists():
                video_path.unlink()
            checkpoint.pop(spec.case_id, None)
        _write_rows(checkpoint_path, checkpoint.values())

    availability_path = (
        output_directory / f"availability_{args.collection}.csv"
    )
    selected_collections = {spec.collection for spec in specs}
    selected_availability = (
        row for row in availability if row.collection in selected_collections
    )
    _write_rows(
        availability_path,
        (row.__dict__ for row in selected_availability),
    )
    _write_rows(
        output_directory / f"selection_{args.collection}.csv",
        (_spec_row(spec) for spec in specs),
    )

    for index, spec in enumerate(specs, start=1):
        video_path = _video_path(spec, output_directory)
        completed = checkpoint.get(spec.case_id)
        if completed is not None:
            if completed["source_sha256"] != spec.source_sha256:
                raise RuntimeError(
                    f"source hash changed for completed case {spec.case_id}"
                )
            if not video_path.exists():
                raise FileNotFoundError(
                    f"checkpointed video is missing: {video_path}"
                )
            print(
                f"[{index}/{len(specs)}] SKIP completed {spec.case_id}",
                flush=True,
            )
            continue
        if video_path.exists():
            raise FileExistsError(
                f"uncheckpointed video exists: {video_path}; use "
                "--overwrite for this selected collection"
            )

        print(
            f"[{index}/{len(specs)}] RECORD {spec.case_id} | "
            f"expected={spec.expected_outcome}",
            flush=True,
        )
        row = record_spec(spec, output_directory=output_directory)
        checkpoint[spec.case_id] = {
            key: "" if value is None else str(value)
            for key, value in row.items()
        }
        _write_rows(checkpoint_path, checkpoint.values())
        print(
            f"  PASS replay={row['replayed_outcome']} | "
            f"wait={row['waiting_time']:.0f} s | "
            f"PET={_display(row['minimum_pet'], digits=3)} s",
            flush=True,
        )

    selected_rows = [checkpoint[spec.case_id] for spec in specs]
    manifest_path = output_directory / f"manifest_{args.collection}.csv"
    _write_rows(manifest_path, selected_rows)
    print(f"\nCompleted {len(specs)} selected evidence videos.", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Availability: {availability_path}", flush=True)
    print(
        "Raw CSVs and summary tables contain the quantitative results; "
        "videos are validated illustrative replays.",
        flush=True,
    )

if __name__ == "__main__":
    main()
