"""Shared development episode execution, diagnostics, and aggregation for fixed-controller tuning."""

from __future__ import annotations
from experiment_config import FIXED_GAP, PREDICTIVE, UNCERTAINTY_AWARE, FROZEN_CROSSING_TIME, CURRENT_GAP_MARGIN, BRAKING_THRESHOLDS, RealismCandidate


import csv


from dataclasses import dataclass

from math import cos, hypot, isfinite, sin, sqrt

from pathlib import Path

from statistics import mean

from time import perf_counter

from fixed_gap_controller import ControllerState, DEFAULT_DRIVER_PROFILE_PROBABILITIES, FixedGapConfig, FixedGapController, ego_has_cleared_conflict_zone

from predictive_fixed_margin_controller import DEFAULT_FIXED_CLEARANCE, PredictiveFixedMarginConfig, PredictiveFixedMarginController

from scenarios import make_environment

from safety_metrics import PET_DEFINITION, GroundTruthSafetyTracker, SafetyMetricConfig, capture_policy_step_safety_metrics, observed_conflict_zone_ttc, serialize_ego_hard_braking_events

from state_estimation import KalmanObservationProvider, make_shared_kalman_observation_provider

from trajectory_prediction import DEFAULT_PREDICTION_CONFIG

from uncertainty_aware_predictive_controller import FROZEN_BASE_CLEARANCE, FROZEN_UNCERTAINTY_MULTIPLIER, UncertaintyAwarePredictiveConfig, UncertaintyAwarePredictiveController

from vehicle_observations import DelayedCorrelatedGaussianObservationProvider, ObservationLatencyConfig

@dataclass(frozen=True)
class PilotTask:
    candidate: RealismCandidate
    controller: str
    seed: int
    duration: float
    initial_vehicle_count: int
    spawn_probability: float
    safety_margin: float = CURRENT_GAP_MARGIN
    fixed_clearance: float = DEFAULT_FIXED_CLEARANCE
    base_clearance: float = FROZEN_BASE_CLEARANCE
    uncertainty_multiplier: float = (
        FROZEN_UNCERTAINTY_MULTIPLIER
    )

@dataclass(frozen=True)
class EstimationAccumulator:
    # Sensor error compares each noisy measurement with truth at the
    # measurement timestamp.  Stale and filtered errors both compare with
    # truth at the delivery timestamp, making their comparison time-aligned.
    sensor_position_sse: float = 0.0
    sensor_speed_sse: float = 0.0
    sensor_samples: int = 0
    stale_position_sse: float = 0.0
    stale_speed_sse: float = 0.0
    filtered_position_sse: float = 0.0
    filtered_speed_sse: float = 0.0
    current_samples: int = 0
    absolute_position_bias_sum: float = 0.0
    absolute_speed_bias_sum: float = 0.0
    bias_samples: int = 0
    latency_sum: float = 0.0
    latency_samples: int = 0
    maximum_latency: float = 0.0

def _longitudinal_speed(vehicle) -> float:
    return float(
        vehicle.velocity[0] * cos(vehicle.heading)
        + vehicle.velocity[1] * sin(vehicle.heading)
    )

def _updated_estimation_accumulator(
    accumulator: EstimationAccumulator,
    provider: KalmanObservationProvider,
    env,
) -> EstimationAccumulator:
    measurement = provider.last_measurement
    if measurement is None:
        raise RuntimeError("Kalman provider did not retain its measurement")
    source = provider.measurement_provider
    if not isinstance(source, DelayedCorrelatedGaussianObservationProvider):
        raise TypeError("pilot requires the delayed correlated-noise provider")

    sensor_position_sse = accumulator.sensor_position_sse
    sensor_speed_sse = accumulator.sensor_speed_sse
    sensor_samples = accumulator.sensor_samples
    for measured in measurement.other_vehicles:
        # When measurement and delivery coincide, use the exact live state.
        # This avoids reconstructing the present from highway-env's history,
        # whose newest entry can lag the live vehicle by one physics sample.
        truth_vehicle = source._truth_vehicle_by_id.get(measured.vehicle_id)
        if (
            measurement.latency <= 1e-9
            and truth_vehicle is not None
            and truth_vehicle in env.road.vehicles
        ):
            truth = truth_vehicle
        else:
            truth = source._state_at_time(
                measured.vehicle_id,
                measurement.timestamp,
            )
        if truth is None:
            continue
        position_error = hypot(
            measured.position[0] - truth.position[0],
            measured.position[1] - truth.position[1],
        )
        speed_error = _longitudinal_speed(measured) - _longitudinal_speed(truth)
        sensor_position_sse += position_error**2
        sensor_speed_sse += speed_error**2
        sensor_samples += 1

    measurement_by_id = {
        measured.vehicle_id: measured
        for measured in measurement.other_vehicles
    }
    stale_position_sse = accumulator.stale_position_sse
    stale_speed_sse = accumulator.stale_speed_sse
    filtered_position_sse = accumulator.filtered_position_sse
    filtered_speed_sse = accumulator.filtered_speed_sse
    current_samples = accumulator.current_samples
    absolute_position_bias_sum = accumulator.absolute_position_bias_sum
    absolute_speed_bias_sum = accumulator.absolute_speed_bias_sum
    bias_samples = accumulator.bias_samples
    for estimate in provider.last_estimates:
        vehicle_id = estimate.vehicle.vehicle_id
        measured = measurement_by_id.get(vehicle_id)
        truth_vehicle = source._truth_vehicle_by_id.get(vehicle_id)
        if (
            measured is None
            or truth_vehicle is None
            or truth_vehicle not in env.road.vehicles
        ):
            continue

        stale_position_error = hypot(
            measured.position[0] - float(truth_vehicle.position[0]),
            measured.position[1] - float(truth_vehicle.position[1]),
        )
        stale_speed_error = (
            _longitudinal_speed(measured) - _longitudinal_speed(truth_vehicle)
        )
        stale_position_sse += stale_position_error**2
        stale_speed_sse += stale_speed_error**2

        filtered = estimate.vehicle
        position_error = hypot(
            filtered.position[0] - float(truth_vehicle.position[0]),
            filtered.position[1] - float(truth_vehicle.position[1]),
        )
        speed_error = (
            _longitudinal_speed(filtered) - _longitudinal_speed(truth_vehicle)
        )
        filtered_position_sse += position_error**2
        filtered_speed_sse += speed_error**2
        current_samples += 1
        absolute_position_bias_sum += abs(estimate.state.position_bias)
        absolute_speed_bias_sum += abs(estimate.state.speed_bias)
        bias_samples += 1

    return EstimationAccumulator(
        sensor_position_sse=sensor_position_sse,
        sensor_speed_sse=sensor_speed_sse,
        sensor_samples=sensor_samples,
        stale_position_sse=stale_position_sse,
        stale_speed_sse=stale_speed_sse,
        filtered_position_sse=filtered_position_sse,
        filtered_speed_sse=filtered_speed_sse,
        current_samples=current_samples,
        absolute_position_bias_sum=absolute_position_bias_sum,
        absolute_speed_bias_sum=absolute_speed_bias_sum,
        bias_samples=bias_samples,
        latency_sum=accumulator.latency_sum + measurement.latency,
        latency_samples=accumulator.latency_samples + 1,
        maximum_latency=max(accumulator.maximum_latency, measurement.latency),
    )

def _rmse(sse: float, samples: int) -> float | None:
    return sqrt(sse / samples) if samples else None

def _optional_mean(values: list[float]) -> float | None:
    return mean(values) if values else None

def _make_controller(task: PilotTask, provider: KalmanObservationProvider):
    if task.controller == FIXED_GAP:
        return FixedGapController(
            FixedGapConfig(
                crossing_time=FROZEN_CROSSING_TIME,
                safety_margin=task.safety_margin,
            ),
            observation_provider=provider,
        )
    if task.controller == PREDICTIVE:
        return PredictiveFixedMarginController(
            PredictiveFixedMarginConfig(
                clearance=task.fixed_clearance,
                prediction=DEFAULT_PREDICTION_CONFIG,
            ),
            observation_provider=provider,
        )
    if task.controller == UNCERTAINTY_AWARE:
        return UncertaintyAwarePredictiveController(
            UncertaintyAwarePredictiveConfig(
                base_clearance=task.base_clearance,
                uncertainty_multiplier=task.uncertainty_multiplier,
                prediction=DEFAULT_PREDICTION_CONFIG,
            ),
            observation_provider=provider,
        )
    raise ValueError(f"unknown controller: {task.controller}")

def run_pilot_task(task: PilotTask) -> dict[str, object]:
    """Run one headless paired-development episode."""
    noise = task.candidate.noise
    provider = make_shared_kalman_observation_provider(
        noise=noise,
        latency=ObservationLatencyConfig(task.candidate.latency),
        acceleration_process_std=task.candidate.acceleration_process_std,
        seed=task.seed,
    )
    controller = _make_controller(task, provider)
    env = make_environment(
        None,
        duration=task.duration,
        initial_vehicle_count=task.initial_vehicle_count,
        spawn_probability=task.spawn_probability,
        driver_profile_probabilities=DEFAULT_DRIVER_PROFILE_PROBABILITIES,
    )
    trackers = {
        threshold: GroundTruthSafetyTracker(
            env,
            config=SafetyMetricConfig(hard_braking_threshold=threshold),
        )
        for threshold in BRAKING_THRESHOLDS
    }
    estimation = EstimationAccumulator()
    waiting_time = 0.0
    decision_times_ms: list[float] = []
    commit_time: float | None = None
    clear_time: float | None = None
    minimum_predicted_clearance_at_commit: float | None = None
    physical_overlap_predicted_at_commit: bool | None = None
    maximum_required_clearance_at_commit: float | None = None
    minimum_clearance_slack_at_commit: float | None = None
    maximum_position_std_at_commit: float | None = None
    all_background_vehicles: set[object] = set()

    try:
        env.reset(seed=task.seed)
        env.road.record_history = True
        controller.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            decision_start = perf_counter()
            decision = controller.decide(env)
            decision_times_ms.append((perf_counter() - decision_start) * 1000)
            estimation = _updated_estimation_accumulator(
                estimation,
                provider,
                env,
            )

            if decision.state is ControllerState.WAIT:
                waiting_time += 1.0 / float(env.config["policy_frequency"])
            if decision.state is ControllerState.COMMIT and commit_time is None:
                commit_time = float(env.time)
                minimum_predicted_clearance_at_commit = getattr(
                    decision, "minimum_predicted_clearance", None
                )
                physical_overlap_predicted_at_commit = getattr(
                    decision, "physical_overlap_predicted", None
                )
                maximum_required_clearance_at_commit = getattr(
                    decision,
                    "maximum_required_clearance",
                    None,
                )
                minimum_clearance_slack_at_commit = getattr(
                    decision,
                    "minimum_clearance_slack",
                    None,
                )
                adaptive_clearances = getattr(
                    decision,
                    "adaptive_clearances",
                    (),
                )
                if adaptive_clearances:
                    maximum_position_std_at_commit = max(
                        profile.maximum_position_std
                        for profile in adaptive_clearances
                    )

            step_start_time = float(env.time)
            if controller.last_observation is None:
                raise RuntimeError("controller did not retain its observation")
            perceived_ttc = observed_conflict_zone_ttc(
                env,
                controller.last_observation,
            )
            associations = {}
            for threshold, tracker in trackers.items():
                associated = tracker.associate_background_vehicles(env)
                associations[threshold] = associated
                all_background_vehicles.update(
                    vehicle for _, vehicle in associated
                )
                tracker.observe(
                    absolute_time=step_start_time,
                    ego=env.vehicle,
                    background_vehicles=associated,
                    ego_committed=decision.state is ControllerState.COMMIT,
                    env=env,
                    perceived_ttc=perceived_ttc,
                    controller_state=decision.state.value,
                    action_name=decision.action_name,
                )

            _, _, terminated, truncated, _ = env.step(decision.action)
            for threshold, tracker in trackers.items():
                capture_policy_step_safety_metrics(
                    env=env,
                    tracker=tracker,
                    step_start_time=step_start_time,
                    background_vehicles=associations[threshold],
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
        base_safety = trackers[3.0].result()
        threshold_results = {
            threshold: tracker.result()
            for threshold, tracker in trackers.items()
        }
        completion_time = float(env.time) if success else None
        commit_to_clear_time = (
            clear_time - commit_time
            if clear_time is not None and commit_time is not None
            else None
        )
        row: dict[str, object] = {
            "candidate": task.candidate.name,
            "controller": task.controller,
            "seed": task.seed,
            "traffic_profile": f"spawn_{task.spawn_probability:.1f}",
            "duration": task.duration,
            "initial_vehicle_count": task.initial_vehicle_count,
            "spawn_probability": task.spawn_probability,
            "correlation": task.candidate.correlation,
            "position_bias_innovation_std": (
                task.candidate.position_bias_innovation_std
            ),
            "position_white_std": task.candidate.position_white_std,
            "position_stationary_bias_std": noise.position_stationary_bias_std,
            "position_measurement_std": noise.position_measurement_std,
            "speed_bias_innovation_std": (
                task.candidate.speed_bias_innovation_std
            ),
            "speed_white_std": task.candidate.speed_white_std,
            "speed_stationary_bias_std": noise.speed_stationary_bias_std,
            "speed_measurement_std": noise.speed_measurement_std,
            "configured_latency": task.candidate.latency,
            "acceleration_process_std": (
                task.candidate.acceleration_process_std
            ),
            "safety_margin": (
                task.safety_margin if task.controller == FIXED_GAP else None
            ),
            "fixed_clearance": (
                task.fixed_clearance
                if task.controller == PREDICTIVE
                else None
            ),
            "base_clearance": (
                task.base_clearance
                if task.controller == UNCERTAINTY_AWARE
                else None
            ),
            "uncertainty_multiplier": (
                task.uncertainty_multiplier
                if task.controller == UNCERTAINTY_AWARE
                else None
            ),
            "pet_definition": PET_DEFINITION,
            "success": success,
            "collision": collision,
            "background_collision": any(
                bool(getattr(vehicle, "crashed", False))
                for vehicle in all_background_vehicles
            ),
            "timeout": timeout,
            "episode_time": float(env.time),
            "completion_time": completion_time,
            "waiting_time": waiting_time,
            "commit_time": commit_time,
            "clear_time": clear_time,
            "commit_to_clear_time": commit_to_clear_time,
            "minimum_predicted_clearance_at_commit": (
                minimum_predicted_clearance_at_commit
            ),
            "physical_overlap_predicted_at_commit": (
                physical_overlap_predicted_at_commit
            ),
            "maximum_required_clearance_at_commit": (
                maximum_required_clearance_at_commit
            ),
            "minimum_clearance_slack_at_commit": (
                minimum_clearance_slack_at_commit
            ),
            "maximum_position_std_at_commit": (
                maximum_position_std_at_commit
            ),
            "minimum_actual_ttc": base_safety.minimum_actual_ttc,
            "minimum_actual_ttc_during_commit": (
                base_safety.minimum_actual_ttc_during_commit
            ),
            "minimum_pet": base_safety.minimum_pet,
            "simultaneous_zone_occupancy": (
                base_safety.simultaneous_zone_occupancy
            ),
            "pet_vehicle_pairs": base_safety.pet_vehicle_pairs,
            "safety_metric_physics_samples": base_safety.physics_samples,
            "sensor_position_sse": estimation.sensor_position_sse,
            "sensor_speed_sse": estimation.sensor_speed_sse,
            "sensor_samples": estimation.sensor_samples,
            "stale_position_sse": estimation.stale_position_sse,
            "stale_speed_sse": estimation.stale_speed_sse,
            "filtered_position_sse": estimation.filtered_position_sse,
            "filtered_speed_sse": estimation.filtered_speed_sse,
            "current_samples": estimation.current_samples,
            "sensor_position_rmse": _rmse(
                estimation.sensor_position_sse,
                estimation.sensor_samples,
            ),
            "sensor_speed_rmse": _rmse(
                estimation.sensor_speed_sse,
                estimation.sensor_samples,
            ),
            "stale_position_rmse": _rmse(
                estimation.stale_position_sse,
                estimation.current_samples,
            ),
            "stale_speed_rmse": _rmse(
                estimation.stale_speed_sse,
                estimation.current_samples,
            ),
            "filtered_position_rmse": _rmse(
                estimation.filtered_position_sse,
                estimation.current_samples,
            ),
            "filtered_speed_rmse": _rmse(
                estimation.filtered_speed_sse,
                estimation.current_samples,
            ),
            "mean_absolute_estimated_position_bias": (
                estimation.absolute_position_bias_sum / estimation.bias_samples
                if estimation.bias_samples
                else None
            ),
            "mean_absolute_estimated_speed_bias": (
                estimation.absolute_speed_bias_sum / estimation.bias_samples
                if estimation.bias_samples
                else None
            ),
            "mean_effective_latency": (
                estimation.latency_sum / estimation.latency_samples
                if estimation.latency_samples
                else 0.0
            ),
            "maximum_effective_latency": estimation.maximum_latency,
            "mean_decision_time_ms": mean(decision_times_ms),
            "maximum_decision_time_ms": max(decision_times_ms),
        }
        for threshold, result in threshold_results.items():
            suffix = str(threshold).replace(".", "p")
            events = result.ego_hard_braking_event_records
            actual_ttc_values = [
                event.actual_ttc
                for event in events
                if event.actual_ttc is not None
            ]
            perceived_ttc_values = [
                event.perceived_ttc
                for event in events
                if event.perceived_ttc is not None
            ]
            row[f"ego_hard_braking_events_{suffix}"] = (
                result.ego_hard_braking_events
            )
            row[f"ego_hard_braking_event_log_{suffix}"] = (
                serialize_ego_hard_braking_events(events)
            )
            row[f"ego_hard_braking_actual_ttc_sum_{suffix}"] = sum(
                actual_ttc_values
            )
            row[f"ego_hard_braking_actual_ttc_samples_{suffix}"] = len(
                actual_ttc_values
            )
            row[f"ego_hard_braking_perceived_ttc_sum_{suffix}"] = sum(
                perceived_ttc_values
            )
            row[f"ego_hard_braking_perceived_ttc_samples_{suffix}"] = len(
                perceived_ttc_values
            )
            row[f"ego_hard_braking_actual_ttc_le_1s_{suffix}"] = sum(
                event.actual_ttc is not None and event.actual_ttc <= 1.0
                for event in events
            )
            row[f"ego_hard_braking_actual_ttc_le_2s_{suffix}"] = sum(
                event.actual_ttc is not None and event.actual_ttc <= 2.0
                for event in events
            )
            row[f"ego_hard_braking_actual_ttc_le_3s_{suffix}"] = sum(
                event.actual_ttc is not None and event.actual_ttc <= 3.0
                for event in events
            )
            row[f"ego_hard_braking_actual_ttc_gt_3s_or_none_{suffix}"] = sum(
                event.actual_ttc is None or event.actual_ttc > 3.0
                for event in events
            )
            row[
                f"ego_hard_braking_perceived_le_3s_actual_gt_3s_or_none_{suffix}"
            ] = sum(
                event.perceived_ttc is not None
                and event.perceived_ttc <= 3.0
                and (event.actual_ttc is None or event.actual_ttc > 3.0)
                for event in events
            )
            row[f"maximum_ego_hard_braking_deceleration_{suffix}"] = (
                result.maximum_ego_hard_braking_deceleration
            )
        return row
    finally:
        env.close()

def _aggregate_rmse(
    rows: list[dict[str, object]],
    *,
    sse_key: str,
    count_key: str,
) -> float | None:
    samples = sum(int(row[count_key]) for row in rows)
    if not samples:
        return None
    return sqrt(sum(float(row[sse_key]) for row in rows) / samples)

def summarize_group(
    rows: list[dict[str, object]],
    *,
    traffic_profile: str,
    spawn_probability: float | str,
) -> dict[str, object]:
    first = rows[0]
    completion_times = [
        float(row["completion_time"])
        for row in rows
        if row["success"] and row["completion_time"] is not None
    ]
    pet_values = [
        float(row["minimum_pet"])
        for row in rows
        if row["minimum_pet"] is not None
    ]
    ttc_values = [
        float(row["minimum_actual_ttc"])
        for row in rows
        if row["minimum_actual_ttc"] is not None
    ]
    commit_ttc_values = [
        float(row["minimum_actual_ttc_during_commit"])
        for row in rows
        if row["minimum_actual_ttc_during_commit"] is not None
    ]
    predicted_clearance_values = [
        float(row["minimum_predicted_clearance_at_commit"])
        for row in rows
        if row["minimum_predicted_clearance_at_commit"] is not None
        and isfinite(float(row["minimum_predicted_clearance_at_commit"]))
    ]
    predicted_overlap_values = [
        bool(row["physical_overlap_predicted_at_commit"])
        for row in rows
        if row["physical_overlap_predicted_at_commit"] is not None
    ]
    required_clearance_values = [
        float(row["maximum_required_clearance_at_commit"])
        for row in rows
        if row["maximum_required_clearance_at_commit"] is not None
        and isfinite(float(row["maximum_required_clearance_at_commit"]))
    ]
    clearance_slack_values = [
        float(row["minimum_clearance_slack_at_commit"])
        for row in rows
        if row["minimum_clearance_slack_at_commit"] is not None
        and isfinite(float(row["minimum_clearance_slack_at_commit"]))
    ]
    maximum_position_std_values = [
        float(row["maximum_position_std_at_commit"])
        for row in rows
        if row["maximum_position_std_at_commit"] is not None
        and isfinite(float(row["maximum_position_std_at_commit"]))
    ]
    summary = {
        key: first[key]
        for key in (
            "candidate",
            "controller",
            "correlation",
            "position_bias_innovation_std",
            "position_white_std",
            "position_stationary_bias_std",
            "position_measurement_std",
            "speed_bias_innovation_std",
            "speed_white_std",
            "speed_stationary_bias_std",
            "speed_measurement_std",
            "configured_latency",
            "acceleration_process_std",
            "safety_margin",
            "fixed_clearance",
            "base_clearance",
            "uncertainty_multiplier",
            "pet_definition",
        )
    }
    summary.update(
        {
            "traffic_profile": traffic_profile,
            "spawn_probability": spawn_probability,
            "episodes": len(rows),
            "success_rate": sum(bool(row["success"]) for row in rows)
            / len(rows),
            "collision_rate": sum(bool(row["collision"]) for row in rows)
            / len(rows),
            "background_collision_rate": sum(
                bool(row["background_collision"]) for row in rows
            )
            / len(rows),
            "timeout_rate": sum(bool(row["timeout"]) for row in rows)
            / len(rows),
            "mean_waiting_time": mean(
                float(row["waiting_time"]) for row in rows
            ),
            "mean_successful_completion_time": _optional_mean(
                completion_times
            ),
            "minimum_actual_ttc": min(ttc_values) if ttc_values else None,
            "minimum_actual_ttc_during_commit": (
                min(commit_ttc_values) if commit_ttc_values else None
            ),
            "minimum_pet": min(pet_values) if pet_values else None,
            "minimum_predicted_clearance_at_commit": (
                min(predicted_clearance_values)
                if predicted_clearance_values
                else None
            ),
            "predicted_overlap_at_commit_rate": (
                sum(predicted_overlap_values) / len(predicted_overlap_values)
                if predicted_overlap_values
                else None
            ),
            "mean_maximum_required_clearance_at_commit": (
                mean(required_clearance_values)
                if required_clearance_values
                else None
            ),
            "maximum_required_clearance_at_commit": (
                max(required_clearance_values)
                if required_clearance_values
                else None
            ),
            "minimum_clearance_slack_at_commit": (
                min(clearance_slack_values)
                if clearance_slack_values
                else None
            ),
            "mean_maximum_position_std_at_commit": (
                mean(maximum_position_std_values)
                if maximum_position_std_values
                else None
            ),
            "simultaneous_zone_occupancy_rate": sum(
                bool(row["simultaneous_zone_occupancy"]) for row in rows
            )
            / len(rows),
            "sensor_position_rmse": _aggregate_rmse(
                rows,
                sse_key="sensor_position_sse",
                count_key="sensor_samples",
            ),
            "sensor_speed_rmse": _aggregate_rmse(
                rows,
                sse_key="sensor_speed_sse",
                count_key="sensor_samples",
            ),
            "stale_position_rmse": _aggregate_rmse(
                rows,
                sse_key="stale_position_sse",
                count_key="current_samples",
            ),
            "stale_speed_rmse": _aggregate_rmse(
                rows,
                sse_key="stale_speed_sse",
                count_key="current_samples",
            ),
            "filtered_position_rmse": _aggregate_rmse(
                rows,
                sse_key="filtered_position_sse",
                count_key="current_samples",
            ),
            "filtered_speed_rmse": _aggregate_rmse(
                rows,
                sse_key="filtered_speed_sse",
                count_key="current_samples",
            ),
            "mean_effective_latency": mean(
                float(row["mean_effective_latency"]) for row in rows
            ),
            "mean_decision_time_ms": mean(
                float(row["mean_decision_time_ms"]) for row in rows
            ),
            "maximum_decision_time_ms": max(
                float(row["maximum_decision_time_ms"]) for row in rows
            ),
        }
    )
    for threshold in BRAKING_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        summary[f"mean_ego_hard_braking_events_{suffix}"] = mean(
            int(row[f"ego_hard_braking_events_{suffix}"]) for row in rows
        )
        actual_ttc_samples = sum(
            int(row[f"ego_hard_braking_actual_ttc_samples_{suffix}"])
            for row in rows
        )
        perceived_ttc_samples = sum(
            int(row[f"ego_hard_braking_perceived_ttc_samples_{suffix}"])
            for row in rows
        )
        actual_ttc_sum = sum(
            float(row[f"ego_hard_braking_actual_ttc_sum_{suffix}"])
            for row in rows
        )
        perceived_ttc_sum = sum(
            float(row[f"ego_hard_braking_perceived_ttc_sum_{suffix}"])
            for row in rows
        )
        summary[f"ego_hard_braking_actual_ttc_sum_{suffix}"] = actual_ttc_sum
        summary[f"ego_hard_braking_actual_ttc_samples_{suffix}"] = (
            actual_ttc_samples
        )
        summary[f"ego_hard_braking_perceived_ttc_sum_{suffix}"] = (
            perceived_ttc_sum
        )
        summary[f"ego_hard_braking_perceived_ttc_samples_{suffix}"] = (
            perceived_ttc_samples
        )
        summary[f"mean_actual_ttc_at_ego_hard_braking_{suffix}"] = (
            actual_ttc_sum / actual_ttc_samples
            if actual_ttc_samples
            else None
        )
        summary[f"mean_perceived_ttc_at_ego_hard_braking_{suffix}"] = (
            perceived_ttc_sum / perceived_ttc_samples
            if perceived_ttc_samples
            else None
        )
        for label in (
            "actual_ttc_le_1s",
            "actual_ttc_le_2s",
            "actual_ttc_le_3s",
            "actual_ttc_gt_3s_or_none",
            "perceived_le_3s_actual_gt_3s_or_none",
        ):
            summary[f"mean_ego_hard_braking_{label}_{suffix}"] = mean(
                int(row[f"ego_hard_braking_{label}_{suffix}"])
                for row in rows
            )
        decelerations = [
            float(row[f"maximum_ego_hard_braking_deceleration_{suffix}"])
            for row in rows
            if row[f"maximum_ego_hard_braking_deceleration_{suffix}"]
            is not None
        ]
        summary[f"maximum_ego_hard_braking_deceleration_{suffix}"] = (
            max(decelerations) if decelerations else None
        )
    return summary

def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output
