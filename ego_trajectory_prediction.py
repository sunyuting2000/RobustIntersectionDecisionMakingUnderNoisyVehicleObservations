"""Isolated Highway-env rollout for the candidate ego CROSS plan.

The predictor reconstructs the observed ego on an isolated Road containing no
background vehicles. It shares only the read-only road-network geometry with
the real environment, copies the random-generator state, and advances the ego
with the same physics and COMMIT meta-action schedule as the real simulation.
"""

from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np

from highway_env.road.road import Road
from highway_env.vehicle.controller import MDPVehicle

from fixed_gap_controller import (
    EGO_EXIT_LANE,
    MajorRoadStraightIntersectionEnv,
)
from trajectory_prediction import DEFAULT_PREDICTION_CONFIG, PredictionConfig
from vehicle_observations import (
    CandidateTrajectory,
    ObservationSnapshot,
    PredictedVehicleState,
)


@dataclass(frozen=True)
class EgoTrajectoryPrediction:
    """Candidate CROSS trajectory plus horizon-coverage diagnostics."""

    trajectory: CandidateTrajectory
    cleared_conflict_zone: bool
    clear_time: float | None
    physics_time_step: float


@dataclass(frozen=True)
class _RawEgoState:
    """One state captured at the isolated physics-update frequency."""

    relative_time: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    heading: float


def _copy_ego_to_isolated_road(
    env: MajorRoadStraightIntersectionEnv,
    observation: ObservationSnapshot,
) -> tuple[Road, MDPVehicle]:
    """Reconstruct the observed ego on a road with no other vehicles."""
    source_ego = env.vehicle
    prediction_road = Road(
        network=env.road.network,
        vehicles=[],
        road_objects=[],
        np_random=deepcopy(env.road.np_random),
        record_history=False,
        neighbour_vehicles_connected_lanes=(
            env.road.neighbour_vehicles_connected_lanes
        ),
    )
    prediction_ego = MDPVehicle(
        road=prediction_road,
        position=np.asarray(observation.ego.position, dtype=float).copy(),
        heading=observation.ego.heading,
        speed=observation.ego.speed,
        target_lane_index=source_ego.target_lane_index,
        target_speed=float(source_ego.target_speed),
        target_speeds=np.asarray(source_ego.target_speeds, dtype=float).copy(),
        route=deepcopy(source_ego.route),
    )
    prediction_road.vehicles = [prediction_ego]
    return prediction_road, prediction_ego


def _has_cleared_conflict_zone(ego: MDPVehicle) -> bool:
    """Whether the predicted ego rear bumper has cleared the intersection."""
    if ego.lane_index[:2] != EGO_EXIT_LANE[:2]:
        return False
    longitudinal, _ = ego.lane.local_coordinates(ego.position)
    return longitudinal >= ego.LENGTH / 2


def _capture(relative_time: float, ego: MDPVehicle) -> _RawEgoState:
    return _RawEgoState(
        relative_time=relative_time,
        position=(float(ego.position[0]), float(ego.position[1])),
        velocity=(float(ego.velocity[0]), float(ego.velocity[1])),
        heading=float(ego.heading),
    )


def _commit_action(ego: MDPVehicle) -> str:
    """Mirror the controller's post-commit FASTER-then-IDLE policy."""
    maximum_speed = float(ego.target_speeds[-1])
    return "FASTER" if ego.target_speed < maximum_speed - 1e-6 else "IDLE"


def _interpolate_heading(start: float, end: float, fraction: float) -> float:
    """Interpolate angles along their shortest wrapped difference."""
    difference = atan2(sin(end - start), cos(end - start))
    return start + fraction * difference


def _interpolate_raw_state(
    raw_states: tuple[_RawEgoState, ...], relative_time: float
) -> _RawEgoState:
    """Interpolate a requested sample from fixed-frequency physics states."""
    raw_times = [state.relative_time for state in raw_states]
    index = bisect_left(raw_times, relative_time)
    tolerance = 1e-12

    if index == 0:
        return raw_states[0]
    if index == len(raw_states):
        return raw_states[-1]
    if abs(raw_times[index] - relative_time) <= tolerance:
        return raw_states[index]

    before = raw_states[index - 1]
    after = raw_states[index]
    fraction = (relative_time - before.relative_time) / (
        after.relative_time - before.relative_time
    )
    position = tuple(
        before.position[axis]
        + fraction * (after.position[axis] - before.position[axis])
        for axis in range(2)
    )
    velocity = tuple(
        before.velocity[axis]
        + fraction * (after.velocity[axis] - before.velocity[axis])
        for axis in range(2)
    )
    return _RawEgoState(
        relative_time=relative_time,
        position=(float(position[0]), float(position[1])),
        velocity=(float(velocity[0]), float(velocity[1])),
        heading=_interpolate_heading(before.heading, after.heading, fraction),
    )


def predict_ego_crossing(
    env: MajorRoadStraightIntersectionEnv,
    observation: ObservationSnapshot,
    *,
    config: PredictionConfig = DEFAULT_PREDICTION_CONFIG,
) -> EgoTrajectoryPrediction:
    """Roll out an isolated candidate CROSS plan without mutating ``env``."""
    simulation_frequency = int(env.config["simulation_frequency"])
    policy_frequency = int(env.config["policy_frequency"])
    if simulation_frequency <= 0 or policy_frequency <= 0:
        raise ValueError("simulation and policy frequencies must be positive")
    if simulation_frequency % policy_frequency:
        raise ValueError("simulation_frequency must be divisible by policy_frequency")

    physics_time_step = 1.0 / simulation_frequency
    frames_per_policy_action = simulation_frequency // policy_frequency
    maximum_frames = int(np.ceil(config.horizon * simulation_frequency - 1e-12))
    prediction_road, prediction_ego = _copy_ego_to_isolated_road(env, observation)

    raw_states = [_capture(0.0, prediction_ego)]
    clear_time: float | None = (
        0.0 if _has_cleared_conflict_zone(prediction_ego) else None
    )

    for frame in range(maximum_frames):
        if clear_time is not None:
            break
        if frame % frames_per_policy_action == 0:
            prediction_ego.act(_commit_action(prediction_ego))
        prediction_road.act()
        prediction_road.step(physics_time_step)

        relative_time = min((frame + 1) * physics_time_step, config.horizon)
        raw_states.append(_capture(relative_time, prediction_ego))
        if _has_cleared_conflict_zone(prediction_ego):
            clear_time = relative_time

    final_time = clear_time if clear_time is not None else config.horizon
    requested_times = [
        time for time in config.relative_times if time <= final_time + 1e-12
    ]
    if not requested_times or abs(requested_times[-1] - final_time) > 1e-12:
        requested_times.append(final_time)
    else:
        requested_times[-1] = final_time

    immutable_raw_states = tuple(raw_states)
    samples = tuple(
        PredictedVehicleState(
            vehicle_id=observation.ego.vehicle_id,
            relative_time=relative_time,
            position=interpolated.position,
            velocity=interpolated.velocity,
            heading=interpolated.heading,
            length=observation.ego.length,
            width=observation.ego.width,
        )
        for relative_time in requested_times
        for interpolated in (
            _interpolate_raw_state(immutable_raw_states, relative_time),
        )
    )
    trajectory = CandidateTrajectory(
        plan_name="CROSS",
        # Ego state is exact at controller delivery/decision time even when
        # background measurements carry a fixed delay.
        start_time=float(observation.delivery_timestamp),
        samples=samples,
    )
    return EgoTrajectoryPrediction(
        trajectory=trajectory,
        cleared_conflict_zone=clear_time is not None,
        clear_time=clear_time,
        physics_time_step=physics_time_step,
    )
