"""Fixed critical-gap controller for the CS5478 intersection project.

The controller implements the basic constant critical-gap rule

    cross if min(TTC_j) >= measured_crossing_time + safety_margin

for straight-through major-road traffic. Route information is used to keep
only vehicles whose paths conflict with the ego vehicle's straight crossing.
The default exact input passes through the shared Kalman observation layer;
experiments can inject the correlated-noise and latency pipeline explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import inf, isfinite
from typing import ClassVar

import numpy as np

from highway_env.envs.intersection_env import ConnectedLaneIntersectionEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

from state_estimation import KalmanObservationProvider
from vehicle_observations import (
    ObservationSnapshot,
    ObservedVehicleState,
    VehicleObservationProvider,
)


# The ego enters from the south (o0) and exits to the north (o2).
EGO_INCOMING_LANE = ("o0", "ir0", 0)
EGO_INTERSECTION_LANE = ("ir0", "il2", 0)
EGO_EXIT_LANE = ("il2", "o2", 0)

# Background vehicles travel straight on the horizontal, higher-priority road.
MAJOR_INCOMING_EDGES = {("o1", "ir1"), ("o3", "ir3")}
MAJOR_INTERSECTION_EDGES = {("ir1", "il3"), ("ir3", "il1")}
MAJOR_EXIT_EDGES = {("il3", "o3"), ("il1", "o1")}

# Selected using the medium_q1p5 coarse/refined search on paired development
# seeds 0-49. Experiments can supply an explicit margin.
FROZEN_SAFETY_MARGIN = 0.0


@dataclass(frozen=True)
class DriverProfileDefinition:
    """Frozen IDM parameters for one background-driver profile.

    These values define simulation truth.  A controller observes vehicle
    motion, but it is not given the profile name or these parameters.
    """

    name: str
    desired_speed: float
    maximum_acceleration: float
    comfortable_acceleration: float
    comfortable_deceleration: float
    standstill_distance: float
    time_headway: float
    delta_range: tuple[float, float] = (3.5, 4.5)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("driver profile name must not be empty")
        if self.desired_speed <= 0:
            raise ValueError("desired_speed must be positive")
        if self.maximum_acceleration <= 0:
            raise ValueError("maximum_acceleration must be positive")
        if not 0 < self.comfortable_acceleration <= self.maximum_acceleration:
            raise ValueError(
                "comfortable_acceleration must be positive and no greater "
                "than maximum_acceleration"
            )
        if self.comfortable_deceleration >= 0:
            raise ValueError("comfortable_deceleration must be negative")
        if self.standstill_distance <= 0:
            raise ValueError("standstill_distance must be positive")
        if self.time_headway <= 0:
            raise ValueError("time_headway must be positive")
        if not 0 < self.delta_range[0] <= self.delta_range[1]:
            raise ValueError("delta_range must contain ordered positive values")


# Calibrated on 90 development episodes covering 30 paired seeds at spawn
# probabilities 0.1, 0.3, and 0.6. The intersection
# speed limit is 10 m/s, so the fast target does not rely on limit clipping.
DRIVER_PROFILE_CONFIGURATION_FROZEN = True
SLOW_DRIVER_PROFILE = DriverProfileDefinition(
    name="slow",
    desired_speed=6.0,
    maximum_acceleration=4.0,
    comfortable_acceleration=2.0,
    comfortable_deceleration=-3.0,
    standstill_distance=10.0,
    time_headway=2.0,
)
NORMAL_DRIVER_PROFILE = DriverProfileDefinition(
    name="normal",
    desired_speed=8.0,
    maximum_acceleration=6.0,
    comfortable_acceleration=3.0,
    comfortable_deceleration=-4.0,
    standstill_distance=8.0,
    time_headway=1.5,
)
FAST_DRIVER_PROFILE = DriverProfileDefinition(
    name="fast",
    desired_speed=10.0,
    maximum_acceleration=6.0,
    comfortable_acceleration=4.0,
    comfortable_deceleration=-5.0,
    standstill_distance=6.0,
    time_headway=1.0,
)
DRIVER_PROFILE_DEFINITIONS = (
    SLOW_DRIVER_PROFILE,
    NORMAL_DRIVER_PROFILE,
    FAST_DRIVER_PROFILE,
)
DEFAULT_DRIVER_PROFILE_PROBABILITIES = {
    "slow": 1.0 / 3.0,
    "normal": 1.0 / 3.0,
    "fast": 1.0 / 3.0,
}


def validate_driver_profile_probabilities(
    probabilities: Mapping[str, float],
) -> tuple[float, ...]:
    """Validate and order a categorical slow/normal/fast mixture."""
    if not isinstance(probabilities, Mapping):
        raise ValueError("driver_profile_probabilities must be a mapping")

    expected_names = tuple(profile.name for profile in DRIVER_PROFILE_DEFINITIONS)
    supplied_names = set(probabilities)
    if supplied_names != set(expected_names):
        raise ValueError(
            "driver_profile_probabilities must contain exactly "
            f"{expected_names}"
        )

    try:
        ordered = tuple(float(probabilities[name]) for name in expected_names)
    except (TypeError, ValueError) as error:
        raise ValueError("driver profile probabilities must be numeric") from error

    if any(not isfinite(probability) for probability in ordered):
        raise ValueError("driver profile probabilities must be finite")
    if any(probability < 0 for probability in ordered):
        raise ValueError("driver profile probabilities must be non-negative")
    if not np.isclose(sum(ordered), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("driver profile probabilities must sum to one")
    return ordered


class NonReversingIDMVehicle(IDMVehicle):
    """IDM traffic vehicle that retains collision logic but cannot reverse."""

    MIN_SPEED = 0.0

    def step(self, dt: float) -> None:
        """Advance normally, then prevent braking from producing negative speed."""
        self.speed = max(0.0, float(self.speed))
        super().step(dt)
        if self.speed < 0:
            self.speed = 0.0
            self.action["acceleration"] = max(
                0.0,
                float(self.action["acceleration"]),
            )


class ProfiledNonReversingIDMVehicle(NonReversingIDMVehicle):
    """Non-reversing IDM vehicle configured by a driver profile."""

    PROFILE: ClassVar[DriverProfileDefinition]

    def __init__(
        self,
        road,
        position,
        heading: float = 0,
        speed: float = 0,
        target_lane_index=None,
        target_speed: float | None = None,
        route=None,
        enable_lane_change: bool = True,
        timer: float | None = None,
    ) -> None:
        # Supplying the desired speed here prevents ControlledVehicle from
        # silently using the randomly sampled initial speed as its target.
        if target_speed is None:
            target_speed = self.PROFILE.desired_speed
        super().__init__(
            road=road,
            position=position,
            heading=heading,
            speed=speed,
            target_lane_index=target_lane_index,
            target_speed=target_speed,
            route=route,
            enable_lane_change=enable_lane_change,
            timer=timer,
        )
        # This label is available to the simulator and evaluation code only;
        # observation snapshots intentionally do not expose it.
        self.driver_profile_name = self.PROFILE.name

    def act(self, action: dict | str | None = None) -> None:
        """Restore this profile's cruise speed after road-rule yielding."""
        if not self.crashed and not getattr(self, "is_yielding", False):
            desired_speed = self.PROFILE.desired_speed
            if self.lane is not None and self.lane.speed_limit is not None:
                desired_speed = min(desired_speed, float(self.lane.speed_limit))
            self.target_speed = desired_speed
        super().act(action)


class SlowDriverVehicle(ProfiledNonReversingIDMVehicle):
    """Lower-speed driver with a longer desired following headway."""

    PROFILE = SLOW_DRIVER_PROFILE
    ACC_MAX = PROFILE.maximum_acceleration
    COMFORT_ACC_MAX = PROFILE.comfortable_acceleration
    COMFORT_ACC_MIN = PROFILE.comfortable_deceleration
    DISTANCE_WANTED = PROFILE.standstill_distance
    TIME_WANTED = PROFILE.time_headway
    DELTA_RANGE = list(PROFILE.delta_range)


class NormalDriverVehicle(ProfiledNonReversingIDMVehicle):
    """Middle profile used as the nominal background-driving behavior."""

    PROFILE = NORMAL_DRIVER_PROFILE
    ACC_MAX = PROFILE.maximum_acceleration
    COMFORT_ACC_MAX = PROFILE.comfortable_acceleration
    COMFORT_ACC_MIN = PROFILE.comfortable_deceleration
    DISTANCE_WANTED = PROFILE.standstill_distance
    TIME_WANTED = PROFILE.time_headway
    DELTA_RANGE = list(PROFILE.delta_range)


class FastDriverVehicle(ProfiledNonReversingIDMVehicle):
    """Higher-speed driver with a shorter desired following headway."""

    PROFILE = FAST_DRIVER_PROFILE
    ACC_MAX = PROFILE.maximum_acceleration
    COMFORT_ACC_MAX = PROFILE.comfortable_acceleration
    COMFORT_ACC_MIN = PROFILE.comfortable_deceleration
    DISTANCE_WANTED = PROFILE.standstill_distance
    TIME_WANTED = PROFILE.time_headway
    DELTA_RANGE = list(PROFILE.delta_range)


DRIVER_PROFILE_VEHICLE_TYPES = (
    SlowDriverVehicle,
    NormalDriverVehicle,
    FastDriverVehicle,
)
DRIVER_PROFILE_VEHICLE_TYPE_BY_NAME = {
    vehicle_type.PROFILE.name: vehicle_type
    for vehicle_type in DRIVER_PROFILE_VEHICLE_TYPES
}


class MajorRoadStraightIntersectionEnv(ConnectedLaneIntersectionEnv):
    """Intersection-v2 restricted to straight major-road background traffic."""

    @classmethod
    def default_config(cls) -> dict:
        """Use a seeded heterogeneous mixture of non-reversing IDM traffic."""
        config = super().default_config()
        # Kept as the fallback type for deterministic validators that create
        # individual vehicles manually. Normal episode spawning selects one
        # of the three profiled subclasses instead.
        config["other_vehicles_type"] = (
            "fixed_gap_controller.NonReversingIDMVehicle"
        )
        config["driver_profile_probabilities"] = dict(
            DEFAULT_DRIVER_PROFILE_PROBABILITIES
        )
        return config

    def _sample_driver_vehicle_type(
        self,
    ) -> type[ProfiledNonReversingIDMVehicle]:
        """Sample one hidden profile using the environment's seeded RNG."""
        probabilities = validate_driver_profile_probabilities(
            self.config["driver_profile_probabilities"]
        )
        profile_index = int(
            self.np_random.choice(
                len(DRIVER_PROFILE_VEHICLE_TYPES),
                p=probabilities,
            )
        )
        return DRIVER_PROFILE_VEHICLE_TYPES[profile_index]

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> Vehicle | None:
        """Spawn a vehicle on either horizontal approach with a straight route."""
        del go_straight  # All background vehicles are straight in this experiment.

        if self.np_random.uniform() > spawn_probability:
            return None

        origin = int(self.np_random.choice([1, 3]))
        destination = (origin + 2) % 4
        lane_index = (f"o{origin}", f"ir{origin}", 0)
        lane = self.road.network.get_lane(lane_index)
        vehicle_type = self._sample_driver_vehicle_type()
        initial_speed = (
            vehicle_type.PROFILE.desired_speed
            + self.np_random.normal() * speed_deviation
        )
        initial_speed = max(0.0, float(initial_speed))
        if lane.speed_limit is not None:
            initial_speed = min(initial_speed, float(lane.speed_limit))
        vehicle = vehicle_type.make_on_lane(
            self.road,
            lane_index,
            longitudinal=(
                longitudinal
                + 5.0
                + self.np_random.normal() * position_deviation
            ),
            speed=initial_speed,
        )

        if any(
            np.linalg.norm(other.position - vehicle.position) < 15
            for other in self.road.vehicles
        ):
            return None

        vehicle.plan_route_to(f"o{destination}")
        vehicle.enable_lane_change = False
        vehicle.collidable = True
        vehicle.solid = True
        vehicle.check_collisions = True
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        super()._make_vehicles(n_vehicles)

        # Calibration uses the same road and ego dynamics without traffic.
        if self.config.get("calibration_mode", False):
            self.road.vehicles = list(self.controlled_vehicles)


class ControllerState(str, Enum):
    """High-level state of the fixed-gap controller."""

    APPROACH = "APPROACH"
    WAIT = "WAIT"
    COMMIT = "COMMIT"
    DONE = "DONE"


@dataclass(frozen=True)
class FixedGapConfig:
    """Parameters for the fixed critical-gap controller."""

    crossing_time: float
    safety_margin: float = FROZEN_SAFETY_MARGIN
    speed_epsilon: float = 0.1
    stopped_speed: float = 0.5
    slow_down_distance: float = 20.0
    braking_distance: float = 7.5

    def __post_init__(self) -> None:
        if self.crossing_time <= 0:
            raise ValueError("crossing_time must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if self.braking_distance <= 0:
            raise ValueError("braking_distance must be positive")
        if self.slow_down_distance <= self.braking_distance:
            raise ValueError(
                "slow_down_distance must be larger than braking_distance"
            )

    @property
    def critical_gap(self) -> float:
        """Minimum acceptable time before the next conflicting arrival."""
        return self.crossing_time + self.safety_margin


@dataclass(frozen=True)
class ControllerDecision:
    """One controller decision plus diagnostics useful for debugging."""

    action: int
    action_name: str
    state: ControllerState
    available_gap: float | None
    critical_gap: float
    conflict_zone_occupied: bool


class FixedGapController:
    """Four-state controller using one fixed critical time gap."""

    def __init__(
        self,
        config: FixedGapConfig,
        observation_provider: VehicleObservationProvider | None = None,
    ):
        self.config = config
        self.state = ControllerState.APPROACH
        self.observation_provider = (
            observation_provider
            if observation_provider is not None
            else KalmanObservationProvider()
        )
        self.last_observation: ObservationSnapshot | None = None

    def reset(self) -> None:
        self.state = ControllerState.APPROACH
        self.last_observation = None
        self.observation_provider.reset()

    def decide(
        self, env: MajorRoadStraightIntersectionEnv
    ) -> ControllerDecision:
        """Choose a longitudinal meta-action from the current observation."""
        observation = self.observation_provider.observe(env)
        self.last_observation = observation
        ego = observation.ego

        if observed_ego_has_cleared_conflict_zone(env, ego):
            self.state = ControllerState.DONE

        available_gap: float | None = None
        occupied = False

        if self.state is ControllerState.APPROACH:
            distance = observed_ego_distance_to_intersection(env, ego)

            # A coarse policy frequency can occasionally carry the front
            # bumper to the stop line. Never turn that into an unchecked
            # commitment: apply the same critical-gap test first.
            if distance <= 0:
                available_gap, occupied = self.available_gap(env, observation)
                if not occupied and available_gap >= self.config.critical_gap:
                    self.state = ControllerState.COMMIT
                    action_name = "FASTER"
                else:
                    action_name = self._reduce_target_speed(env, target_index=0)
            elif (
                distance <= self.config.braking_distance
                and ego.speed <= self.config.stopped_speed
            ):
                self.state = ControllerState.WAIT
                available_gap, occupied = self.available_gap(env, observation)
                if not occupied and available_gap >= self.config.critical_gap:
                    self.state = ControllerState.COMMIT
                    action_name = "FASTER"
                else:
                    action_name = "IDLE"
            elif distance <= self.config.braking_distance:
                action_name = self._reduce_target_speed(env, target_index=0)
            elif distance <= self.config.slow_down_distance:
                action_name = self._reduce_target_speed(env, target_index=1)
            else:
                action_name = "IDLE"

        elif self.state is ControllerState.WAIT:
            available_gap, occupied = self.available_gap(env, observation)
            if not occupied and available_gap >= self.config.critical_gap:
                self.state = ControllerState.COMMIT
                action_name = "FASTER"
            else:
                action_name = "IDLE"

        elif self.state is ControllerState.COMMIT:
            action_name = self._increase_to_maximum_speed(env)

        else:
            action_name = "IDLE"

        return ControllerDecision(
            action=self._action_index(env, action_name),
            action_name=action_name,
            state=self.state,
            available_gap=available_gap,
            critical_gap=self.config.critical_gap,
            conflict_zone_occupied=occupied,
        )

    def available_gap(
        self,
        env: MajorRoadStraightIntersectionEnv,
        observation: ObservationSnapshot | None = None,
    ) -> tuple[float, bool]:
        """Return the earliest conflicting TTC and current occupancy flag."""
        if observation is None:
            observation = self.observation_provider.observe(env)
        earliest_ttc = inf
        occupied = False

        for vehicle in observation.other_vehicles:
            ttc, is_occupied = self._time_to_conflict_zone(env, vehicle)
            if is_occupied:
                occupied = True
                earliest_ttc = 0.0
            elif ttc is not None:
                earliest_ttc = min(earliest_ttc, ttc)

        return earliest_ttc, occupied

    def _time_to_conflict_zone(
        self,
        env: MajorRoadStraightIntersectionEnv,
        vehicle: ObservedVehicleState,
    ) -> tuple[float | None, bool]:
        """Estimate TTC from the observed state of a major-road vehicle."""
        if vehicle.lane_index is None:
            return None, False
        edge = vehicle.lane_index[:2]
        lane = env.road.network.get_lane(vehicle.lane_index)

        if edge in MAJOR_INTERSECTION_EDGES:
            return 0.0, True

        if edge in MAJOR_EXIT_EDGES:
            longitudinal, _ = lane.local_coordinates(vehicle.position)
            rear_has_cleared = longitudinal >= vehicle.length / 2
            return (None, False) if rear_has_cleared else (0.0, True)

        if edge not in MAJOR_INCOMING_EDGES:
            return None, False

        # Use the front bumper's distance to the start of the conflict zone.
        longitudinal, _ = lane.local_coordinates(vehicle.position)
        distance = max(
            lane.length - longitudinal - vehicle.length / 2,
            0.0,
        )
        if distance == 0:
            return 0.0, True

        # Project the observed velocity onto the approach lane direction.
        approach_speed = float(np.dot(vehicle.velocity, lane.direction))
        if approach_speed <= self.config.speed_epsilon:
            return None, False

        return distance / max(approach_speed, self.config.speed_epsilon), False

    @staticmethod
    def _action_index(
        env: MajorRoadStraightIntersectionEnv, action_name: str
    ) -> int:
        return int(env.action_type.actions_indexes[action_name])

    @staticmethod
    def _reduce_target_speed(
        env: MajorRoadStraightIntersectionEnv, target_index: int
    ) -> str:
        ego = env.vehicle
        target_speeds = env.action_type.target_speeds
        desired_speed = float(target_speeds[target_index])
        return "SLOWER" if ego.target_speed > desired_speed + 1e-6 else "IDLE"

    @staticmethod
    def _increase_to_maximum_speed(
        env: MajorRoadStraightIntersectionEnv,
    ) -> str:
        maximum_speed = float(env.action_type.target_speeds[-1])
        return "FASTER" if env.vehicle.target_speed < maximum_speed - 1e-6 else "IDLE"


def ego_distance_to_intersection(ego: Vehicle) -> float:
    """Distance from the ego front bumper to the intersection entrance."""
    if ego.lane_index[:2] != EGO_INCOMING_LANE[:2]:
        return -inf
    longitudinal, _ = ego.lane.local_coordinates(ego.position)
    return ego.lane.length - longitudinal - ego.LENGTH / 2


def observed_ego_distance_to_intersection(
    env: MajorRoadStraightIntersectionEnv,
    ego: ObservedVehicleState,
) -> float:
    """Observed ego front-bumper distance to the intersection entrance."""
    if ego.lane_index is None or ego.lane_index[:2] != EGO_INCOMING_LANE[:2]:
        return -inf
    lane = env.road.network.get_lane(ego.lane_index)
    longitudinal, _ = lane.local_coordinates(ego.position)
    return lane.length - longitudinal - ego.length / 2


def ego_has_cleared_conflict_zone(ego: Vehicle) -> bool:
    """Whether the ego rear bumper has left the intersection."""
    if ego.lane_index[:2] != EGO_EXIT_LANE[:2]:
        return False
    longitudinal, _ = ego.lane.local_coordinates(ego.position)
    return longitudinal >= ego.LENGTH / 2


def observed_ego_has_cleared_conflict_zone(
    env: MajorRoadStraightIntersectionEnv,
    ego: ObservedVehicleState,
) -> bool:
    """Whether the observed ego rear bumper has left the intersection."""
    if ego.lane_index is None or ego.lane_index[:2] != EGO_EXIT_LANE[:2]:
        return False
    lane = env.road.network.get_lane(ego.lane_index)
    longitudinal, _ = lane.local_coordinates(ego.position)
    return longitudinal >= ego.length / 2


def place_ego_at_stop_line(
    env: MajorRoadStraightIntersectionEnv, stop_line_clearance: float
) -> None:
    """Place a stationary ego upstream of the conflict-zone entrance."""
    ego = env.vehicle
    lane = env.road.network.get_lane(EGO_INCOMING_LANE)
    longitudinal = lane.length - ego.LENGTH / 2 - stop_line_clearance

    ego.position = lane.position(longitudinal, 0)
    ego.heading = lane.heading_at(longitudinal)
    ego.speed = 0.0
    ego.lane_index = EGO_INCOMING_LANE
    ego.lane = lane
    ego.target_lane_index = EGO_INCOMING_LANE
    ego.plan_route_to("o2")
    ego.speed_index = 0
    ego.target_speed = float(env.action_type.target_speeds[0])


def measure_crossing_time(
    *,
    seed: int = 0,
    stop_line_clearance: float = 4.0,
    policy_frequency: int = 1,
) -> float:
    """Measure and return one traffic-free crossing time for later freezing.

    The timer starts with the ego stationary at the stopping point and ends
    when its rear bumper has completely cleared the intersection.
    """
    env = MajorRoadStraightIntersectionEnv(
        config={
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": [0, 4.5, 9],
            },
            "calibration_mode": True,
            "destination": "o2",
            "duration": 30,
            "initial_vehicle_count": 1,
            "policy_frequency": policy_frequency,
            "spawn_probability": 0.0,
        }
    )

    try:
        env.reset(seed=seed)
        place_ego_at_stop_line(env, stop_line_clearance)
        start_time = env.time
        faster = int(env.action_type.actions_indexes["FASTER"])
        idle = int(env.action_type.actions_indexes["IDLE"])
        maximum_speed = float(env.action_type.target_speeds[-1])
        maximum_steps = int(env.config["duration"] * policy_frequency) + 1

        for _ in range(maximum_steps):
            action = (
                faster
                if env.vehicle.target_speed < maximum_speed - 1e-6
                else idle
            )
            _, _, terminated, truncated, _ = env.step(action)

            if ego_has_cleared_conflict_zone(env.vehicle):
                return env.time - start_time
            if env.vehicle.crashed:
                raise RuntimeError("Unexpected collision during empty calibration")
            if terminated or truncated:
                break

        raise RuntimeError("Ego did not clear the intersection during calibration")
    finally:
        env.close()
