"""Build development traffic, shifted evaluation traffic, and deterministic hazards."""
from __future__ import annotations
from math import isfinite
from typing import Protocol
from highway_env.vehicle.kinematics import Vehicle
from fixed_gap_controller import DriverProfileDefinition, FastDriverVehicle, NormalDriverVehicle, SlowDriverVehicle, ProfiledNonReversingIDMVehicle, MajorRoadStraightIntersectionEnv, place_ego_at_stop_line, validate_driver_profile_probabilities
from experiment_config import EVALUATION_DRIVER_SHIFTS, EVALUATION_SENSOR_SHIFTS, EVALUATION_HAZARD_VARIANTS, EVALUATION_EPISODE_DURATION, EVALUATION_INITIAL_VEHICLE_COUNT, EVALUATION_POLICY_FREQUENCY, EVALUATION_HAZARD_DURATION, EVALUATION_HAZARD_ENVIRONMENT_SEED

STOP_LINE_CLEARANCE = 4.0
POLICY_FREQUENCY = EVALUATION_POLICY_FREQUENCY

class ScenarioTask(Protocol):
    seed: int
    spawn_probability: float | None
    driver_shift: str | None
    hazard_variant: str | None

def make_environment(
    render_mode: str | None,
    *,
    duration: float = 13,
    initial_vehicle_count: int = 10,
    spawn_probability: float = 0.6,
    driver_profile_probabilities: dict[str, float] | None = None,
) -> MajorRoadStraightIntersectionEnv:
    """Construct the straight, major-road-priority experiment."""
    config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": [0, 4.5, 9],
        },
        "destination": "o2",
        "duration": duration,
        "initial_vehicle_count": initial_vehicle_count,
        "policy_frequency": 1,
        "spawn_probability": spawn_probability,
    }
    if driver_profile_probabilities is not None:
        config["driver_profile_probabilities"] = dict(
            driver_profile_probabilities
        )
    return MajorRoadStraightIntersectionEnv(config=config, render_mode=render_mode)

def make_empty_environment(
    *,
    duration: float = 30,
    seed: int = 0,
    render_mode: str | None = None,
) -> MajorRoadStraightIntersectionEnv:
    """Build the experiment road with no background vehicles or new spawns."""
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
            "duration": duration,
            "initial_vehicle_count": 1,
            "policy_frequency": POLICY_FREQUENCY,
            "spawn_probability": 0.0,
        },
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env

def _add_profile_vehicle(
    env: MajorRoadStraightIntersectionEnv,
    *,
    vehicle_type: type[ProfiledNonReversingIDMVehicle],
    origin: int,
    front_distance: float,
    speed: float,
) -> Vehicle:
    """Place a deterministic straight priority vehicle before the zone."""
    if origin not in {1, 3}:
        raise ValueError("origin must be 1 (left/west) or 3 (right/east)")
    if not isfinite(front_distance) or front_distance <= 0:
        raise ValueError("front_distance must be finite and positive")
    if not isfinite(speed) or speed < 0:
        raise ValueError("speed must be finite and non-negative")

    lane_index = (f"o{origin}", f"ir{origin}", 0)
    lane = env.road.network.get_lane(lane_index)
    longitudinal = lane.length - Vehicle.LENGTH / 2 - front_distance
    if longitudinal < 0:
        raise ValueError(
            f"front distance {front_distance:.2f} m exceeds the incoming lane"
        )

    vehicle = vehicle_type.make_on_lane(
        env.road,
        lane_index,
        longitudinal=longitudinal,
        speed=speed,
    )
    destination = (origin + 2) % 4
    vehicle.plan_route_to(f"o{destination}")
    vehicle.target_speed = vehicle_type.PROFILE.desired_speed
    vehicle.enable_lane_change = False
    vehicle.collidable = True
    vehicle.solid = True
    vehicle.check_collisions = True
    env.road.vehicles.append(vehicle)
    return vehicle

def _profile_definition(parameters) -> DriverProfileDefinition:
    return DriverProfileDefinition(
        name=parameters.name,
        desired_speed=parameters.desired_speed,
        maximum_acceleration=parameters.maximum_acceleration,
        comfortable_acceleration=parameters.comfortable_acceleration,
        comfortable_deceleration=parameters.comfortable_deceleration,
        standstill_distance=parameters.standstill_distance,
        time_headway=parameters.time_headway,
    )

def _vehicle_type(name: str, parameters):
    profile = _profile_definition(parameters)
    return type(
        name,
        (ProfiledNonReversingIDMVehicle,),
        {
            "PROFILE": profile,
            "ACC_MAX": profile.maximum_acceleration,
            "COMFORT_ACC_MAX": profile.comfortable_acceleration,
            "COMFORT_ACC_MIN": profile.comfortable_deceleration,
            "DISTANCE_WANTED": profile.standstill_distance,
            "TIME_WANTED": profile.time_headway,
            "DELTA_RANGE": list(profile.delta_range),
        },
    )

EVALUATION_VEHICLE_TYPES = {
    shift.name: tuple(
        _vehicle_type(
            f"HeldOut_{shift.name}_{parameters.name}",
            parameters,
        )
        for parameters in shift.profiles
    )
    for shift in EVALUATION_DRIVER_SHIFTS
}

DRIVER_SHIFT_BY_NAME = {shift.name: shift for shift in EVALUATION_DRIVER_SHIFTS}

SENSOR_SHIFT_BY_NAME = {shift.name: shift for shift in EVALUATION_SENSOR_SHIFTS}

HAZARD_BY_NAME = {variant.name: variant for variant in EVALUATION_HAZARD_VARIANTS}

class ShiftedTrafficEnvironment(MajorRoadStraightIntersectionEnv):
    """Straight-traffic environment using one frozen shifted driver set."""

    def __init__(self, *, driver_shift_name: str, config: dict) -> None:
        self.driver_shift_name = driver_shift_name
        super().__init__(config=config, render_mode=None)

    def _sample_driver_vehicle_type(self):
        probabilities = validate_driver_profile_probabilities(
            self.config["driver_profile_probabilities"]
        )
        profile_index = int(
            self.np_random.choice(3, p=probabilities)
        )
        return EVALUATION_VEHICLE_TYPES[self.driver_shift_name][profile_index]

def _make_stochastic_environment(task: ScenarioTask):
    if task.driver_shift is None or task.spawn_probability is None:
        raise ValueError("stochastic task is missing its shifted conditions")
    shift = DRIVER_SHIFT_BY_NAME[task.driver_shift]
    probabilities = {
        profile.name: probability
        for profile, probability in zip(shift.profiles, shift.probabilities, strict=True)
    }
    config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": [0, 4.5, 9],
        },
        "destination": "o2",
        "duration": EVALUATION_EPISODE_DURATION,
        "initial_vehicle_count": EVALUATION_INITIAL_VEHICLE_COUNT,
        "policy_frequency": EVALUATION_POLICY_FREQUENCY,
        "spawn_probability": task.spawn_probability,
        "driver_profile_probabilities": probabilities,
    }
    env = ShiftedTrafficEnvironment(
        driver_shift_name=task.driver_shift,
        config=config,
    )
    env.reset(seed=task.seed)
    return env, ()

def _hazard_parameters(task: ScenarioTask) -> dict[str, float | int | str]:
    if task.hazard_variant is None:
        raise ValueError("hazard task has no variant")
    return dict(HAZARD_BY_NAME[task.hazard_variant].truth_parameters)

def _make_hazard_environment(task: ScenarioTask):
    variant = HAZARD_BY_NAME[str(task.hazard_variant)]
    values = _hazard_parameters(task)
    env = make_empty_environment(
        duration=EVALUATION_HAZARD_DURATION,
        seed=EVALUATION_HAZARD_ENVIRONMENT_SEED,
    )
    place_ego_at_stop_line(env, STOP_LINE_CLEARANCE)

    if variant.name in {"left_boundary_ttc_shift", "right_boundary_speed_shift"}:
        vehicles = (
            _add_profile_vehicle(
                env,
                vehicle_type=FastDriverVehicle,
                origin=int(values["origin"]),
                front_distance=float(values["front_distance"]),
                speed=float(values["speed"]),
            ),
        )
    elif variant.name in {
        "shifted_slow_leader_fast_follower",
        "shifted_leader_induced_braking",
    }:
        vehicles = (
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
    elif variant.name == "shifted_two_direction_arrival":
        vehicles = (
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
    elif variant.name == "mirrored_biased_latency_boundary":
        vehicles = (
            _add_profile_vehicle(
                env,
                vehicle_type=NormalDriverVehicle,
                origin=int(values["origin"]),
                front_distance=float(values["front_distance"]),
                speed=float(values["speed"]),
            ),
        )
    else:
        raise ValueError(f"unknown held-out hazard: {variant.name}")
    return env, vehicles
