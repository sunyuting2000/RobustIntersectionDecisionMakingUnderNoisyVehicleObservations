"""Shared immutable observation and prediction data structures.

Controllers consume snapshots from this module instead of reading background
vehicle kinematics directly. Exact, independent-noise, and correlated-noise
providers share the same interface and never change the ground-truth
simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin
from typing import Protocol, TypeAlias

import numpy as np

from highway_env.vehicle.kinematics import Vehicle


Vector2: TypeAlias = tuple[float, float]
LaneIndex: TypeAlias = tuple[str, str, int]


def _vector2(value: object, *, field_name: str) -> Vector2:
    """Copy a two-dimensional numeric value into an immutable tuple."""
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{field_name} must contain exactly two values")
    return float(array[0]), float(array[1])


@dataclass(frozen=True)
class ObservedVehicleState:
    """One vehicle state as available to a controller at one instant."""

    vehicle_id: int
    position: Vector2
    velocity: Vector2
    heading: float
    length: float
    width: float
    lane_index: LaneIndex | None
    is_ego: bool = False

    @property
    def speed(self) -> float:
        """Magnitude of the observed velocity vector in metres per second."""
        return float(np.hypot(*self.velocity))


@dataclass(frozen=True)
class ObservationSnapshot:
    """All vehicle observations supplied for one controller decision.

    ``timestamp`` is when the background-vehicle measurement was made.
    ``delivery_timestamp`` is when it became available to the controller. They
    are equal for exact and zero-latency observations. The ego state remains
    exact at delivery time because this project delays background V2V/sensor
    measurements only.
    """

    timestamp: float
    ego: ObservedVehicleState
    other_vehicles: tuple[ObservedVehicleState, ...]
    delivery_timestamp: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        delivery_timestamp = (
            self.timestamp
            if self.delivery_timestamp is None
            else self.delivery_timestamp
        )
        if not isfinite(delivery_timestamp):
            raise ValueError("delivery_timestamp must be finite")
        if delivery_timestamp < self.timestamp - 1e-9:
            raise ValueError("delivery_timestamp cannot precede timestamp")
        object.__setattr__(self, "delivery_timestamp", float(delivery_timestamp))

    @property
    def latency(self) -> float:
        """Return the effective measurement-to-delivery delay in seconds."""
        if self.delivery_timestamp is None:  # Normalized in __post_init__.
            raise RuntimeError("delivery timestamp was not initialized")
        return max(0.0, self.delivery_timestamp - self.timestamp)

    @property
    def vehicles(self) -> tuple[ObservedVehicleState, ...]:
        """Return the ego followed by all observed background vehicles."""
        return (self.ego, *self.other_vehicles)


@dataclass(frozen=True)
class PredictedVehicleState:
    """One immutable future sample for a vehicle trajectory."""

    vehicle_id: int
    relative_time: float
    position: Vector2
    velocity: Vector2
    heading: float
    length: float
    width: float

    def __post_init__(self) -> None:
        if self.relative_time < 0:
            raise ValueError("relative_time must be non-negative")


@dataclass(frozen=True)
class CandidateTrajectory:
    """Ordered future samples for one candidate vehicle plan."""

    plan_name: str
    start_time: float
    samples: tuple[PredictedVehicleState, ...]

    def __post_init__(self) -> None:
        if not self.plan_name:
            raise ValueError("plan_name must not be empty")
        if not self.samples:
            raise ValueError("a candidate trajectory requires at least one sample")
        sample_times = [sample.relative_time for sample in self.samples]
        if sample_times != sorted(sample_times):
            raise ValueError("trajectory samples must be ordered by relative_time")


class VehicleObservationProvider(Protocol):
    """Interface shared by exact and noisy observation providers."""

    def reset(self) -> None:
        """Reset episode-specific provider state."""

    def observe(self, env: object) -> ObservationSnapshot:
        """Return one observation snapshot without mutating the environment."""


class ExactObservationProvider:
    """Copy exact current simulator values into immutable observations."""

    def __init__(self) -> None:
        self._vehicle_ids: dict[int, int] = {}
        self._next_vehicle_id = 0

    def reset(self) -> None:
        self._vehicle_ids.clear()
        self._next_vehicle_id = 0

    def _identifier_for(self, vehicle: Vehicle) -> int:
        identity = id(vehicle)
        if identity not in self._vehicle_ids:
            self._vehicle_ids[identity] = self._next_vehicle_id
            self._next_vehicle_id += 1
        return self._vehicle_ids[identity]

    def _observe_vehicle(
        self, vehicle: Vehicle, *, is_ego: bool
    ) -> ObservedVehicleState:
        lane_index = (
            None
            if vehicle.lane_index is None
            else (
                str(vehicle.lane_index[0]),
                str(vehicle.lane_index[1]),
                int(vehicle.lane_index[2]),
            )
        )
        return ObservedVehicleState(
            vehicle_id=self._identifier_for(vehicle),
            position=_vector2(vehicle.position, field_name="position"),
            velocity=_vector2(vehicle.velocity, field_name="velocity"),
            heading=float(vehicle.heading),
            length=float(vehicle.LENGTH),
            width=float(vehicle.WIDTH),
            lane_index=lane_index,
            is_ego=is_ego,
        )

    def observe(self, env: object) -> ObservationSnapshot:
        """Copy the current ego and road vehicles without changing either."""
        ego_vehicle = env.vehicle
        ego = self._observe_vehicle(ego_vehicle, is_ego=True)
        other_vehicles = tuple(
            self._observe_vehicle(vehicle, is_ego=False)
            for vehicle in env.road.vehicles
            if vehicle is not ego_vehicle
        )
        return ObservationSnapshot(
            timestamp=float(env.time),
            ego=ego,
            other_vehicles=other_vehicles,
        )


@dataclass(frozen=True)
class GaussianNoiseConfig:
    """Known standard deviations for background-vehicle measurements.

    Position noise is sampled independently for the global x and y
    coordinates. Speed noise is one scalar sample applied along the vehicle's
    observed heading, preserving its known straight route. The ego vehicle,
    headings, dimensions, lane identities, timestamp, and road geometry remain
    exact.
    """

    position_std: float = 0.0
    speed_std: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.position_std) or self.position_std < 0:
            raise ValueError("position_std must be finite and non-negative")
        if not isfinite(self.speed_std) or self.speed_std < 0:
            raise ValueError("speed_std must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return self.position_std > 0 or self.speed_std > 0


class GaussianObservationProvider(ExactObservationProvider):
    """Return exact ego state and noisy background position/speed states.

    A fresh independent sample is drawn for every observed background vehicle
    at every call to :meth:`observe`. Calling :meth:`reset` restores both the
    stable vehicle-ID mapping and the original random stream, which makes an
    episode reproducible and permits paired controller experiments.
    """

    def __init__(
        self,
        config: GaussianNoiseConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        super().__init__()
        self.config = config or GaussianNoiseConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)

    def observe(self, env: object) -> ObservationSnapshot:
        exact = super().observe(env)
        if not self.config.enabled:
            return exact
        return ObservationSnapshot(
            timestamp=exact.timestamp,
            ego=exact.ego,
            other_vehicles=tuple(
                self._add_noise(vehicle) for vehicle in exact.other_vehicles
            ),
            delivery_timestamp=exact.delivery_timestamp,
        )

    def _add_noise(self, vehicle: ObservedVehicleState) -> ObservedVehicleState:
        position = vehicle.position
        if self.config.position_std > 0:
            position_error = self._rng.normal(
                loc=0.0,
                scale=self.config.position_std,
                size=2,
            )
            position = (
                vehicle.position[0] + float(position_error[0]),
                vehicle.position[1] + float(position_error[1]),
            )

        velocity = vehicle.velocity
        if self.config.speed_std > 0:
            speed_error = float(
                self._rng.normal(loc=0.0, scale=self.config.speed_std)
            )
            velocity = (
                vehicle.velocity[0] + speed_error * cos(vehicle.heading),
                vehicle.velocity[1] + speed_error * sin(vehicle.heading),
            )

        return ObservedVehicleState(
            vehicle_id=vehicle.vehicle_id,
            position=position,
            velocity=velocity,
            heading=vehicle.heading,
            length=vehicle.length,
            width=vehicle.width,
            lane_index=vehicle.lane_index,
            is_ego=False,
        )


@dataclass(frozen=True)
class CorrelatedGaussianNoiseConfig:
    """Parameters for persistent-bias plus white Gaussian measurement noise.

    For every background vehicle and measured component, the persistent bias
    follows the stationary first-order process

        bias[k] = correlation * bias[k - 1] + innovation[k].

    The configured innovation standard deviations describe the new random
    bias increment at each controller observation. Independent white noise is
    then added to the persistent bias. Position parameters apply separately
    to global x and y; speed parameters apply along the known vehicle heading.

    The initial bias is sampled from the corresponding stationary
    distribution. Consequently, the stationary bias standard deviation is
    ``innovation_std / sqrt(1 - correlation**2)``. The dataclass defaults
    remain clean/exact for deterministic validation; the frozen realistic
    experiment configuration is declared below.
    """

    correlation: float = 0.0
    position_bias_innovation_std: float = 0.0
    position_white_std: float = 0.0
    speed_bias_innovation_std: float = 0.0
    speed_white_std: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.correlation) or not 0 <= self.correlation < 1:
            raise ValueError("correlation must be finite and in [0, 1)")
        for field_name in (
            "position_bias_innovation_std",
            "position_white_std",
            "speed_bias_innovation_std",
            "speed_white_std",
        ):
            value = getattr(self, field_name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return any(
            value > 0
            for value in (
                self.position_bias_innovation_std,
                self.position_white_std,
                self.speed_bias_innovation_std,
                self.speed_white_std,
            )
        )

    @property
    def position_stationary_bias_std(self) -> float:
        return self.position_bias_innovation_std / float(
            np.sqrt(1 - self.correlation**2)
        )

    @property
    def speed_stationary_bias_std(self) -> float:
        return self.speed_bias_innovation_std / float(
            np.sqrt(1 - self.correlation**2)
        )

    @property
    def position_measurement_std(self) -> float:
        return float(
            np.hypot(
                self.position_stationary_bias_std,
                self.position_white_std,
            )
        )

    @property
    def speed_measurement_std(self) -> float:
        return float(
            np.hypot(
                self.speed_stationary_bias_std,
                self.speed_white_std,
            )
        )


@dataclass(frozen=True)
class _CorrelatedBiasState:
    """Persistent measurement bias for one background vehicle."""

    position: Vector2
    speed: float


class CorrelatedGaussianObservationProvider(ExactObservationProvider):
    """Add seeded, vehicle-specific temporally correlated measurement noise.

    Bias state is maintained independently for every stable background
    vehicle identifier. Each call to :meth:`observe` is one measurement step,
    which currently corresponds to the controllers' frozen 1 Hz policy rate.
    The ego and all non-kinematic metadata remain exact. Calling :meth:`reset`
    restores the original random stream, vehicle identifiers, and bias state.
    """

    def __init__(
        self,
        config: CorrelatedGaussianNoiseConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        super().__init__()
        self.config = config or CorrelatedGaussianNoiseConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._bias_by_vehicle_id: dict[int, _CorrelatedBiasState] = {}

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self._bias_by_vehicle_id.clear()

    def observe(self, env: object) -> ObservationSnapshot:
        exact = super().observe(env)
        if not self.config.enabled:
            return exact
        return ObservationSnapshot(
            timestamp=exact.timestamp,
            ego=exact.ego,
            other_vehicles=tuple(
                self._add_correlated_noise(vehicle)
                for vehicle in exact.other_vehicles
            ),
            delivery_timestamp=exact.delivery_timestamp,
        )

    def _initial_bias(self) -> _CorrelatedBiasState:
        position_std = self.config.position_stationary_bias_std
        speed_std = self.config.speed_stationary_bias_std
        position = (
            tuple(float(value) for value in self._rng.normal(0.0, position_std, 2))
            if position_std > 0
            else (0.0, 0.0)
        )
        speed = (
            float(self._rng.normal(0.0, speed_std)) if speed_std > 0 else 0.0
        )
        return _CorrelatedBiasState(position=position, speed=speed)

    def _advance_bias(
        self, previous: _CorrelatedBiasState
    ) -> _CorrelatedBiasState:
        position_innovation = (
            self._rng.normal(
                0.0,
                self.config.position_bias_innovation_std,
                2,
            )
            if self.config.position_bias_innovation_std > 0
            else np.zeros(2)
        )
        speed_innovation = (
            float(
                self._rng.normal(
                    0.0,
                    self.config.speed_bias_innovation_std,
                )
            )
            if self.config.speed_bias_innovation_std > 0
            else 0.0
        )
        return _CorrelatedBiasState(
            position=(
                self.config.correlation * previous.position[0]
                + float(position_innovation[0]),
                self.config.correlation * previous.position[1]
                + float(position_innovation[1]),
            ),
            speed=self.config.correlation * previous.speed + speed_innovation,
        )

    def _next_bias(self, vehicle_id: int) -> _CorrelatedBiasState:
        previous = self._bias_by_vehicle_id.get(vehicle_id)
        bias = (
            self._initial_bias()
            if previous is None
            else self._advance_bias(previous)
        )
        self._bias_by_vehicle_id[vehicle_id] = bias
        return bias

    def _add_correlated_noise(
        self, vehicle: ObservedVehicleState
    ) -> ObservedVehicleState:
        bias = self._next_bias(vehicle.vehicle_id)
        position_white = (
            self._rng.normal(0.0, self.config.position_white_std, 2)
            if self.config.position_white_std > 0
            else np.zeros(2)
        )
        speed_white = (
            float(self._rng.normal(0.0, self.config.speed_white_std))
            if self.config.speed_white_std > 0
            else 0.0
        )
        speed_error = bias.speed + speed_white
        return ObservedVehicleState(
            vehicle_id=vehicle.vehicle_id,
            position=(
                vehicle.position[0] + bias.position[0] + float(position_white[0]),
                vehicle.position[1] + bias.position[1] + float(position_white[1]),
            ),
            velocity=(
                vehicle.velocity[0] + speed_error * cos(vehicle.heading),
                vehicle.velocity[1] + speed_error * sin(vehicle.heading),
            ),
            heading=vehicle.heading,
            length=vehicle.length,
            width=vehicle.width,
            lane_index=vehicle.lane_index,
            is_ego=False,
        )


@dataclass(frozen=True)
class ObservationLatencyConfig:
    """Fixed delay between a background measurement and its delivery."""

    latency: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.latency) or self.latency < 0:
            raise ValueError("latency must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return self.latency > 0


# Shared nominal sensing configuration. The dataclass defaults above provide
# a noise-free, zero-latency pipeline when requested explicitly.
SHARED_REALISM_CONFIGURATION_FROZEN = True
FROZEN_CORRELATED_NOISE_CONFIG = CorrelatedGaussianNoiseConfig(
    correlation=0.90,
    position_bias_innovation_std=0.436,
    position_white_std=0.30,
    speed_bias_innovation_std=0.174,
    speed_white_std=0.15,
)
FROZEN_OBSERVATION_LATENCY_CONFIG = ObservationLatencyConfig(latency=1.0)


class DelayedCorrelatedGaussianObservationProvider(
    CorrelatedGaussianObservationProvider
):
    """Deliver timestamped delayed background truth with optional noise.

    Highway-env's physics-rate vehicle history supplies states between the
    1 Hz controller decisions. Histories are copied into provider-owned
    immutable observations and linearly interpolated at the requested
    measurement timestamp. Ground truth is used only to generate sensor/V2V
    measurements; it is never supplied as a current correction to a
    controller.

    At episode startup, unavailable pre-episode history is clamped to the
    episode's initial timestamp. Thus the configured delay is reached once
    enough simulated time has elapsed. A clamped timestamp reuses the same
    measurement rather than drawing a second noise sample for the same time.
    Vehicle appearances and disappearances are evaluated at measurement time,
    not delivery time.
    """

    def __init__(
        self,
        noise_config: CorrelatedGaussianNoiseConfig | None = None,
        latency_config: ObservationLatencyConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__(noise_config, seed=seed)
        self.latency_config = latency_config or ObservationLatencyConfig()
        self._episode_start_time: float | None = None
        self._truth_history: dict[
            int,
            dict[float, ObservedVehicleState],
        ] = {}
        self._truth_vehicle_by_id: dict[int, Vehicle] = {}
        self._seen_history_sample_ids: dict[int, set[int]] = {}
        self._latest_truth_time: dict[int, float] = {}
        self._measurement_cache: dict[
            float,
            tuple[ObservedVehicleState, ...],
        ] = {}

    def reset(self) -> None:
        super().reset()
        self._episode_start_time = None
        self._truth_history.clear()
        self._truth_vehicle_by_id.clear()
        self._seen_history_sample_ids.clear()
        self._latest_truth_time.clear()
        self._measurement_cache.clear()

    def observe(self, env: object) -> ObservationSnapshot:
        # This flag records immutable physics snapshots only. It does not
        # change vehicle dynamics, actions, collisions, or controller inputs.
        env.road.record_history = True
        exact = ExactObservationProvider.observe(self, env)
        delivery_time = exact.timestamp
        if self._episode_start_time is None:
            self._episode_start_time = delivery_time
        self._harvest_truth_history(env, exact)

        if self._episode_start_time is None:
            raise RuntimeError("episode start timestamp was not initialized")
        measurement_time = max(
            self._episode_start_time,
            delivery_time - self.latency_config.latency,
        )
        measurement_time = float(measurement_time)

        if abs(measurement_time - delivery_time) <= 1e-9:
            measured_vehicles = exact.other_vehicles
        else:
            measured_vehicles = tuple(
                state
                for vehicle_id in sorted(self._truth_history)
                if (
                    state := self._state_at_time(vehicle_id, measurement_time)
                )
                is not None
            )

        measurement_key = self._history_key(measurement_time)
        cached_measurement = self._measurement_cache.get(measurement_key)
        if cached_measurement is not None:
            measured_vehicles = cached_measurement
        else:
            if self.config.enabled:
                measured_vehicles = tuple(
                    self._add_correlated_noise(vehicle)
                    for vehicle in measured_vehicles
                )
            self._measurement_cache[measurement_key] = measured_vehicles

        return ObservationSnapshot(
            timestamp=measurement_time,
            delivery_timestamp=delivery_time,
            ego=exact.ego,
            other_vehicles=measured_vehicles,
        )

    @staticmethod
    def _history_key(timestamp: float) -> float:
        return round(float(timestamp), 12)

    @staticmethod
    def _copy_vehicle_with_id(
        vehicle: Vehicle,
        vehicle_id: int,
    ) -> ObservedVehicleState:
        lane_index = (
            None
            if vehicle.lane_index is None
            else (
                str(vehicle.lane_index[0]),
                str(vehicle.lane_index[1]),
                int(vehicle.lane_index[2]),
            )
        )
        return ObservedVehicleState(
            vehicle_id=vehicle_id,
            position=_vector2(vehicle.position, field_name="position"),
            velocity=_vector2(vehicle.velocity, field_name="velocity"),
            heading=float(vehicle.heading),
            length=float(vehicle.LENGTH),
            width=float(vehicle.WIDTH),
            lane_index=lane_index,
            is_ego=False,
        )

    def _harvest_truth_history(
        self,
        env: object,
        exact: ObservationSnapshot,
    ) -> None:
        background_vehicles = tuple(
            vehicle for vehicle in env.road.vehicles if vehicle is not env.vehicle
        )
        if len(background_vehicles) != len(exact.other_vehicles):
            raise RuntimeError(
                "truth vehicle count changed while creating an observation"
            )
        simulation_frequency = float(env.config["simulation_frequency"])
        if not isfinite(simulation_frequency) or simulation_frequency <= 0:
            raise ValueError("simulation_frequency must be finite and positive")
        physics_time_step = 1.0 / simulation_frequency
        delivery_time = exact.timestamp

        for observed, vehicle in zip(
            exact.other_vehicles,
            background_vehicles,
            strict=True,
        ):
            self._truth_vehicle_by_id[observed.vehicle_id] = vehicle
            history = self._truth_history.setdefault(observed.vehicle_id, {})
            seen_sample_ids = self._seen_history_sample_ids.setdefault(
                observed.vehicle_id,
                set(),
            )
            history[self._history_key(delivery_time)] = observed
            self._latest_truth_time[observed.vehicle_id] = delivery_time
            for sample_index, historical_vehicle in enumerate(vehicle.history):
                sample_time = delivery_time - sample_index * physics_time_step
                if (
                    self._episode_start_time is not None
                    and sample_time < self._episode_start_time - 1e-9
                ):
                    continue
                history[self._history_key(sample_time)] = (
                    self._copy_vehicle_with_id(
                        historical_vehicle,
                        observed.vehicle_id,
                    )
                )
                seen_sample_ids.add(id(historical_vehicle))

        current_vehicle_ids = {
            observed.vehicle_id for observed in exact.other_vehicles
        }
        for vehicle_id, vehicle in self._truth_vehicle_by_id.items():
            if vehicle_id in current_vehicle_ids:
                continue
            history = self._truth_history[vehicle_id]
            seen_sample_ids = self._seen_history_sample_ids[vehicle_id]
            latest_time = self._latest_truth_time[vehicle_id]
            unseen_samples = [
                historical_vehicle
                for historical_vehicle in reversed(vehicle.history)
                if id(historical_vehicle) not in seen_sample_ids
            ]
            for historical_vehicle in unseen_samples:
                latest_time += physics_time_step
                history[self._history_key(latest_time)] = (
                    self._copy_vehicle_with_id(historical_vehicle, vehicle_id)
                )
                seen_sample_ids.add(id(historical_vehicle))
            self._latest_truth_time[vehicle_id] = latest_time

    def _state_at_time(
        self,
        vehicle_id: int,
        timestamp: float,
    ) -> ObservedVehicleState | None:
        samples = self._truth_history[vehicle_id]
        sample_times = sorted(samples)
        if (
            not sample_times
            or timestamp < sample_times[0] - 1e-9
            or timestamp > sample_times[-1] + 1e-9
        ):
            return None

        target_key = self._history_key(timestamp)
        if target_key in samples:
            return samples[target_key]

        later_index = next(
            index
            for index, sample_time in enumerate(sample_times)
            if sample_time > timestamp
        )
        earlier_time = sample_times[later_index - 1]
        later_time = sample_times[later_index]
        earlier = samples[earlier_time]
        later = samples[later_time]
        fraction = (timestamp - earlier_time) / (later_time - earlier_time)
        return self._interpolate_vehicle(earlier, later, fraction)

    @staticmethod
    def _interpolate_vehicle(
        earlier: ObservedVehicleState,
        later: ObservedVehicleState,
        fraction: float,
    ) -> ObservedVehicleState:
        if earlier.vehicle_id != later.vehicle_id:
            raise ValueError("cannot interpolate different vehicles")
        fraction = min(max(float(fraction), 0.0), 1.0)
        heading_difference = atan2(
            sin(later.heading - earlier.heading),
            cos(later.heading - earlier.heading),
        )

        def blend(first: float, second: float) -> float:
            return first + fraction * (second - first)

        return ObservedVehicleState(
            vehicle_id=earlier.vehicle_id,
            position=(
                blend(earlier.position[0], later.position[0]),
                blend(earlier.position[1], later.position[1]),
            ),
            velocity=(
                blend(earlier.velocity[0], later.velocity[0]),
                blend(earlier.velocity[1], later.velocity[1]),
            ),
            heading=earlier.heading + fraction * heading_difference,
            length=blend(earlier.length, later.length),
            width=blend(earlier.width, later.width),
            lane_index=(
                earlier.lane_index if fraction < 0.5 else later.lane_index
            ),
            is_ego=False,
        )
