"""Shared lane-aligned Kalman state estimation for background vehicles.

The estimator consumes immutable timestamped measurements from
``vehicle_observations``. For each stable vehicle identifier it estimates

    [longitudinal position, speed, position bias, speed bias]

at the measurement timestamp, then propagates a copy to the controller's
current delivery time. Controller 1 and Controller 2 consume paired filtered
means. Controller 3 also uses the covariance to adapt its required clearance.
The covariance does not alter the first two controllers' decision margins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, isfinite, sin
from typing import TypeAlias

import numpy as np

from vehicle_observations import (
    CorrelatedGaussianNoiseConfig,
    DelayedCorrelatedGaussianObservationProvider,
    ExactObservationProvider,
    FROZEN_CORRELATED_NOISE_CONFIG,
    FROZEN_OBSERVATION_LATENCY_CONFIG,
    ObservationLatencyConfig,
    ObservationSnapshot,
    ObservedVehicleState,
    VehicleObservationProvider,
)


StateMean: TypeAlias = tuple[float, float, float, float]
FROZEN_KALMAN_ACCELERATION_PROCESS_STD = 1.5
StateCovariance: TypeAlias = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class KalmanFilterConfig:
    """Process and measurement model for the shared Kalman estimator."""

    noise: CorrelatedGaussianNoiseConfig = field(
        default_factory=CorrelatedGaussianNoiseConfig
    )
    acceleration_process_std: float = FROZEN_KALMAN_ACCELERATION_PROCESS_STD
    exact_passthrough_when_clean: bool = True

    def __post_init__(self) -> None:
        if (
            not isfinite(self.acceleration_process_std)
            or self.acceleration_process_std < 0
        ):
            raise ValueError(
                "acceleration_process_std must be finite and non-negative"
            )
        if not isinstance(self.exact_passthrough_when_clean, bool):
            raise ValueError("exact_passthrough_when_clean must be boolean")


@dataclass(frozen=True)
class KalmanStateEstimate:
    """One numeric state estimate and covariance at a requested timestamp."""

    timestamp: float
    mean: StateMean
    covariance: StateCovariance

    @property
    def longitudinal_position(self) -> float:
        return self.mean[0]

    @property
    def speed(self) -> float:
        return self.mean[1]

    @property
    def position_bias(self) -> float:
        return self.mean[2]

    @property
    def speed_bias(self) -> float:
        return self.mean[3]

    @property
    def longitudinal_position_variance(self) -> float:
        return self.covariance[0][0]

    @property
    def speed_variance(self) -> float:
        return self.covariance[1][1]


@dataclass(frozen=True)
class KalmanVehicleEstimate:
    """Filtered vehicle mean plus uncertainty and source timing."""

    vehicle: ObservedVehicleState
    state: KalmanStateEstimate
    measurement_timestamp: float
    estimate_timestamp: float

    def __post_init__(self) -> None:
        if self.vehicle.vehicle_id < 0:
            raise ValueError("vehicle_id must be non-negative")
        if self.estimate_timestamp < self.measurement_timestamp - 1e-9:
            raise ValueError("estimate cannot precede its measurement")
        if abs(self.state.timestamp - self.estimate_timestamp) > 1e-9:
            raise ValueError("state and estimate timestamps must match")


def _covariance_tuple(covariance: np.ndarray) -> StateCovariance:
    if covariance.shape != (4, 4):
        raise ValueError("state covariance must be 4 x 4")
    return tuple(
        tuple(float(value) for value in row)
        for row in covariance
    )  # type: ignore[return-value]


class LaneAlignedKalmanFilter:
    """Kalman filter for one vehicle's motion and persistent sensor biases."""

    _MEASUREMENT_MATRIX = np.asarray(
        (
            (1.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 1.0),
        ),
        dtype=float,
    )

    def __init__(self, config: KalmanFilterConfig | None = None) -> None:
        self.config = config or KalmanFilterConfig()
        self._mean: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._measurement_timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return self._mean is not None

    @property
    def measurement_timestamp(self) -> float | None:
        return self._measurement_timestamp

    def update(
        self,
        *,
        longitudinal_position: float,
        speed: float,
        timestamp: float,
    ) -> KalmanStateEstimate:
        """Assimilate one measurement at its actual measurement time."""
        for name, value in (
            ("longitudinal_position", longitudinal_position),
            ("speed", speed),
            ("timestamp", timestamp),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        measurement = np.asarray((longitudinal_position, speed), dtype=float)
        if self._mean is None:
            self._initialize(measurement, timestamp)
            return self.estimate(timestamp)

        if self._measurement_timestamp is None or self._covariance is None:
            raise RuntimeError("initialized filter state is incomplete")
        if timestamp < self._measurement_timestamp - 1e-9:
            raise ValueError("measurements must not move backwards in time")
        if abs(timestamp - self._measurement_timestamp) <= 1e-9:
            # Startup clamping can deliver the same cached measurement more
            # than once. It must not be counted as independent evidence.
            return self.estimate(timestamp)

        elapsed = timestamp - self._measurement_timestamp
        predicted_mean, predicted_covariance = self._predict_arrays(
            self._mean,
            self._covariance,
            elapsed,
        )
        measurement_matrix = self._MEASUREMENT_MATRIX
        measurement_covariance = np.diag(
            (
                self.config.noise.position_white_std**2,
                self.config.noise.speed_white_std**2,
            )
        )
        innovation = measurement - measurement_matrix @ predicted_mean
        innovation_covariance = (
            measurement_matrix
            @ predicted_covariance
            @ measurement_matrix.T
            + measurement_covariance
        )
        gain = (
            predicted_covariance
            @ measurement_matrix.T
            @ np.linalg.pinv(innovation_covariance)
        )
        updated_mean = predicted_mean + gain @ innovation

        identity = np.eye(4)
        residual_transform = identity - gain @ measurement_matrix
        updated_covariance = (
            residual_transform
            @ predicted_covariance
            @ residual_transform.T
            + gain @ measurement_covariance @ gain.T
        )

        self._mean = updated_mean
        self._covariance = self._stabilize_covariance(updated_covariance)
        self._measurement_timestamp = float(timestamp)
        return self.estimate(timestamp)

    def estimate(self, timestamp: float) -> KalmanStateEstimate:
        """Return a non-mutating prediction at or after the last measurement."""
        if not isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if (
            self._mean is None
            or self._covariance is None
            or self._measurement_timestamp is None
        ):
            raise RuntimeError("filter has not received a measurement")
        if timestamp < self._measurement_timestamp - 1e-9:
            raise ValueError("estimate cannot precede the latest measurement")
        elapsed = max(0.0, timestamp - self._measurement_timestamp)
        mean, covariance = self._predict_arrays(
            self._mean,
            self._covariance,
            elapsed,
        )
        return KalmanStateEstimate(
            timestamp=float(timestamp),
            mean=tuple(float(value) for value in mean),  # type: ignore[arg-type]
            covariance=_covariance_tuple(covariance),
        )

    def _initialize(self, measurement: np.ndarray, timestamp: float) -> None:
        position_bias_variance = (
            self.config.noise.position_stationary_bias_std**2
        )
        speed_bias_variance = self.config.noise.speed_stationary_bias_std**2
        position_white_variance = self.config.noise.position_white_std**2
        speed_white_variance = self.config.noise.speed_white_std**2

        self._mean = np.asarray(
            (measurement[0], measurement[1], 0.0, 0.0),
            dtype=float,
        )
        self._covariance = np.zeros((4, 4), dtype=float)
        self._covariance[0, 0] = (
            position_bias_variance + position_white_variance
        )
        self._covariance[1, 1] = speed_bias_variance + speed_white_variance
        self._covariance[2, 2] = position_bias_variance
        self._covariance[3, 3] = speed_bias_variance
        self._covariance[0, 2] = self._covariance[2, 0] = (
            -position_bias_variance
        )
        self._covariance[1, 3] = self._covariance[3, 1] = (
            -speed_bias_variance
        )
        self._covariance = self._stabilize_covariance(self._covariance)
        self._measurement_timestamp = float(timestamp)

    def _predict_arrays(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        elapsed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if elapsed <= 1e-12:
            return mean.copy(), covariance.copy()

        correlation = self.config.noise.correlation
        bias_decay = correlation**elapsed if correlation > 0 else 0.0
        transition = np.asarray(
            (
                (1.0, elapsed, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, bias_decay, 0.0),
                (0.0, 0.0, 0.0, bias_decay),
            ),
            dtype=float,
        )

        acceleration_variance = self.config.acceleration_process_std**2
        process_covariance = np.zeros((4, 4), dtype=float)
        process_covariance[0, 0] = 0.25 * elapsed**4 * acceleration_variance
        process_covariance[0, 1] = process_covariance[1, 0] = (
            0.5 * elapsed**3 * acceleration_variance
        )
        process_covariance[1, 1] = elapsed**2 * acceleration_variance

        position_bias_variance = (
            self.config.noise.position_stationary_bias_std**2
        )
        speed_bias_variance = self.config.noise.speed_stationary_bias_std**2
        process_covariance[2, 2] = position_bias_variance * (
            1 - bias_decay**2
        )
        process_covariance[3, 3] = speed_bias_variance * (
            1 - bias_decay**2
        )

        predicted_mean = transition @ mean
        predicted_covariance = (
            transition @ covariance @ transition.T + process_covariance
        )
        return predicted_mean, self._stabilize_covariance(predicted_covariance)

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if float(eigenvalues.min()) < -1e-7:
            raise ValueError("Kalman covariance became materially indefinite")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        stabilized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return 0.5 * (stabilized + stabilized.T)


@dataclass(frozen=True)
class _VehicleGeometry:
    heading: float
    lateral_offset: float
    length: float
    width: float
    lane_index: tuple[str, str, int] | None


class KalmanObservationProvider:
    """Convert delayed noisy measurements into current filtered means."""

    def __init__(
        self,
        measurement_provider: VehicleObservationProvider | None = None,
        config: KalmanFilterConfig | None = None,
    ) -> None:
        self.measurement_provider = (
            measurement_provider
            if measurement_provider is not None
            else ExactObservationProvider()
        )
        self.config = config or KalmanFilterConfig()
        self._filters: dict[int, LaneAlignedKalmanFilter] = {}
        self._geometry: dict[int, _VehicleGeometry] = {}
        self.last_measurement: ObservationSnapshot | None = None
        self.last_estimates: tuple[KalmanVehicleEstimate, ...] = ()

    def reset(self) -> None:
        self.measurement_provider.reset()
        self._filters.clear()
        self._geometry.clear()
        self.last_measurement = None
        self.last_estimates = ()

    def observe(self, env: object) -> ObservationSnapshot:
        measurement = self.measurement_provider.observe(env)
        self.last_measurement = measurement
        if self._can_use_exact_passthrough(measurement):
            return self._exact_passthrough(env, measurement)

        delivery_timestamp = float(measurement.delivery_timestamp)
        estimates = tuple(
            self._filter_vehicle(
                env,
                vehicle,
                measurement_timestamp=measurement.timestamp,
                estimate_timestamp=delivery_timestamp,
            )
            for vehicle in measurement.other_vehicles
        )
        self.last_estimates = estimates
        return ObservationSnapshot(
            timestamp=delivery_timestamp,
            delivery_timestamp=delivery_timestamp,
            ego=measurement.ego,
            other_vehicles=tuple(estimate.vehicle for estimate in estimates),
        )

    def estimate_for(self, vehicle_id: int) -> KalmanVehicleEstimate | None:
        return next(
            (
                estimate
                for estimate in self.last_estimates
                if estimate.vehicle.vehicle_id == vehicle_id
            ),
            None,
        )

    @property
    def tracked_vehicle_ids(self) -> tuple[int, ...]:
        """Return stable IDs whose filter histories are retained this episode."""
        return tuple(sorted(self._filters))

    def _can_use_exact_passthrough(
        self,
        measurement: ObservationSnapshot,
    ) -> bool:
        if not self.config.exact_passthrough_when_clean:
            return False
        source_noise_config = getattr(self.measurement_provider, "config", None)
        source_noise_enabled = bool(
            getattr(source_noise_config, "enabled", False)
        )
        source_latency_config = getattr(
            self.measurement_provider,
            "latency_config",
            None,
        )
        source_latency_enabled = bool(
            getattr(source_latency_config, "enabled", False)
        )
        return (
            not source_noise_enabled
            and not source_latency_enabled
            and measurement.latency <= 1e-9
        )

    def _exact_passthrough(
        self,
        env: object,
        measurement: ObservationSnapshot,
    ) -> ObservationSnapshot:
        zero_covariance = _covariance_tuple(np.zeros((4, 4), dtype=float))
        estimates = []
        for vehicle in measurement.other_vehicles:
            progress, speed, geometry = self._measurement_components(env, vehicle)
            state = KalmanStateEstimate(
                timestamp=measurement.timestamp,
                mean=(progress, speed, 0.0, 0.0),
                covariance=zero_covariance,
            )
            estimates.append(
                KalmanVehicleEstimate(
                    vehicle=vehicle,
                    state=state,
                    measurement_timestamp=measurement.timestamp,
                    estimate_timestamp=measurement.timestamp,
                )
            )
            self._geometry[vehicle.vehicle_id] = geometry
        self.last_estimates = tuple(estimates)
        return measurement

    def _filter_vehicle(
        self,
        env: object,
        vehicle: ObservedVehicleState,
        *,
        measurement_timestamp: float,
        estimate_timestamp: float,
    ) -> KalmanVehicleEstimate:
        progress, speed, geometry = self._measurement_components(env, vehicle)
        filter_ = self._filters.setdefault(
            vehicle.vehicle_id,
            LaneAlignedKalmanFilter(self.config),
        )
        filter_.update(
            longitudinal_position=progress,
            speed=speed,
            timestamp=measurement_timestamp,
        )
        state = filter_.estimate(estimate_timestamp)
        self._geometry[vehicle.vehicle_id] = geometry
        filtered_vehicle = self._vehicle_from_state(
            env,
            vehicle.vehicle_id,
            state,
            geometry,
        )
        return KalmanVehicleEstimate(
            vehicle=filtered_vehicle,
            state=state,
            measurement_timestamp=measurement_timestamp,
            estimate_timestamp=estimate_timestamp,
        )

    @staticmethod
    def _measurement_components(
        env: object,
        vehicle: ObservedVehicleState,
    ) -> tuple[float, float, _VehicleGeometry]:
        direction = np.asarray(
            (cos(vehicle.heading), sin(vehicle.heading)),
            dtype=float,
        )
        normal = np.asarray((-direction[1], direction[0]), dtype=float)
        position = np.asarray(vehicle.position, dtype=float)
        lane_index = vehicle.lane_index
        if lane_index is not None:
            lane = env.road.network.get_lane(lane_index)
            lane_longitudinal, _ = lane.local_coordinates(position)
            centre_position = np.asarray(
                lane.position(lane_longitudinal, 0.0),
                dtype=float,
            )
        else:
            centre_position = position
        progress = float(np.dot(centre_position, direction))
        speed = float(np.dot(np.asarray(vehicle.velocity), direction))
        lateral_offset = float(np.dot(centre_position, normal))
        geometry = _VehicleGeometry(
            heading=vehicle.heading,
            lateral_offset=lateral_offset,
            length=vehicle.length,
            width=vehicle.width,
            lane_index=lane_index,
        )
        return progress, speed, geometry

    @staticmethod
    def _vehicle_from_state(
        env: object,
        vehicle_id: int,
        state: KalmanStateEstimate,
        geometry: _VehicleGeometry,
    ) -> ObservedVehicleState:
        direction = np.asarray(
            (cos(geometry.heading), sin(geometry.heading)),
            dtype=float,
        )
        normal = np.asarray((-direction[1], direction[0]), dtype=float)
        position = (
            state.longitudinal_position * direction
            + geometry.lateral_offset * normal
        )
        velocity = state.speed * direction
        closest_lane_index = env.road.network.get_closest_lane_index(
            position,
            geometry.heading,
        )
        lane_index = (
            None
            if closest_lane_index is None
            else (
                str(closest_lane_index[0]),
                str(closest_lane_index[1]),
                int(closest_lane_index[2]),
            )
        )
        return ObservedVehicleState(
            vehicle_id=vehicle_id,
            position=(float(position[0]), float(position[1])),
            velocity=(float(velocity[0]), float(velocity[1])),
            heading=geometry.heading,
            length=geometry.length,
            width=geometry.width,
            lane_index=lane_index,
            is_ego=False,
        )


def make_shared_kalman_observation_provider(
    *,
    noise: CorrelatedGaussianNoiseConfig | None = None,
    latency: ObservationLatencyConfig | None = None,
    acceleration_process_std: float = FROZEN_KALMAN_ACCELERATION_PROCESS_STD,
    seed: int = 0,
) -> KalmanObservationProvider:
    """Build the common measurement-and-estimation pipeline for controllers.

    A new provider must be created for every controller episode. Supplying the
    same seed and configuration gives paired controllers identical measurement
    and estimation inputs without sharing mutable filter state.
    """
    noise_config = noise or CorrelatedGaussianNoiseConfig()
    latency_config = latency or ObservationLatencyConfig()
    return KalmanObservationProvider(
        DelayedCorrelatedGaussianObservationProvider(
            noise_config,
            latency_config,
            seed=seed,
        ),
        KalmanFilterConfig(
            noise=noise_config,
            acceleration_process_std=acceleration_process_std,
        ),
    )


def make_frozen_realism_observation_provider(
    *,
    seed: int = 0,
) -> KalmanObservationProvider:
    """Build the frozen nominal noisy, delayed, filtered controller input."""
    return make_shared_kalman_observation_provider(
        noise=FROZEN_CORRELATED_NOISE_CONFIG,
        latency=FROZEN_OBSERVATION_LATENCY_CONFIG,
        acceleration_process_std=FROZEN_KALMAN_ACCELERATION_PROCESS_STD,
        seed=seed,
    )
