"""Shared trajectory prediction utilities for the predictive controllers.

The initial background-vehicle model uses constant Cartesian velocity:

    predicted_position(t) = observed_position + observed_velocity * t

It consumes immutable observation records and does not access or mutate the
Highway-env simulation.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import atan2, cos, floor, isfinite, sin

from vehicle_observations import (
    CandidateTrajectory,
    ObservationSnapshot,
    ObservedVehicleState,
    PredictedVehicleState,
)


FROZEN_PREDICTION_HORIZON = 6.0
FROZEN_PREDICTION_TIME_STEP = 0.2


@dataclass(frozen=True)
class PredictionConfig:
    """Sampling settings shared by the predictive controllers."""

    horizon: float = FROZEN_PREDICTION_HORIZON
    time_step: float = FROZEN_PREDICTION_TIME_STEP

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("prediction horizon must be positive")
        if self.time_step <= 0:
            raise ValueError("prediction time_step must be positive")

    @property
    def relative_times(self) -> tuple[float, ...]:
        """Return samples from zero through the exact configured horizon."""
        regular_step_count = floor(self.horizon / self.time_step + 1e-12)
        times = [index * self.time_step for index in range(regular_step_count + 1)]

        tolerance = 1e-12
        if self.horizon - times[-1] > tolerance:
            times.append(self.horizon)
        else:
            times[-1] = self.horizon
        return tuple(times)


DEFAULT_PREDICTION_CONFIG = PredictionConfig()


def interpolate_predicted_state(
    trajectory: CandidateTrajectory,
    relative_time: float,
    *,
    tolerance: float = 1e-9,
) -> PredictedVehicleState:
    """Interpolate one trajectory at an in-range relative timestamp."""
    if not isfinite(relative_time):
        raise ValueError("relative_time must be finite")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    samples = trajectory.samples
    first_time = samples[0].relative_time
    last_time = samples[-1].relative_time
    if relative_time < first_time - tolerance or relative_time > last_time + tolerance:
        raise ValueError(
            f"trajectory for vehicle {samples[0].vehicle_id} does not cover "
            f"relative time {relative_time:.6f}"
        )
    if abs(relative_time - first_time) <= tolerance:
        return samples[0]
    if abs(relative_time - last_time) <= tolerance:
        return samples[-1]

    times = [sample.relative_time for sample in samples]
    index = bisect_left(times, relative_time)
    if index < len(samples) and abs(times[index] - relative_time) <= tolerance:
        return samples[index]

    before = samples[index - 1]
    after = samples[index]
    fraction = (relative_time - before.relative_time) / (
        after.relative_time - before.relative_time
    )
    heading_difference = atan2(
        sin(after.heading - before.heading),
        cos(after.heading - before.heading),
    )
    return PredictedVehicleState(
        vehicle_id=before.vehicle_id,
        relative_time=float(relative_time),
        position=tuple(
            before.position[axis]
            + fraction * (after.position[axis] - before.position[axis])
            for axis in range(2)
        ),
        velocity=tuple(
            before.velocity[axis]
            + fraction * (after.velocity[axis] - before.velocity[axis])
            for axis in range(2)
        ),
        heading=before.heading + fraction * heading_difference,
        length=before.length,
        width=before.width,
    )


def predict_constant_velocity(
    vehicle: ObservedVehicleState,
    *,
    start_time: float,
    config: PredictionConfig = DEFAULT_PREDICTION_CONFIG,
) -> CandidateTrajectory:
    """Predict one observed vehicle with constant position velocity."""
    position_x, position_y = vehicle.position
    velocity_x, velocity_y = vehicle.velocity
    samples = tuple(
        PredictedVehicleState(
            vehicle_id=vehicle.vehicle_id,
            relative_time=relative_time,
            position=(
                position_x + velocity_x * relative_time,
                position_y + velocity_y * relative_time,
            ),
            velocity=vehicle.velocity,
            heading=vehicle.heading,
            length=vehicle.length,
            width=vehicle.width,
        )
        for relative_time in config.relative_times
    )
    return CandidateTrajectory(
        plan_name="CONSTANT_VELOCITY",
        start_time=float(start_time),
        samples=samples,
    )


def predict_background_vehicles(
    observation: ObservationSnapshot,
    *,
    config: PredictionConfig = DEFAULT_PREDICTION_CONFIG,
) -> tuple[CandidateTrajectory, ...]:
    """Predict every background observation over the shared sample times."""
    return tuple(
        predict_constant_velocity(
            vehicle,
            start_time=observation.timestamp,
            config=config,
        )
        for vehicle in observation.other_vehicles
    )
