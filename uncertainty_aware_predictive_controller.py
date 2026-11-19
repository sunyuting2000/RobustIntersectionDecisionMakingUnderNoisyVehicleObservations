"""Predictive controller with covariance-dependent spatial clearance.

Controller 3 inherits Controller 2's observation, candidate-plan prediction,
footprint geometry, conflict-zone gate, state machine, and action selection.
Only the clearance hook changes. For each predicted background vehicle it
uses the largest propagated position uncertainty in a short window around
that vehicle's predicted closest approach to the ego plan::

    b_i = b_0 + k max_{|t-t_closest| <= 0.5 s} sigma_position,i(t)

The longitudinal covariance is propagated with the same constant-velocity
transition and acceleration-process variance used by the shared Kalman
estimator. The default parameters are ``b_0=1.5 m`` and ``k=0``. The sensing
parameter study searches profile-specific ``(b_0, k)`` values with the
conflict-window definition above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite, sqrt
from typing import Sequence

from fixed_clearance_collision_checker import (
    FixedClearanceResult,
    check_fixed_clearance,
)
from predictive_fixed_margin_controller import (
    PredictiveControllerDecision,
    PredictiveFixedMarginConfig,
    PredictiveFixedMarginController,
    _PlanEvaluation,
)
from state_estimation import KalmanVehicleEstimate, StateCovariance
from trajectory_prediction import DEFAULT_PREDICTION_CONFIG, PredictionConfig
from vehicle_observations import (
    CandidateTrajectory,
    ObservationSnapshot,
    VehicleObservationProvider,
)


# Initial candidate for the nominal development search.
INITIAL_DEVELOPMENT_BASE_CLEARANCE = 1.5
INITIAL_DEVELOPMENT_UNCERTAINTY_MULTIPLIER = 1.0

# Under the nominal medium_q1p5 development pipeline,
# propagated sigma was nearly constant and nonzero k behaved like an added
# fixed margin. The selected defaults reduce to Controller 2's b=1.5 m rule.
# The sensing parameter study supplies profile-specific parameters.
FROZEN_BASE_CLEARANCE = 1.5
FROZEN_UNCERTAINTY_MULTIPLIER = 0.0
UNCERTAINTY_AWARE_CONFIGURATION_FROZEN = True
DEFAULT_CONFLICT_WINDOW_HALF_WIDTH = 0.5


@dataclass(frozen=True)
class UncertaintyAwarePredictiveConfig:
    """Controller 3 parameters for the adaptive-clearance rule."""

    base_clearance: float = FROZEN_BASE_CLEARANCE
    uncertainty_multiplier: float = (
        FROZEN_UNCERTAINTY_MULTIPLIER
    )
    prediction: PredictionConfig = field(
        default_factory=lambda: DEFAULT_PREDICTION_CONFIG
    )
    conflict_window_half_width: float = DEFAULT_CONFLICT_WINDOW_HALF_WIDTH
    stopped_speed: float = 0.5
    slow_down_distance: float = 20.0
    braking_distance: float = 7.5

    def __post_init__(self) -> None:
        if not isfinite(self.base_clearance) or self.base_clearance < 0:
            raise ValueError(
                "base_clearance must be finite and non-negative"
            )
        if (
            not isfinite(self.uncertainty_multiplier)
            or self.uncertainty_multiplier < 0
        ):
            raise ValueError(
                "uncertainty_multiplier must be finite and non-negative"
            )
        if (
            not isfinite(self.conflict_window_half_width)
            or self.conflict_window_half_width < 0
        ):
            raise ValueError(
                "conflict_window_half_width must be finite and non-negative"
            )
        # Reuse Controller 2's validation for all shared state-machine fields.
        self.as_fixed_margin_config()

    def as_fixed_margin_config(self) -> PredictiveFixedMarginConfig:
        """Return the behaviorally shared Controller 2 configuration."""
        return PredictiveFixedMarginConfig(
            clearance=self.base_clearance,
            prediction=self.prediction,
            stopped_speed=self.stopped_speed,
            slow_down_distance=self.slow_down_distance,
            braking_distance=self.braking_distance,
        )


@dataclass(frozen=True)
class VehicleAdaptiveClearance:
    """Conflict-window uncertainty assigned to one background trajectory."""

    vehicle_id: int
    initial_position_std: float
    maximum_position_std: float
    time_of_maximum_uncertainty: float
    required_clearance: float
    closest_approach_time: float | None = None
    uncertainty_window_start: float | None = None
    uncertainty_window_end: float | None = None


@dataclass(frozen=True)
class UncertaintyAwareControllerDecision(PredictiveControllerDecision):
    """Controller 2 diagnostics plus the adaptive-clearance calculation."""

    base_clearance: float
    uncertainty_multiplier: float
    maximum_required_clearance: float | None
    minimum_clearance_slack: float | None
    adaptive_clearances: tuple[VehicleAdaptiveClearance, ...]


def projected_longitudinal_position_variance(
    covariance: StateCovariance,
    elapsed: float,
    *,
    acceleration_process_std: float,
) -> float:
    """Propagate longitudinal position variance over ``elapsed`` seconds.

    This is the position component of the estimator transition
    ``P(t) = F P F^T + Q`` for physical position and speed. Persistent sensor-
    bias states are already represented in ``P`` and do not drive physical
    motion during future trajectory prediction.
    """
    if not isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed must be finite and non-negative")
    if (
        not isfinite(acceleration_process_std)
        or acceleration_process_std < 0
    ):
        raise ValueError(
            "acceleration_process_std must be finite and non-negative"
        )
    if len(covariance) != 4 or any(len(row) != 4 for row in covariance):
        raise ValueError("state covariance must be 4 x 4")
    if not all(isfinite(value) for row in covariance for value in row):
        raise ValueError("state covariance must contain finite values")

    cross_covariance = 0.5 * (covariance[0][1] + covariance[1][0])
    variance = (
        covariance[0][0]
        + 2.0 * elapsed * cross_covariance
        + elapsed**2 * covariance[1][1]
        + 0.25
        * elapsed**4
        * acceleration_process_std**2
    )
    if variance < -1e-7:
        raise ValueError(
            "projected longitudinal position variance became negative"
        )
    return max(0.0, float(variance))


def projected_longitudinal_position_std(
    covariance: StateCovariance,
    elapsed: float,
    *,
    acceleration_process_std: float,
) -> float:
    """Return the propagated longitudinal position standard deviation."""
    return sqrt(
        projected_longitudinal_position_variance(
            covariance,
            elapsed,
            acceleration_process_std=acceleration_process_std,
        )
    )


class UncertaintyAwarePredictiveController(PredictiveFixedMarginController):
    """Controller 3: Controller 2 with one covariance-aware clearance hook."""

    def __init__(
        self,
        config: UncertaintyAwarePredictiveConfig | None = None,
        observation_provider: VehicleObservationProvider | None = None,
    ) -> None:
        self.adaptive_config = config or UncertaintyAwarePredictiveConfig()
        super().__init__(
            config=self.adaptive_config.as_fixed_margin_config(),
            observation_provider=observation_provider,
        )
        self.last_clearance_profile: tuple[
            VehicleAdaptiveClearance, ...
        ] = ()
        self.last_minimum_clearance_slack: float | None = None

    def reset(self) -> None:
        super().reset()
        self.last_clearance_profile = ()
        self.last_minimum_clearance_slack = None

    def decide(self, env: object) -> UncertaintyAwareControllerDecision:
        self.last_clearance_profile = ()
        self.last_minimum_clearance_slack = None
        return super().decide(env)  # type: ignore[return-value]

    def _check_clearance(
        self,
        ego_trajectory: CandidateTrajectory,
        background_trajectories: tuple[CandidateTrajectory, ...],
        observation: ObservationSnapshot,
    ) -> FixedClearanceResult:
        del observation
        base_clearance = self.adaptive_config.base_clearance

        if not background_trajectories:
            self.last_minimum_clearance_slack = inf
            return check_fixed_clearance(
                ego_trajectory,
                (),
                clearance=base_clearance,
            )

        estimates = tuple(
            self._estimate_for(trajectory.samples[0].vehicle_id)
            for trajectory in background_trajectories
        )
        closest_approaches = tuple(
            check_fixed_clearance(
                ego_trajectory,
                (trajectory,),
                clearance=0.0,
            )
            for trajectory in background_trajectories
        )
        acceleration_std = self._estimator_acceleration_process_std()
        profiles = tuple(
            self._clearance_profile(
                trajectory,
                estimate,
                closest_approach_time=closest.time_of_minimum_clearance,
                acceleration_process_std=acceleration_std,
            )
            for trajectory, estimate, closest in zip(
                background_trajectories,
                estimates,
                closest_approaches,
                strict=True,
            )
        )
        self.last_clearance_profile = profiles

        # This branch makes the k=0 fairness invariant exact, including the
        # original loop's tie-breaking diagnostics.
        if self.adaptive_config.uncertainty_multiplier == 0:
            result = check_fixed_clearance(
                ego_trajectory,
                background_trajectories,
                clearance=base_clearance,
            )
            self.last_minimum_clearance_slack = (
                result.minimum_clearance - base_clearance
            )
            return result

        results = tuple(
            check_fixed_clearance(
                ego_trajectory,
                (trajectory,),
                clearance=profile.required_clearance,
            )
            for trajectory, profile in zip(
                background_trajectories,
                profiles,
                strict=True,
            )
        )
        return self._combine_clearance_results(results, profiles)

    def _clearance_profile(
        self,
        trajectory: CandidateTrajectory,
        estimate: KalmanVehicleEstimate,
        *,
        closest_approach_time: float | None = None,
        acceleration_process_std: float,
    ) -> VehicleAdaptiveClearance:
        if trajectory.samples[0].vehicle_id != estimate.vehicle.vehicle_id:
            raise ValueError("trajectory and covariance vehicle IDs differ")

        timed_standard_deviations: list[tuple[float, float]] = []
        for sample in trajectory.samples:
            absolute_time = trajectory.start_time + sample.relative_time
            elapsed = absolute_time - estimate.estimate_timestamp
            if elapsed < -1e-9:
                raise ValueError(
                    "prediction trajectory begins before its state estimate"
                )
            timed_standard_deviations.append(
                (
                    sample.relative_time,
                    projected_longitudinal_position_std(
                        estimate.state.covariance,
                        max(0.0, elapsed),
                        acceleration_process_std=acceleration_process_std,
                    ),
                )
            )

        # Diagnostic calls without a closest-approach time use the full horizon.
        # Controller decisions supply the closest-approach time from
        # _check_clearance and therefore use the conflict window.
        if closest_approach_time is None:
            window_start = timed_standard_deviations[0][0]
            window_end = timed_standard_deviations[-1][0]
        else:
            half_width = self.adaptive_config.conflict_window_half_width
            window_start = max(
                timed_standard_deviations[0][0],
                closest_approach_time - half_width,
            )
            window_end = min(
                timed_standard_deviations[-1][0],
                closest_approach_time + half_width,
            )
        window_values = [
            item
            for item in timed_standard_deviations
            if window_start - 1e-9 <= item[0] <= window_end + 1e-9
        ]
        if not window_values and closest_approach_time is not None:
            window_values = [
                min(
                    timed_standard_deviations,
                    key=lambda item: abs(item[0] - closest_approach_time),
                )
            ]
        if not window_values:
            raise RuntimeError("uncertainty window contains no samples")

        maximum_time, maximum_std = max(
            window_values,
            key=lambda item: item[1],
        )
        initial_std = timed_standard_deviations[0][1]
        required_clearance = (
            self.adaptive_config.base_clearance
            + self.adaptive_config.uncertainty_multiplier * maximum_std
        )
        return VehicleAdaptiveClearance(
            vehicle_id=estimate.vehicle.vehicle_id,
            initial_position_std=initial_std,
            maximum_position_std=maximum_std,
            time_of_maximum_uncertainty=maximum_time,
            required_clearance=required_clearance,
            closest_approach_time=closest_approach_time,
            uncertainty_window_start=window_start,
            uncertainty_window_end=window_end,
        )

    def _estimate_for(self, vehicle_id: int) -> KalmanVehicleEstimate:
        estimate_getter = getattr(self.observation_provider, "estimate_for", None)
        if not callable(estimate_getter):
            raise TypeError(
                "Controller 3 requires a covariance-capable shared Kalman "
                "observation provider"
            )
        estimate = estimate_getter(vehicle_id)
        if estimate is None:
            raise RuntimeError(
                f"no covariance estimate is available for vehicle {vehicle_id}"
            )
        return estimate

    def _estimator_acceleration_process_std(self) -> float:
        provider_config = getattr(self.observation_provider, "config", None)
        value = getattr(provider_config, "acceleration_process_std", None)
        if value is None or not isfinite(value) or value < 0:
            raise TypeError(
                "Controller 3 requires the shared estimator's finite "
                "non-negative acceleration_process_std"
            )
        return float(value)

    def _combine_clearance_results(
        self,
        results: Sequence[FixedClearanceResult],
        profiles: Sequence[VehicleAdaptiveClearance],
    ) -> FixedClearanceResult:
        minimum_index = min(
            range(len(results)),
            key=lambda index: results[index].minimum_clearance,
        )
        minimum_result = results[minimum_index]
        slacks = tuple(
            result.minimum_clearance - profile.required_clearance
            for result, profile in zip(results, profiles, strict=True)
        )
        self.last_minimum_clearance_slack = min(slacks)
        return FixedClearanceResult(
            safe=all(result.safe for result in results),
            required_clearance=max(
                profile.required_clearance for profile in profiles
            ),
            minimum_clearance=minimum_result.minimum_clearance,
            time_of_minimum_clearance=(
                minimum_result.time_of_minimum_clearance
            ),
            vehicle_id_of_minimum_clearance=(
                minimum_result.vehicle_id_of_minimum_clearance
            ),
            physical_overlap_predicted=any(
                result.physical_overlap_predicted for result in results
            ),
        )

    def _decision(
        self,
        *,
        env: object,
        action_name: str,
        evaluation: _PlanEvaluation | None,
        observed_background_vehicles: int,
    ) -> UncertaintyAwareControllerDecision:
        base_decision = super()._decision(
            env=env,
            action_name=action_name,
            evaluation=evaluation,
            observed_background_vehicles=observed_background_vehicles,
        )
        maximum_required_clearance = (
            None
            if evaluation is None
            else evaluation.clearance.required_clearance
        )
        return UncertaintyAwareControllerDecision(
            action=base_decision.action,
            action_name=base_decision.action_name,
            state=base_decision.state,
            plan_evaluated=base_decision.plan_evaluated,
            plan_complete=base_decision.plan_complete,
            plan_safe=base_decision.plan_safe,
            minimum_predicted_clearance=(
                base_decision.minimum_predicted_clearance
            ),
            time_of_minimum_clearance=(
                base_decision.time_of_minimum_clearance
            ),
            vehicle_id_of_minimum_clearance=(
                base_decision.vehicle_id_of_minimum_clearance
            ),
            physical_overlap_predicted=(
                base_decision.physical_overlap_predicted
            ),
            conflict_zone_occupied=base_decision.conflict_zone_occupied,
            observed_background_vehicles=(
                base_decision.observed_background_vehicles
            ),
            base_clearance=self.adaptive_config.base_clearance,
            uncertainty_multiplier=(
                self.adaptive_config.uncertainty_multiplier
            ),
            maximum_required_clearance=maximum_required_clearance,
            minimum_clearance_slack=(
                None
                if evaluation is None
                else self.last_minimum_clearance_slack
            ),
            adaptive_clearances=self.last_clearance_profile,
        )
