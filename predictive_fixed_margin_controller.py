"""Predictive intersection controller with one fixed spatial clearance.

At every APPROACH or WAIT decision, the controller predicts the complete ego
CROSS plan and constant-velocity background trajectories. It commits when the
ego is predicted to clear the conflict zone and every predicted physical
footprint maintains the configured edge-to-edge clearance ``b``.

The controller observes traffic throughout the approach and may commit before
coming to a stop. Once in COMMIT, it continues through the intersection
without replanning the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite

from ego_trajectory_prediction import predict_ego_crossing
from fixed_clearance_collision_checker import (
    FixedClearanceResult,
    check_fixed_clearance,
)
from fixed_gap_controller import (
    ControllerState,
    MAJOR_EXIT_EDGES,
    MAJOR_INCOMING_EDGES,
    MAJOR_INTERSECTION_EDGES,
    MajorRoadStraightIntersectionEnv,
    observed_ego_distance_to_intersection,
    observed_ego_has_cleared_conflict_zone,
)
from trajectory_prediction import (
    DEFAULT_PREDICTION_CONFIG,
    PredictionConfig,
    predict_background_vehicles,
)
from state_estimation import KalmanObservationProvider
from vehicle_observations import (
    CandidateTrajectory,
    ObservationSnapshot,
    VehicleObservationProvider,
)


# Alternative clearance for reproducing exact-observation development runs.
# The default for noisy-observation experiments is defined separately below.
HISTORICAL_EXACT_FIXED_CLEARANCE = 0.6

# Selected from a coarse grid over development seeds 0-49 under the
# medium_q1p5 sensing pipeline, without local refinement or evaluation seeds.
FROZEN_REALISTIC_FIXED_CLEARANCE = 1.5
DEFAULT_FIXED_CLEARANCE = FROZEN_REALISTIC_FIXED_CLEARANCE


@dataclass(frozen=True)
class PredictiveFixedMarginConfig:
    """Parameters for predictive fixed-clearance crossing decisions."""

    clearance: float = DEFAULT_FIXED_CLEARANCE
    prediction: PredictionConfig = field(
        default_factory=lambda: DEFAULT_PREDICTION_CONFIG
    )
    stopped_speed: float = 0.5
    slow_down_distance: float = 20.0
    braking_distance: float = 7.5

    def __post_init__(self) -> None:
        if not isfinite(self.clearance) or self.clearance < 0:
            raise ValueError("clearance must be finite and non-negative")
        if not isfinite(self.stopped_speed) or self.stopped_speed < 0:
            raise ValueError("stopped_speed must be finite and non-negative")
        if not isfinite(self.braking_distance) or self.braking_distance <= 0:
            raise ValueError("braking_distance must be finite and positive")
        if (
            not isfinite(self.slow_down_distance)
            or self.slow_down_distance <= self.braking_distance
        ):
            raise ValueError(
                "slow_down_distance must be finite and larger than braking_distance"
            )


@dataclass(frozen=True)
class PredictiveControllerDecision:
    """One action and its predictive safety diagnostics."""

    action: int
    action_name: str
    state: ControllerState
    plan_evaluated: bool
    plan_complete: bool | None
    plan_safe: bool | None
    minimum_predicted_clearance: float | None
    time_of_minimum_clearance: float | None
    vehicle_id_of_minimum_clearance: int | None
    physical_overlap_predicted: bool | None
    conflict_zone_occupied: bool | None
    observed_background_vehicles: int


@dataclass(frozen=True)
class _PlanEvaluation:
    """Internal result of evaluating the candidate CROSS plan."""

    complete: bool
    safe: bool
    clearance: FixedClearanceResult
    conflict_zone_occupied: bool


class PredictiveFixedMarginController:
    """Four-state controller using full-trajectory collision prediction."""

    def __init__(
        self,
        config: PredictiveFixedMarginConfig | None = None,
        observation_provider: VehicleObservationProvider | None = None,
    ) -> None:
        self.config = config or PredictiveFixedMarginConfig()
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
        self,
        env: MajorRoadStraightIntersectionEnv,
    ) -> PredictiveControllerDecision:
        """Observe, evaluate when appropriate, and choose a meta-action."""
        observation = self.observation_provider.observe(env)
        self.last_observation = observation
        ego = observation.ego

        if observed_ego_has_cleared_conflict_zone(env, ego):
            self.state = ControllerState.DONE

        evaluation: _PlanEvaluation | None = None
        action_name = "IDLE"

        if self.state in {ControllerState.APPROACH, ControllerState.WAIT}:
            evaluation = self._evaluate_cross_plan(env, observation)

        if self.state is ControllerState.APPROACH:
            if evaluation is not None and evaluation.safe:
                self.state = ControllerState.COMMIT
                action_name = "FASTER"
            else:
                distance = observed_ego_distance_to_intersection(env, ego)
                if distance <= 0:
                    action_name = self._reduce_target_speed(env, target_index=0)
                elif (
                    distance <= self.config.braking_distance
                    and ego.speed <= self.config.stopped_speed
                ):
                    self.state = ControllerState.WAIT
                    action_name = "IDLE"
                elif distance <= self.config.braking_distance:
                    action_name = self._reduce_target_speed(env, target_index=0)
                elif distance <= self.config.slow_down_distance:
                    action_name = self._reduce_target_speed(env, target_index=1)
                else:
                    action_name = "IDLE"

        elif self.state is ControllerState.WAIT:
            if evaluation is not None and evaluation.safe:
                self.state = ControllerState.COMMIT
                action_name = "FASTER"
            else:
                action_name = "IDLE"

        elif self.state is ControllerState.COMMIT:
            action_name = self._increase_to_maximum_speed(env)

        return self._decision(
            env=env,
            action_name=action_name,
            evaluation=evaluation,
            observed_background_vehicles=len(observation.other_vehicles),
        )

    def _evaluate_cross_plan(
        self,
        env: MajorRoadStraightIntersectionEnv,
        observation: ObservationSnapshot,
    ) -> _PlanEvaluation:
        ego_prediction = predict_ego_crossing(
            env,
            observation,
            config=self.config.prediction,
        )
        background_predictions = predict_background_vehicles(
            observation,
            config=self.config.prediction,
        )
        clearance_result = self._check_clearance(
            ego_prediction.trajectory,
            background_predictions,
            observation,
        )
        complete = ego_prediction.cleared_conflict_zone
        conflict_zone_occupied = self._conflict_zone_occupied(env, observation)
        return _PlanEvaluation(
            complete=complete,
            safe=(
                complete
                and clearance_result.safe
                and not conflict_zone_occupied
            ),
            clearance=clearance_result,
            conflict_zone_occupied=conflict_zone_occupied,
        )

    def _check_clearance(
        self,
        ego_trajectory: CandidateTrajectory,
        background_trajectories: tuple[CandidateTrajectory, ...],
        observation: ObservationSnapshot,
    ) -> FixedClearanceResult:
        """Apply this controller's clearance rule to one candidate plan.

        Controller 2 retains its frozen fixed-clearance rule here. Controller
        3 overrides only this hook, so both predictive controllers share the
        observation, prediction, footprint, conflict-zone, and state-machine
        pipeline exactly.
        """
        del observation
        return check_fixed_clearance(
            ego_trajectory,
            background_trajectories,
            clearance=self.config.clearance,
        )

    @staticmethod
    def _conflict_zone_occupied(
        env: MajorRoadStraightIntersectionEnv,
        observation: ObservationSnapshot,
    ) -> bool:
        """Return whether an observed priority-road vehicle occupies the zone."""
        for vehicle in observation.other_vehicles:
            if vehicle.lane_index is None:
                continue
            edge = vehicle.lane_index[:2]
            if edge in MAJOR_INTERSECTION_EDGES:
                return True

            if edge in MAJOR_EXIT_EDGES:
                lane = env.road.network.get_lane(vehicle.lane_index)
                longitudinal, _ = lane.local_coordinates(vehicle.position)
                if longitudinal < vehicle.length / 2:
                    return True

            if edge in MAJOR_INCOMING_EDGES:
                lane = env.road.network.get_lane(vehicle.lane_index)
                longitudinal, _ = lane.local_coordinates(vehicle.position)
                front_distance = lane.length - longitudinal - vehicle.length / 2
                if front_distance <= 0:
                    return True
        return False

    def _decision(
        self,
        *,
        env: MajorRoadStraightIntersectionEnv,
        action_name: str,
        evaluation: _PlanEvaluation | None,
        observed_background_vehicles: int,
    ) -> PredictiveControllerDecision:
        return PredictiveControllerDecision(
            action=self._action_index(env, action_name),
            action_name=action_name,
            state=self.state,
            plan_evaluated=evaluation is not None,
            plan_complete=None if evaluation is None else evaluation.complete,
            plan_safe=None if evaluation is None else evaluation.safe,
            minimum_predicted_clearance=(
                None
                if evaluation is None
                else evaluation.clearance.minimum_clearance
            ),
            time_of_minimum_clearance=(
                None
                if evaluation is None
                else evaluation.clearance.time_of_minimum_clearance
            ),
            vehicle_id_of_minimum_clearance=(
                None
                if evaluation is None
                else evaluation.clearance.vehicle_id_of_minimum_clearance
            ),
            physical_overlap_predicted=(
                None
                if evaluation is None
                else evaluation.clearance.physical_overlap_predicted
            ),
            conflict_zone_occupied=(
                None if evaluation is None else evaluation.conflict_zone_occupied
            ),
            observed_background_vehicles=observed_background_vehicles,
        )

    @staticmethod
    def _action_index(
        env: MajorRoadStraightIntersectionEnv,
        action_name: str,
    ) -> int:
        return int(env.action_type.actions_indexes[action_name])

    @staticmethod
    def _reduce_target_speed(
        env: MajorRoadStraightIntersectionEnv,
        target_index: int,
    ) -> str:
        ego = env.vehicle
        desired_speed = float(env.action_type.target_speeds[target_index])
        return "SLOWER" if ego.target_speed > desired_speed + 1e-6 else "IDLE"

    @staticmethod
    def _increase_to_maximum_speed(
        env: MajorRoadStraightIntersectionEnv,
    ) -> str:
        maximum_speed = float(env.action_type.target_speeds[-1])
        return "FASTER" if env.vehicle.target_speed < maximum_speed - 1e-6 else "IDLE"


def format_predicted_clearance(value: float | None) -> str:
    """Format a clearance diagnostic for command-line output."""
    if value is None:
        return "not checked"
    if value == inf:
        return "infinity"
    return f"{value:.2f} m"
