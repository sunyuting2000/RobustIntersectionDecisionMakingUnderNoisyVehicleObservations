"""Ground-truth-only intersection safety and disturbance metrics.

This module is evaluation infrastructure. Controllers never import it, and no
value calculated here is returned to a controller or used to change an action.

The intersection TTC is a conflict-zone TTC: using current ground-truth
longitudinal position and speed, it reports the time until the ego and one
priority vehicle would first occupy their respective intersection lanes at the
same time under constant-speed motion. It is a diagnostic, not a collision
probability.

PET uses route-pair-specific physical conflict regions. For each perpendicular
ego/priority pair, the region is centred at the centreline crossing and its
length along either route equals the opposing vehicle physical width. Entry
and exit are expanded by the tracked vehicle half-length, so simultaneous
occupancy represents possible rectangular-footprint interaction rather than
merely sharing the broad intersection. Longitudinal hard braking is evaluated
only for the ego vehicle and is derived from physics-rate speed differences.
Every contiguous event retains both ground-truth and controller-perceived
conflict-zone TTC at its onset for later justified/phantom-braking analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from math import inf, isfinite

from highway_env.vehicle.kinematics import Vehicle

from fixed_gap_controller import (
    EGO_EXIT_LANE,
    EGO_INCOMING_LANE,
    EGO_INTERSECTION_LANE,
    MajorRoadStraightIntersectionEnv,
)
from vehicle_observations import ObservationSnapshot, ObservedVehicleState


FROZEN_HARD_BRAKING_THRESHOLD = 3.0
PET_DEFINITION = "pairwise_physical_conflict_region_v2"


@dataclass(frozen=True)
class SafetyMetricConfig:
    """Configuration for evaluation-only physics-rate metrics."""

    hard_braking_threshold: float = FROZEN_HARD_BRAKING_THRESHOLD
    speed_epsilon: float = 0.1
    time_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if (
            not isfinite(self.hard_braking_threshold)
            or self.hard_braking_threshold <= 0
        ):
            raise ValueError("hard_braking_threshold must be finite and positive")
        if not isfinite(self.speed_epsilon) or self.speed_epsilon <= 0:
            raise ValueError("speed_epsilon must be finite and positive")
        if not isfinite(self.time_tolerance) or self.time_tolerance < 0:
            raise ValueError("time_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class ConflictZoneState:
    """One vehicle's longitudinal footprint relative to its conflict zone.

    ``center_progress`` is zero at the start of the relevant longitudinal
    conflict region. ``zone_length`` is the region length along this route.
    """

    center_progress: float
    zone_length: float
    vehicle_length: float
    speed: float

    def __post_init__(self) -> None:
        for name, value in (
            ("center_progress", self.center_progress),
            ("zone_length", self.zone_length),
            ("vehicle_length", self.vehicle_length),
            ("speed", self.speed),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.zone_length <= 0:
            raise ValueError("zone_length must be positive")
        if self.vehicle_length <= 0:
            raise ValueError("vehicle_length must be positive")
        if self.speed < 0:
            raise ValueError("speed must be non-negative")

    @property
    def entry_threshold(self) -> float:
        """Centre progress when the front bumper first enters the zone."""
        return -self.vehicle_length / 2

    @property
    def exit_threshold(self) -> float:
        """Centre progress when the rear bumper has cleared the zone."""
        return self.zone_length + self.vehicle_length / 2

    @property
    def occupies_zone(self) -> bool:
        return (
            self.center_progress >= self.entry_threshold
            and self.center_progress <= self.exit_threshold
        )


@dataclass(frozen=True)
class ConflictZonePassage:
    """Actual entry and optional exit time for one conflict-zone traversal."""

    entry_time: float
    exit_time: float | None

    def __post_init__(self) -> None:
        if not isfinite(self.entry_time):
            raise ValueError("entry_time must be finite")
        if self.exit_time is not None:
            if not isfinite(self.exit_time):
                raise ValueError("exit_time must be finite when present")
            if self.exit_time < self.entry_time:
                raise ValueError("exit_time cannot precede entry_time")


@dataclass(frozen=True)
class EgoHardBrakingEvent:
    """One contiguous ego hard-braking event and its onset context.

    ``actual_ttc`` uses simulator truth only for evaluation. ``perceived_ttc``
    uses the controller's most recent observation and is propagated from its
    1 Hz decision time to the physics-rate event onset under constant speed.
    Neither value is returned to a controller.
    """

    onset_time: float
    onset_acceleration: float
    peak_deceleration: float
    actual_ttc: float | None
    perceived_ttc: float | None
    controller_state: str | None
    action_name: str | None
    stop_line_distance: float | None
    perception_age: float


@dataclass(frozen=True)
class EpisodeSafetyMetrics:
    """Episode-level ground-truth metrics and event records."""

    minimum_actual_ttc: float | None
    minimum_actual_ttc_during_commit: float | None
    minimum_pet: float | None
    simultaneous_zone_occupancy: bool
    ego_hard_braking_events: int
    ego_hard_braking_event_records: tuple[EgoHardBrakingEvent, ...]
    minimum_actual_ttc_at_ego_hard_braking: float | None
    mean_actual_ttc_at_ego_hard_braking: float | None
    minimum_perceived_ttc_at_ego_hard_braking: float | None
    mean_perceived_ttc_at_ego_hard_braking: float | None
    ego_hard_braking_events_without_actual_ttc: int
    ego_hard_braking_events_without_perceived_ttc: int
    maximum_ego_hard_braking_deceleration: float | None
    hard_braking_threshold: float
    physics_samples: int
    pet_vehicle_pairs: int


def serialize_ego_hard_braking_events(
    events: tuple[EgoHardBrakingEvent, ...],
) -> str:
    """Serialize event-level diagnostics into a stable CSV-safe JSON value."""
    return json.dumps(
        [asdict(event) for event in events],
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class _RoutePath:
    incoming: tuple[str, str, int]
    conflict: tuple[str, str, int]
    outgoing: tuple[str, str, int]

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return (
            self.incoming[:2],
            self.conflict[:2],
            self.outgoing[:2],
        )


_EGO_PATH = _RoutePath(
    incoming=EGO_INCOMING_LANE,
    conflict=EGO_INTERSECTION_LANE,
    outgoing=EGO_EXIT_LANE,
)
_PRIORITY_PATHS = (
    _RoutePath(
        incoming=("o1", "ir1", 0),
        conflict=("ir1", "il3", 0),
        outgoing=("il3", "o3", 0),
    ),
    _RoutePath(
        incoming=("o3", "ir3", 0),
        conflict=("ir3", "il1", 0),
        outgoing=("il1", "o1", 0),
    ),
)


def projected_conflict_zone_ttc(
    first: ConflictZoneState,
    second: ConflictZoneState,
    *,
    speed_epsilon: float = 0.1,
) -> float | None:
    """Return time to predicted simultaneous zone occupancy, if any.

    Each occupancy interval is projected from the current true state under
    constant non-negative longitudinal speed. ``None`` means the two projected
    intervals do not overlap or a stopped vehicle has not yet entered.
    """
    if not isfinite(speed_epsilon) or speed_epsilon <= 0:
        raise ValueError("speed_epsilon must be finite and positive")
    first_interval = _future_occupancy_interval(first, speed_epsilon)
    second_interval = _future_occupancy_interval(second, speed_epsilon)
    if first_interval is None or second_interval is None:
        return None
    overlap_start = max(first_interval[0], second_interval[0])
    overlap_end = min(first_interval[1], second_interval[1])
    if overlap_start <= overlap_end + 1e-12:
        return max(0.0, overlap_start)
    return None


def observed_conflict_zone_ttc(
    env: MajorRoadStraightIntersectionEnv,
    observation: ObservationSnapshot,
    *,
    speed_epsilon: float = 0.1,
) -> float | None:
    """Return TTC perceived from one controller observation.

    The ego state is exact at the observation delivery time; background states
    are the filtered means supplied to the controller. The calculation uses
    the same conflict-zone definition as the ground-truth evaluation metric.
    """
    if not isfinite(speed_epsilon) or speed_epsilon <= 0:
        raise ValueError("speed_epsilon must be finite and positive")
    ego = _observed_conflict_zone_state(env, observation.ego, _EGO_PATH)
    if ego is None:
        return None
    values = []
    for vehicle in observation.other_vehicles:
        background = _observed_priority_conflict_zone_state(env, vehicle)
        if background is None:
            continue
        ttc = projected_conflict_zone_ttc(
            ego,
            background,
            speed_epsilon=speed_epsilon,
        )
        if ttc is not None:
            values.append(ttc)
    return min(values) if values else None


def post_encroachment_time(
    first: ConflictZonePassage,
    second: ConflictZonePassage,
) -> float:
    """Return pairwise PET; zero denotes simultaneous zone occupancy."""
    first_exit = inf if first.exit_time is None else first.exit_time
    second_exit = inf if second.exit_time is None else second.exit_time
    if first_exit <= second.entry_time:
        return max(0.0, second.entry_time - first_exit)
    if second_exit <= first.entry_time:
        return max(0.0, first.entry_time - second_exit)
    return 0.0


def _future_occupancy_interval(
    state: ConflictZoneState,
    speed_epsilon: float,
) -> tuple[float, float] | None:
    if state.center_progress > state.exit_threshold:
        return None
    if state.speed < speed_epsilon:
        return (0.0, inf) if state.occupies_zone else None
    entry_time = max(
        0.0,
        (state.entry_threshold - state.center_progress) / state.speed,
    )
    exit_time = max(
        0.0,
        (state.exit_threshold - state.center_progress) / state.speed,
    )
    if exit_time + 1e-12 < entry_time:
        return None
    return entry_time, exit_time


@dataclass
class _PassageTracker:
    entry_time: float | None = None
    exit_time: float | None = None
    previous_time: float | None = None
    previous_progress: float | None = None

    def passage(self) -> ConflictZonePassage | None:
        if self.entry_time is None:
            return None
        return ConflictZonePassage(self.entry_time, self.exit_time)


@dataclass
class _PairPassageTracker:
    """Independent ego/background passages through one physical conflict."""

    ego: _PassageTracker = field(default_factory=_PassageTracker)
    background: _PassageTracker = field(default_factory=_PassageTracker)


@dataclass
class _BrakingState:
    previous_time: float | None = None
    previous_speed: float | None = None
    active: bool = False
    events: int = 0
    event_records: list[EgoHardBrakingEvent] = field(default_factory=list)
    previous_actual_ttc: float | None = None
    previous_perceived_ttc: float | None = None
    previous_controller_state: str | None = None
    previous_action_name: str | None = None
    previous_stop_line_distance: float | None = None
    previous_perception_age: float = 0.0


class GroundTruthSafetyTracker:
    """Accumulate TTC, pair-specific PET, and braking at physics frequency."""

    def __init__(
        self,
        env: MajorRoadStraightIntersectionEnv,
        *,
        config: SafetyMetricConfig | None = None,
    ) -> None:
        self.config = config or SafetyMetricConfig()
        simulation_frequency = float(env.config["simulation_frequency"])
        if not isfinite(simulation_frequency) or simulation_frequency <= 0:
            raise ValueError("simulation_frequency must be finite and positive")
        self.physics_time_step = 1.0 / simulation_frequency
        self._background_ids: dict[int, int] = {}
        self._background_objects: dict[int, Vehicle] = {}
        self._next_background_id = 0
        self._pet_pairs: dict[int, _PairPassageTracker] = {}
        self._ego_braking = _BrakingState()
        self._minimum_ttc = inf
        self._minimum_commit_ttc = inf
        self._physics_samples = 0
        self._last_sample_time: float | None = None

    def associate_background_vehicles(
        self,
        env: MajorRoadStraightIntersectionEnv,
    ) -> tuple[tuple[int, Vehicle], ...]:
        """Give current background objects stable evaluation-only IDs."""
        associated = []
        for vehicle in env.road.vehicles:
            if vehicle is env.vehicle:
                continue
            identity = id(vehicle)
            if identity not in self._background_ids:
                self._background_ids[identity] = self._next_background_id
                self._background_objects[identity] = vehicle
                self._next_background_id += 1
            associated.append((self._background_ids[identity], vehicle))
        return tuple(associated)

    def observe(
        self,
        *,
        absolute_time: float,
        ego: Vehicle,
        background_vehicles: tuple[tuple[int, Vehicle], ...],
        ego_committed: bool,
        env: MajorRoadStraightIntersectionEnv,
        perceived_ttc: float | None = None,
        controller_state: str | None = None,
        action_name: str | None = None,
        perception_age: float = 0.0,
    ) -> None:
        """Consume one simultaneous ground-truth sample."""
        if not isfinite(absolute_time):
            raise ValueError("absolute_time must be finite")
        if perceived_ttc is not None and (
            not isfinite(perceived_ttc) or perceived_ttc < 0
        ):
            raise ValueError("perceived_ttc must be finite and non-negative")
        if not isfinite(perception_age) or perception_age < 0:
            raise ValueError("perception_age must be finite and non-negative")
        tolerance = self.config.time_tolerance
        if (
            self._last_sample_time is not None
            and absolute_time < self._last_sample_time - tolerance
        ):
            raise ValueError("metric samples must be chronological")

        ego_zone = _conflict_zone_state(env, ego, _EGO_PATH)
        background_zones = tuple(
            (
                vehicle_id,
                vehicle,
                _priority_route_path(vehicle),
                _priority_conflict_zone_state(env, vehicle),
            )
            for vehicle_id, vehicle in background_vehicles
        )
        finite_ttc = self._sample_ttc(
            ego_zone,
            tuple(
                zone
                for _, _, _, zone in background_zones
                if zone is not None
            ),
        )
        if finite_ttc is not None:
            self._minimum_ttc = min(self._minimum_ttc, finite_ttc)
            if ego_committed:
                self._minimum_commit_ttc = min(
                    self._minimum_commit_ttc,
                    finite_ttc,
                )

        stop_line_distance = (
            None
            if ego_zone is None
            else ego_zone.entry_threshold - ego_zone.center_progress
        )

        if (
            self._last_sample_time is not None
            and absolute_time <= self._last_sample_time + tolerance
        ):
            self._set_ego_braking_context(
                actual_ttc=finite_ttc,
                perceived_ttc=perceived_ttc,
                controller_state=controller_state,
                action_name=action_name,
                stop_line_distance=stop_line_distance,
                perception_age=perception_age,
            )
            return
        self._last_sample_time = float(absolute_time)
        self._physics_samples += 1

        self._update_ego_braking(
            float(ego.speed),
            absolute_time,
            actual_ttc=finite_ttc,
            perceived_ttc=perceived_ttc,
            controller_state=controller_state,
            action_name=action_name,
            stop_line_distance=stop_line_distance,
            perception_age=perception_age,
        )

        for vehicle_id, vehicle, path, _ in background_zones:
            if path is None:
                continue
            pair_states = _pair_pet_conflict_zone_states(
                env,
                ego,
                vehicle,
                path,
            )
            if pair_states is None:
                continue
            pair = self._pet_pairs.setdefault(
                vehicle_id,
                _PairPassageTracker(),
            )
            ego_pet_zone, background_pet_zone = pair_states
            self._update_passage(pair.ego, ego_pet_zone, absolute_time)
            self._update_passage(
                pair.background,
                background_pet_zone,
                absolute_time,
            )

    def result(self) -> EpisodeSafetyMetrics:
        """Return immutable episode metrics collected so far."""
        pet_values = []
        for pair in self._pet_pairs.values():
            ego_passage = pair.ego.passage()
            background_passage = pair.background.passage()
            if ego_passage is not None and background_passage is not None:
                pet_values.append(
                    post_encroachment_time(ego_passage, background_passage)
                )

        minimum_pet = min(pet_values) if pet_values else None
        ego_braking = self._ego_braking
        actual_ttc_values = [
            event.actual_ttc
            for event in ego_braking.event_records
            if event.actual_ttc is not None
        ]
        perceived_ttc_values = [
            event.perceived_ttc
            for event in ego_braking.event_records
            if event.perceived_ttc is not None
        ]
        peak_decelerations = [
            event.peak_deceleration for event in ego_braking.event_records
        ]
        return EpisodeSafetyMetrics(
            minimum_actual_ttc=(
                None if self._minimum_ttc == inf else self._minimum_ttc
            ),
            minimum_actual_ttc_during_commit=(
                None
                if self._minimum_commit_ttc == inf
                else self._minimum_commit_ttc
            ),
            minimum_pet=minimum_pet,
            simultaneous_zone_occupancy=(
                minimum_pet is not None and minimum_pet <= self.config.time_tolerance
            ),
            ego_hard_braking_events=ego_braking.events,
            ego_hard_braking_event_records=tuple(ego_braking.event_records),
            minimum_actual_ttc_at_ego_hard_braking=(
                min(actual_ttc_values) if actual_ttc_values else None
            ),
            mean_actual_ttc_at_ego_hard_braking=(
                sum(actual_ttc_values) / len(actual_ttc_values)
                if actual_ttc_values
                else None
            ),
            minimum_perceived_ttc_at_ego_hard_braking=(
                min(perceived_ttc_values) if perceived_ttc_values else None
            ),
            mean_perceived_ttc_at_ego_hard_braking=(
                sum(perceived_ttc_values) / len(perceived_ttc_values)
                if perceived_ttc_values
                else None
            ),
            ego_hard_braking_events_without_actual_ttc=(
                ego_braking.events - len(actual_ttc_values)
            ),
            ego_hard_braking_events_without_perceived_ttc=(
                ego_braking.events - len(perceived_ttc_values)
            ),
            maximum_ego_hard_braking_deceleration=(
                max(peak_decelerations) if peak_decelerations else None
            ),
            hard_braking_threshold=self.config.hard_braking_threshold,
            physics_samples=self._physics_samples,
            pet_vehicle_pairs=len(pet_values),
        )

    def _sample_ttc(
        self,
        ego: ConflictZoneState | None,
        backgrounds: tuple[ConflictZoneState, ...],
    ) -> float | None:
        if ego is None:
            return None
        values = tuple(
            ttc
            for background in backgrounds
            if (
                ttc := projected_conflict_zone_ttc(
                    ego,
                    background,
                    speed_epsilon=self.config.speed_epsilon,
                )
            )
            is not None
        )
        return min(values) if values else None

    def _update_passage(
        self,
        tracker: _PassageTracker,
        zone: ConflictZoneState,
        absolute_time: float,
    ) -> None:
        previous_time = tracker.previous_time
        previous_progress = tracker.previous_progress

        if tracker.entry_time is None and zone.center_progress >= zone.entry_threshold:
            if zone.center_progress <= zone.exit_threshold:
                tracker.entry_time = _crossing_time(
                    previous_time,
                    previous_progress,
                    absolute_time,
                    zone.center_progress,
                    zone.entry_threshold,
                )
            elif previous_progress is not None and previous_progress < zone.entry_threshold:
                tracker.entry_time = _crossing_time(
                    previous_time,
                    previous_progress,
                    absolute_time,
                    zone.center_progress,
                    zone.entry_threshold,
                )

        if (
            tracker.entry_time is not None
            and tracker.exit_time is None
            and zone.center_progress >= zone.exit_threshold
        ):
            tracker.exit_time = _crossing_time(
                previous_time,
                previous_progress,
                absolute_time,
                zone.center_progress,
                zone.exit_threshold,
            )

        tracker.previous_time = float(absolute_time)
        tracker.previous_progress = zone.center_progress

    def _update_ego_braking(
        self,
        speed: float,
        absolute_time: float,
        *,
        actual_ttc: float | None = None,
        perceived_ttc: float | None = None,
        controller_state: str | None = None,
        action_name: str | None = None,
        stop_line_distance: float | None = None,
        perception_age: float = 0.0,
    ) -> None:
        state = self._ego_braking
        if state.previous_time is not None and state.previous_speed is not None:
            elapsed = absolute_time - state.previous_time
            if elapsed > self.physics_time_step * 1.5:
                state.active = False
            elif elapsed > self.config.time_tolerance:
                acceleration = (speed - state.previous_speed) / elapsed
                hard_braking = (
                    acceleration <= -self.config.hard_braking_threshold
                )
                if hard_braking and not state.active:
                    state.events += 1
                    state.event_records.append(
                        EgoHardBrakingEvent(
                            onset_time=float(state.previous_time),
                            onset_acceleration=float(acceleration),
                            peak_deceleration=float(-acceleration),
                            actual_ttc=state.previous_actual_ttc,
                            perceived_ttc=state.previous_perceived_ttc,
                            controller_state=state.previous_controller_state,
                            action_name=state.previous_action_name,
                            stop_line_distance=state.previous_stop_line_distance,
                            perception_age=state.previous_perception_age,
                        )
                    )
                elif hard_braking and state.event_records:
                    current = state.event_records[-1]
                    state.event_records[-1] = replace(
                        current,
                        peak_deceleration=max(
                            current.peak_deceleration,
                            float(-acceleration),
                        ),
                    )
                state.active = hard_braking
        state.previous_time = float(absolute_time)
        state.previous_speed = float(speed)
        self._set_ego_braking_context(
            actual_ttc=actual_ttc,
            perceived_ttc=perceived_ttc,
            controller_state=controller_state,
            action_name=action_name,
            stop_line_distance=stop_line_distance,
            perception_age=perception_age,
        )

    def _set_ego_braking_context(
        self,
        *,
        actual_ttc: float | None,
        perceived_ttc: float | None,
        controller_state: str | None,
        action_name: str | None,
        stop_line_distance: float | None,
        perception_age: float,
    ) -> None:
        state = self._ego_braking
        state.previous_actual_ttc = actual_ttc
        state.previous_perceived_ttc = perceived_ttc
        state.previous_controller_state = controller_state
        state.previous_action_name = action_name
        state.previous_stop_line_distance = stop_line_distance
        state.previous_perception_age = float(perception_age)


def capture_policy_step_safety_metrics(
    *,
    env: MajorRoadStraightIntersectionEnv,
    tracker: GroundTruthSafetyTracker,
    step_start_time: float,
    background_vehicles: tuple[tuple[int, Vehicle], ...],
    ego_committed: bool,
    perceived_ttc_at_step_start: float | None = None,
    controller_state: str | None = None,
    action_name: str | None = None,
) -> None:
    """Consume the just-completed policy step at simulation frequency."""
    simulation_frequency = int(env.config["simulation_frequency"])
    policy_frequency = int(env.config["policy_frequency"])
    if simulation_frequency <= 0 or policy_frequency <= 0:
        raise ValueError("simulation and policy frequencies must be positive")
    if simulation_frequency % policy_frequency:
        raise ValueError("simulation_frequency must be divisible by policy_frequency")
    frame_count = simulation_frequency // policy_frequency
    ego_states = _recent_physics_states(env.vehicle, frame_count)
    background_states = tuple(
        (
            vehicle_id,
            _recent_physics_states(vehicle, frame_count),
        )
        for vehicle_id, vehicle in background_vehicles
    )

    for frame_index, ego_state in enumerate(ego_states):
        absolute_time = step_start_time + (frame_index + 1) / simulation_frequency
        perception_age = absolute_time - step_start_time
        perceived_ttc = (
            None
            if perceived_ttc_at_step_start is None
            else max(0.0, perceived_ttc_at_step_start - perception_age)
        )
        tracker.observe(
            absolute_time=absolute_time,
            ego=ego_state,
            background_vehicles=tuple(
                (vehicle_id, states[frame_index])
                for vehicle_id, states in background_states
                if frame_index < len(states)
            ),
            ego_committed=ego_committed,
            env=env,
            perceived_ttc=perceived_ttc,
            controller_state=controller_state,
            action_name=action_name,
            perception_age=perception_age,
        )


def _recent_physics_states(
    vehicle: Vehicle,
    frame_count: int,
) -> tuple[Vehicle, ...]:
    newest_first = tuple(vehicle.history)
    if len(newest_first) < frame_count:
        raise RuntimeError(
            f"vehicle history has {len(newest_first)} states; "
            f"{frame_count} are required"
        )
    return tuple(reversed(newest_first[:frame_count]))


def _observed_priority_conflict_zone_state(
    env: MajorRoadStraightIntersectionEnv,
    vehicle: ObservedVehicleState,
) -> ConflictZoneState | None:
    for path in _PRIORITY_PATHS:
        state = _observed_conflict_zone_state(env, vehicle, path)
        if state is not None:
            return state
    return None


def _observed_conflict_zone_state(
    env: MajorRoadStraightIntersectionEnv,
    vehicle: ObservedVehicleState,
    path: _RoutePath,
) -> ConflictZoneState | None:
    if vehicle.lane_index is None:
        return None
    edge = vehicle.lane_index[:2]
    if edge not in path.edges:
        return None
    lane = env.road.network.get_lane(vehicle.lane_index)
    longitudinal, _ = lane.local_coordinates(vehicle.position)
    conflict_lane = env.road.network.get_lane(path.conflict)
    if edge == path.incoming[:2]:
        center_progress = float(longitudinal - lane.length)
    elif edge == path.conflict[:2]:
        center_progress = float(longitudinal)
    else:
        center_progress = float(conflict_lane.length + longitudinal)
    return ConflictZoneState(
        center_progress=center_progress,
        zone_length=float(conflict_lane.length),
        vehicle_length=vehicle.length,
        speed=max(0.0, vehicle.speed),
    )


def _priority_route_path(vehicle: Vehicle) -> _RoutePath | None:
    """Return the straight priority path currently occupied by a vehicle."""
    if vehicle.lane_index is None:
        return None
    edge = vehicle.lane_index[:2]
    for path in _PRIORITY_PATHS:
        if edge in path.edges:
            return path
    return None


def _route_center_progress(
    env: MajorRoadStraightIntersectionEnv,
    vehicle: Vehicle,
    path: _RoutePath,
) -> float | None:
    """Return centre progress from the start of the route conflict lane."""
    if vehicle.lane_index is None:
        return None
    edge = vehicle.lane_index[:2]
    if edge not in path.edges:
        return None
    lane = env.road.network.get_lane(vehicle.lane_index)
    longitudinal, _ = lane.local_coordinates(vehicle.position)
    conflict_lane = env.road.network.get_lane(path.conflict)
    if edge == path.incoming[:2]:
        return float(longitudinal - lane.length)
    if edge == path.conflict[:2]:
        return float(longitudinal)
    return float(conflict_lane.length + longitudinal)


def _straight_conflict_longitudinals(
    env: MajorRoadStraightIntersectionEnv,
    first_path: _RoutePath,
    second_path: _RoutePath,
) -> tuple[float, float] | None:
    """Return route longitudinal coordinates of two centreline crossings."""
    first_lane = env.road.network.get_lane(first_path.conflict)
    second_lane = env.road.network.get_lane(second_path.conflict)
    first_start = first_lane.position(0.0, 0.0)
    first_end = first_lane.position(first_lane.length, 0.0)
    second_start = second_lane.position(0.0, 0.0)
    second_end = second_lane.position(second_lane.length, 0.0)
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = second_start - first_start

    def cross(first, second) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    denominator = cross(first_direction, second_direction)
    if abs(denominator) <= 1e-12:
        return None
    first_fraction = cross(offset, second_direction) / denominator
    second_fraction = cross(offset, first_direction) / denominator
    tolerance = 1e-9
    if not (
        -tolerance <= first_fraction <= 1.0 + tolerance
        and -tolerance <= second_fraction <= 1.0 + tolerance
    ):
        return None

    crossing = first_start + first_fraction * first_direction
    first_longitudinal, _ = first_lane.local_coordinates(crossing)
    second_longitudinal, _ = second_lane.local_coordinates(crossing)
    return float(first_longitudinal), float(second_longitudinal)


def _pair_pet_conflict_zone_states(
    env: MajorRoadStraightIntersectionEnv,
    ego: Vehicle,
    background: Vehicle,
    background_path: _RoutePath,
) -> tuple[ConflictZoneState, ConflictZoneState] | None:
    """Build physical PET regions for one perpendicular route pair.

    Along each route, the region length equals the opposing vehicle width.
    Expanding that interval by the tracked vehicle half-length makes concurrent
    occupancy equivalent to possible overlap of the two axis-aligned vehicle
    rectangles on these fixed straight paths.
    """
    ego_progress = _route_center_progress(env, ego, _EGO_PATH)
    background_progress = _route_center_progress(
        env,
        background,
        background_path,
    )
    crossing = _straight_conflict_longitudinals(
        env,
        _EGO_PATH,
        background_path,
    )
    if (
        ego_progress is None
        or background_progress is None
        or crossing is None
    ):
        return None

    ego_crossing, background_crossing = crossing
    ego_zone_length = float(background.WIDTH)
    background_zone_length = float(ego.WIDTH)
    if ego_zone_length <= 0 or background_zone_length <= 0:
        raise ValueError("vehicle widths must be positive")

    ego_region_start = ego_crossing - ego_zone_length / 2
    background_region_start = (
        background_crossing - background_zone_length / 2
    )
    return (
        ConflictZoneState(
            center_progress=ego_progress - ego_region_start,
            zone_length=ego_zone_length,
            vehicle_length=float(ego.LENGTH),
            speed=max(0.0, float(ego.speed)),
        ),
        ConflictZoneState(
            center_progress=background_progress - background_region_start,
            zone_length=background_zone_length,
            vehicle_length=float(background.LENGTH),
            speed=max(0.0, float(background.speed)),
        ),
    )


def _priority_conflict_zone_state(
    env: MajorRoadStraightIntersectionEnv,
    vehicle: Vehicle,
) -> ConflictZoneState | None:
    for path in _PRIORITY_PATHS:
        state = _conflict_zone_state(env, vehicle, path)
        if state is not None:
            return state
    return None


def _conflict_zone_state(
    env: MajorRoadStraightIntersectionEnv,
    vehicle: Vehicle,
    path: _RoutePath,
) -> ConflictZoneState | None:
    if vehicle.lane_index is None:
        return None
    edge = vehicle.lane_index[:2]
    if edge not in path.edges:
        return None
    lane = env.road.network.get_lane(vehicle.lane_index)
    longitudinal, _ = lane.local_coordinates(vehicle.position)
    conflict_lane = env.road.network.get_lane(path.conflict)
    if edge == path.incoming[:2]:
        center_progress = float(longitudinal - lane.length)
    elif edge == path.conflict[:2]:
        center_progress = float(longitudinal)
    else:
        center_progress = float(conflict_lane.length + longitudinal)
    return ConflictZoneState(
        center_progress=center_progress,
        zone_length=float(conflict_lane.length),
        vehicle_length=float(vehicle.LENGTH),
        speed=max(0.0, float(vehicle.speed)),
    )


def _crossing_time(
    previous_time: float | None,
    previous_progress: float | None,
    current_time: float,
    current_progress: float,
    threshold: float,
) -> float:
    if (
        previous_time is None
        or previous_progress is None
        or current_progress <= previous_progress + 1e-12
        or previous_progress >= threshold
    ):
        return float(current_time)
    fraction = (threshold - previous_progress) / (
        current_progress - previous_progress
    )
    fraction = min(1.0, max(0.0, fraction))
    return float(previous_time + fraction * (current_time - previous_time))
