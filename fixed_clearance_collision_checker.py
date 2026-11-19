"""Fixed-clearance checks for predicted oriented vehicle footprints.

The clearance parameter ``b`` is measured in metres as the minimum Euclidean
edge-to-edge distance between the physical ego and background rectangles. A
candidate plan is safe only if no physical footprints overlap and every
sampled clearance is at least ``b``.

All reported times are relative to the prediction start. The caller supplies
only trajectories that should be treated as potentially conflicting.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import atan2, cos, hypot, inf, isfinite, sin
from typing import TypeAlias

from vehicle_observations import CandidateTrajectory, PredictedVehicleState


Point: TypeAlias = tuple[float, float]
Rectangle: TypeAlias = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class FixedClearanceResult:
    """Safety decision and closest-approach diagnostics for one ego plan."""

    safe: bool
    required_clearance: float
    minimum_clearance: float
    time_of_minimum_clearance: float | None
    vehicle_id_of_minimum_clearance: int | None
    physical_overlap_predicted: bool


def oriented_rectangle_corners(state: PredictedVehicleState) -> Rectangle:
    """Return counter-clockwise corners of one physical vehicle footprint."""
    _validate_state(state)
    center_x, center_y = state.position
    longitudinal = (cos(state.heading), sin(state.heading))
    lateral = (-longitudinal[1], longitudinal[0])
    half_length = state.length / 2.0
    half_width = state.width / 2.0

    front_x = longitudinal[0] * half_length
    front_y = longitudinal[1] * half_length
    left_x = lateral[0] * half_width
    left_y = lateral[1] * half_width
    return (
        (center_x + front_x + left_x, center_y + front_y + left_y),
        (center_x - front_x + left_x, center_y - front_y + left_y),
        (center_x - front_x - left_x, center_y - front_y - left_y),
        (center_x + front_x - left_x, center_y + front_y - left_y),
    )


def rectangles_overlap(
    first: Rectangle,
    second: Rectangle,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether two oriented rectangles overlap or touch."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    for rectangle in (first, second):
        for index in range(2):
            edge_start = rectangle[index]
            edge_end = rectangle[(index + 1) % 4]
            edge = (
                edge_end[0] - edge_start[0],
                edge_end[1] - edge_start[1],
            )
            axis = (-edge[1], edge[0])
            axis_length = hypot(*axis)
            if axis_length <= tolerance:
                raise ValueError("rectangle contains a zero-length edge")
            unit_axis = (axis[0] / axis_length, axis[1] / axis_length)
            first_projection = _project(first, unit_axis)
            second_projection = _project(second, unit_axis)
            if (
                first_projection[1] < second_projection[0] - tolerance
                or second_projection[1] < first_projection[0] - tolerance
            ):
                return False
    return True


def rectangle_edge_clearance(
    first: Rectangle,
    second: Rectangle,
    *,
    tolerance: float = 1e-9,
) -> float:
    """Return non-negative Euclidean edge clearance between two rectangles."""
    if rectangles_overlap(first, second, tolerance=tolerance):
        return 0.0

    distances: list[float] = []
    for point in first:
        for edge_start, edge_end in _edges(second):
            distances.append(_point_segment_distance(point, edge_start, edge_end))
    for point in second:
        for edge_start, edge_end in _edges(first):
            distances.append(_point_segment_distance(point, edge_start, edge_end))
    return min(distances)


def check_fixed_clearance(
    ego_trajectory: CandidateTrajectory,
    background_trajectories: tuple[CandidateTrajectory, ...],
    *,
    clearance: float,
    tolerance: float = 1e-9,
) -> FixedClearanceResult:
    """Check an ego plan against predicted background-vehicle trajectories.

    Background states are sampled at every ego prediction time. Linear
    interpolation is used when a requested time lies between two background
    samples. A background trajectory must cover the full ego trajectory.
    """
    if not isfinite(clearance) or clearance < 0:
        raise ValueError("clearance must be a finite non-negative value")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative value")

    _validate_trajectory(ego_trajectory)
    for trajectory in background_trajectories:
        _validate_trajectory(trajectory)
        if trajectory.samples[0].vehicle_id == ego_trajectory.samples[0].vehicle_id:
            raise ValueError("background trajectories must not contain the ego")

    minimum_clearance = inf
    minimum_time: float | None = None
    minimum_vehicle_id: int | None = None
    overlap_predicted = False
    unsafe = False

    for ego_state in ego_trajectory.samples:
        absolute_time = ego_trajectory.start_time + ego_state.relative_time
        ego_rectangle = oriented_rectangle_corners(ego_state)

        for trajectory in background_trajectories:
            background_relative_time = absolute_time - trajectory.start_time
            background_state = _interpolate_state(
                trajectory,
                background_relative_time,
                tolerance=tolerance,
            )
            background_rectangle = oriented_rectangle_corners(background_state)
            physical_overlap = rectangles_overlap(
                ego_rectangle,
                background_rectangle,
                tolerance=tolerance,
            )
            edge_clearance = rectangle_edge_clearance(
                ego_rectangle,
                background_rectangle,
                tolerance=tolerance,
            )

            overlap_predicted = overlap_predicted or physical_overlap
            if physical_overlap or edge_clearance + tolerance < clearance:
                unsafe = True

            if edge_clearance < minimum_clearance - tolerance:
                minimum_clearance = edge_clearance
                minimum_time = ego_state.relative_time
                minimum_vehicle_id = background_state.vehicle_id

    return FixedClearanceResult(
        safe=not unsafe,
        required_clearance=float(clearance),
        minimum_clearance=float(minimum_clearance),
        time_of_minimum_clearance=minimum_time,
        vehicle_id_of_minimum_clearance=minimum_vehicle_id,
        physical_overlap_predicted=overlap_predicted,
    )


def _validate_state(state: PredictedVehicleState) -> None:
    numeric_values = (
        state.relative_time,
        *state.position,
        *state.velocity,
        state.heading,
        state.length,
        state.width,
    )
    if not all(isfinite(value) for value in numeric_values):
        raise ValueError("predicted vehicle states must contain finite values")
    if state.length <= 0 or state.width <= 0:
        raise ValueError("predicted vehicle dimensions must be positive")


def _validate_trajectory(trajectory: CandidateTrajectory) -> None:
    if not isfinite(trajectory.start_time):
        raise ValueError("trajectory start_time must be finite")

    first = trajectory.samples[0]
    previous_time: float | None = None
    for sample in trajectory.samples:
        _validate_state(sample)
        if sample.vehicle_id != first.vehicle_id:
            raise ValueError("all trajectory samples must have one vehicle_id")
        if sample.length != first.length or sample.width != first.width:
            raise ValueError("vehicle dimensions must remain constant in a trajectory")
        if previous_time is not None and sample.relative_time <= previous_time:
            raise ValueError("trajectory sample times must be strictly increasing")
        previous_time = sample.relative_time


def _interpolate_state(
    trajectory: CandidateTrajectory,
    relative_time: float,
    *,
    tolerance: float,
) -> PredictedVehicleState:
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


def _project(rectangle: Rectangle, axis: Point) -> tuple[float, float]:
    projections = [point[0] * axis[0] + point[1] * axis[1] for point in rectangle]
    return min(projections), max(projections)


def _edges(rectangle: Rectangle) -> tuple[tuple[Point, Point], ...]:
    return tuple(
        (rectangle[index], rectangle[(index + 1) % 4])
        for index in range(4)
    )


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    edge_x = end[0] - start[0]
    edge_y = end[1] - start[1]
    squared_length = edge_x * edge_x + edge_y * edge_y
    if squared_length == 0:
        return hypot(point[0] - start[0], point[1] - start[1])

    fraction = (
        (point[0] - start[0]) * edge_x + (point[1] - start[1]) * edge_y
    ) / squared_length
    fraction = max(0.0, min(1.0, fraction))
    closest = (start[0] + fraction * edge_x, start[1] + fraction * edge_y)
    return hypot(point[0] - closest[0], point[1] - closest[1])
