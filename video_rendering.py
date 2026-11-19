"""Shared rendering, diagnostic overlays, and sensing configuration for report-video replays."""

from __future__ import annotations



from dataclasses import dataclass





from typing import Iterable

import gymnasium as gym

import numpy as np

import pygame


from fixed_gap_controller import DEFAULT_DRIVER_PROFILE_PROBABILITIES, MajorRoadStraightIntersectionEnv, validate_driver_profile_probabilities

from experiment_config import EVALUATION_EPISODE_DURATION, EVALUATION_INITIAL_VEHICLE_COUNT, EVALUATION_KALMAN_ACCELERATION_PROCESS_STD, EVALUATION_POLICY_FREQUENCY

from experiment_config import FIXED_GAP, PREDICTIVE

from predictive_fixed_margin_controller import FROZEN_REALISTIC_FIXED_CLEARANCE

from experiment_config import NOISE_BY_NAME

from scenarios import DRIVER_SHIFT_BY_NAME, EVALUATION_VEHICLE_TYPES, SENSOR_SHIFT_BY_NAME





from vehicle_observations import CorrelatedGaussianNoiseConfig

DEVELOPMENT_EQUAL_MIX = "development_equal_mix"

@dataclass(frozen=True)
class RecordingSpec:
    """One predeclared post-hoc illustrative replay."""

    name: str
    description: str
    controller: str
    sensing_profile: str
    seed: int
    spawn_probability: float
    driver_shift: str
    base_clearance: float | None = None
    uncertainty_multiplier: float | None = None
    expected_outcome: str | None = None
    expected_waiting_time: float | None = None
    expected_required_clearance: float | None = None

@dataclass
class ReplayDiagnostics:
    """Latest diagnostics retained across the irrevocable COMMIT state."""

    available_gap: float | None = None
    conflict_zone_occupied: bool | None = None
    minimum_predicted_clearance: float | None = None
    required_clearance: float | None = None
    clearance_slack: float | None = None
    maximum_conflict_sigma: float | None = None

class RecordingShiftEnvironment(MajorRoadStraightIntersectionEnv):
    """Shifted-driver environment with RGB-array rendering enabled."""

    def __init__(self, *, driver_shift_name: str, config: dict) -> None:
        self.driver_shift_name = driver_shift_name
        super().__init__(config=config, render_mode="rgb_array")

    def _sample_driver_vehicle_type(self):
        probabilities = validate_driver_profile_probabilities(
            self.config["driver_profile_probabilities"]
        )
        profile_index = int(self.np_random.choice(3, p=probabilities))
        return EVALUATION_VEHICLE_TYPES[self.driver_shift_name][profile_index]

class DiagnosticOverlay(gym.Wrapper):
    """Render a compact, consistent diagnostic panel on recorded frames."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lines = ("Initialising illustrative replay",)
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)

    def set_lines(self, lines: Iterable[str]) -> None:
        self.lines = tuple(lines)

    def render(self):
        # Delayed observations require ``road.record_history=True`` so the
        # provider can reconstruct past truth states.  Highway-Env also uses
        # that same flag to draw translucent trajectories.  Suppress it only
        # for the duration of rendering, then restore it before simulation
        # continues; this changes pixels, not observations or controller data.
        road = getattr(self.unwrapped, "road", None)
        record_history = (
            bool(road.record_history) if road is not None else False
        )
        if road is not None:
            road.record_history = False
        try:
            frame = self.env.render()
        finally:
            if road is not None:
                road.record_history = record_history
        if frame is None:
            return None
        pixels = np.asarray(frame)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            return frame

        surface = pygame.surfarray.make_surface(
            np.transpose(pixels[:, :, :3], (1, 0, 2))
        )
        padding = 6
        line_height = self.font.get_linesize()
        panel_width = surface.get_width() - 2 * padding
        panel_height = padding * 2 + line_height * len(self.lines)
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 195))
        for index, line in enumerate(self.lines):
            rendered = self.font.render(line, True, (255, 255, 255))
            panel.blit(rendered, (padding, padding + index * line_height))
        surface.blit(panel, (padding, padding))
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

def _display(value: object, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)

def _sensing_configuration(name: str):
    """Return noise, latency, process std, correlation for one profile."""
    if name in NOISE_BY_NAME:
        profile = NOISE_BY_NAME[name]
        return (
            profile.noise,
            float(profile.latency),
            float(profile.acceleration_process_std),
            float(profile.correlation),
        )
    if name in SENSOR_SHIFT_BY_NAME:
        profile = SENSOR_SHIFT_BY_NAME[name]
        noise = CorrelatedGaussianNoiseConfig(
            correlation=profile.correlation,
            position_bias_innovation_std=(
                profile.position_bias_innovation_std
            ),
            position_white_std=profile.position_white_std,
            speed_bias_innovation_std=profile.speed_bias_innovation_std,
            speed_white_std=profile.speed_white_std,
        )
        return (
            noise,
            float(profile.latency),
            EVALUATION_KALMAN_ACCELERATION_PROCESS_STD,
            float(profile.correlation),
        )
    raise ValueError(f"unknown sensing profile: {name}")

def _make_environment(spec: RecordingSpec) -> MajorRoadStraightIntersectionEnv:
    if spec.driver_shift == DEVELOPMENT_EQUAL_MIX:
        probabilities = dict(DEFAULT_DRIVER_PROFILE_PROBABILITIES)
    else:
        shift = DRIVER_SHIFT_BY_NAME[spec.driver_shift]
        probabilities = {
            profile.name: probability
            for profile, probability in zip(
                shift.profiles,
                shift.probabilities,
                strict=True,
            )
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
        "spawn_probability": spec.spawn_probability,
        "driver_profile_probabilities": probabilities,
        # Retain history for delayed observations and hide translucent trails
        # in the recorded frames.
        "show_trajectories": False,
    }
    if spec.driver_shift == DEVELOPMENT_EQUAL_MIX:
        return MajorRoadStraightIntersectionEnv(
            config=config,
            render_mode="rgb_array",
        )
    return RecordingShiftEnvironment(
        driver_shift_name=spec.driver_shift,
        config=config,
    )

def _update_diagnostics(
    spec: RecordingSpec,
    decision: object,
    diagnostics: ReplayDiagnostics,
) -> None:
    if spec.controller == FIXED_GAP:
        diagnostics.available_gap = getattr(decision, "available_gap", None)
        diagnostics.conflict_zone_occupied = getattr(
            decision,
            "conflict_zone_occupied",
            None,
        )
        return

    minimum_clearance = getattr(
        decision,
        "minimum_predicted_clearance",
        None,
    )
    if minimum_clearance is not None:
        diagnostics.minimum_predicted_clearance = minimum_clearance

    if spec.controller == PREDICTIVE:
        diagnostics.required_clearance = FROZEN_REALISTIC_FIXED_CLEARANCE
        if diagnostics.minimum_predicted_clearance is not None:
            diagnostics.clearance_slack = (
                diagnostics.minimum_predicted_clearance
                - diagnostics.required_clearance
            )
        return

    required = getattr(decision, "maximum_required_clearance", None)
    slack = getattr(decision, "minimum_clearance_slack", None)
    profiles = getattr(decision, "adaptive_clearances", ())
    if required is not None:
        diagnostics.required_clearance = required
    if slack is not None:
        diagnostics.clearance_slack = slack
    if profiles:
        diagnostics.maximum_conflict_sigma = max(
            profile.maximum_position_std for profile in profiles
        )
