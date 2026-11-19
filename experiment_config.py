"""Declared traffic, sensing, search grids, and output paths for report reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from vehicle_observations import CorrelatedGaussianNoiseConfig, FROZEN_CORRELATED_NOISE_CONFIG, FROZEN_OBSERVATION_LATENCY_CONFIG
from state_estimation import FROZEN_KALMAN_ACCELERATION_PROCESS_STD

EVALUATION_SEEDS = tuple(range(10_000, 10_050))

EVALUATION_TRAFFIC_SPAWN_PROBABILITIES = (0.20, 0.45, 0.70)

EVALUATION_EPISODE_DURATION = 40.0

EVALUATION_INITIAL_VEHICLE_COUNT = 3

EVALUATION_POLICY_FREQUENCY = 1

EVALUATION_HAZARD_DURATION = 35.0

EVALUATION_HAZARD_ENVIRONMENT_SEED = 20_000

@dataclass(frozen=True)
class DriverProfileParameters:
    name: str
    desired_speed: float
    maximum_acceleration: float
    comfortable_acceleration: float
    comfortable_deceleration: float
    standstill_distance: float
    time_headway: float

@dataclass(frozen=True)
class DriverShift:
    name: str
    probabilities: tuple[float, float, float]
    profiles: tuple[
        DriverProfileParameters,
        DriverProfileParameters,
        DriverProfileParameters,
    ]

SLOW_HEAVY_SHIFT = DriverShift(
    name="slow_heavy_cautious",
    probabilities=(0.60, 0.25, 0.15),
    profiles=(
        DriverProfileParameters("slow", 5.5, 3.5, 1.8, -2.8, 11.0, 2.2),
        DriverProfileParameters("normal", 7.5, 5.5, 2.8, -3.8, 9.0, 1.7),
        DriverProfileParameters("fast", 9.5, 5.5, 3.8, -4.8, 7.0, 1.2),
    ),
)

FAST_HEAVY_SHIFT = DriverShift(
    name="fast_heavy_assertive",
    probabilities=(0.15, 0.25, 0.60),
    profiles=(
        DriverProfileParameters("slow", 6.5, 4.5, 2.2, -3.2, 9.0, 1.8),
        DriverProfileParameters("normal", 8.5, 6.5, 3.2, -4.2, 7.0, 1.3),
        DriverProfileParameters("fast", 10.0, 6.5, 4.2, -5.2, 5.0, 0.8),
    ),
)

EVALUATION_DRIVER_SHIFTS = (SLOW_HEAVY_SHIFT, FAST_HEAVY_SHIFT)

@dataclass(frozen=True)
class SensorShift:
    name: str
    correlation: float
    position_bias_innovation_std: float
    position_white_std: float
    speed_bias_innovation_std: float
    speed_white_std: float
    latency: float

EXACT_SENSOR_BASELINE = SensorShift("exact", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

INTERMEDIATE_SENSOR_SHIFT = SensorShift(
    "rho0p87_latency0p75",
    0.87,
    0.394,
    0.25,
    0.158,
    0.125,
    0.75,
)

PERSISTENT_SENSOR_SHIFT = SensorShift(
    "rho0p93_latency1p25",
    0.93,
    0.441,
    0.40,
    0.176,
    0.20,
    1.25,
)

EVALUATION_SENSOR_SHIFTS = (
    EXACT_SENSOR_BASELINE,
    INTERMEDIATE_SENSOR_SHIFT,
    PERSISTENT_SENSOR_SHIFT,
)

EVALUATION_KALMAN_ACCELERATION_PROCESS_STD = 1.5

EVALUATION_EGO_HARD_BRAKING_THRESHOLD = 3.0

@dataclass(frozen=True)
class HazardVariant:
    name: str
    description: str
    truth_parameters: tuple[tuple[str, float | int | str], ...]
    sensor_shift: str
    noise_seed: int | None = None

EVALUATION_HAZARD_VARIANTS = (
    HazardVariant(
        "left_boundary_ttc_shift",
        "Left priority vehicle with an unseen 4.7 s initial TTC.",
        (("origin", 1), ("front_distance", 47.0), ("speed", 10.0)),
        "exact",
    ),
    HazardVariant(
        "right_boundary_speed_shift",
        "Right priority vehicle with shifted speed and boundary distance.",
        (("origin", 3), ("front_distance", 43.0), ("speed", 9.5)),
        "exact",
    ),
    HazardVariant(
        "shifted_slow_leader_fast_follower",
        "Unseen leader/follower speeds and spacing on the left approach.",
        (
            ("origin", 1),
            ("leader_front_distance", 16.0),
            ("leader_speed", 5.5),
            ("follower_front_distance", 29.5),
            ("follower_speed", 9.5),
        ),
        "exact",
    ),
    HazardVariant(
        "shifted_two_direction_arrival",
        "Asymmetric unseen arrival distances and speeds from both directions.",
        (
            ("left_front_distance", 19.0),
            ("left_speed", 9.5),
            ("right_front_distance", 23.0),
            ("right_speed", 10.0),
        ),
        "exact",
    ),
    HazardVariant(
        "shifted_leader_induced_braking",
        "Unseen slow-leader and fast-follower braking geometry.",
        (
            ("origin", 1),
            ("leader_front_distance", 17.0),
            ("leader_speed", 2.5),
            ("follower_front_distance", 31.5),
            ("follower_speed", 9.5),
        ),
        "exact",
    ),
    HazardVariant(
        "mirrored_biased_latency_boundary",
        "Mirrored near-threshold arrival under the persistent held-out sensor.",
        (("origin", 3), ("front_distance", 37.4), ("speed", 8.5)),
        "rho0p93_latency1p25",
        noise_seed=20_071,
    ),
)

EVALUATION_RAW_RESULTS_PATH = "results/baselines_raw.csv"

EVALUATION_SUMMARY_RESULTS_PATH = "results/baselines_summary.csv"

EVALUATION_CHECKPOINT_PATH = "results/baselines_checkpoint.csv"

def stochastic_conditions_per_controller() -> int:
    return (
        len(EVALUATION_SEEDS)
        * len(EVALUATION_TRAFFIC_SPAWN_PROBABILITIES)
        * len(EVALUATION_DRIVER_SHIFTS)
        * len(EVALUATION_SENSOR_SHIFTS)
    )

def planned_episode_count(controller_count: int = 2) -> int:
    """Return the final stochastic plus deterministic-hazard episode count."""
    return controller_count * (
        stochastic_conditions_per_controller() + len(EVALUATION_HAZARD_VARIANTS)
    )

PROJECT_DIRECTORY = Path(__file__).resolve().parent
PROTOCOL_VERSION = "report_reproduction_v1"
FIXED_GAP = "fixed_gap"

PREDICTIVE = "predictive_fixed_clearance"

UNCERTAINTY_AWARE = "predictive_uncertainty_aware"

FROZEN_CROSSING_TIME = 5.0

CURRENT_GAP_MARGIN = 0.0

DEFAULT_SPAWN_PROBABILITIES = (0.1, 0.3, 0.6)

BRAKING_THRESHOLDS = (2.5, 3.0, 3.5)

FROZEN_DEVELOPMENT_REALISM_CANDIDATE_NAME = "medium_q1p5"

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "results"

@dataclass(frozen=True)
class RealismCandidate:
    """One joint sensing, latency, and filter-process candidate."""

    name: str
    correlation: float
    position_bias_innovation_std: float
    position_white_std: float
    speed_bias_innovation_std: float
    speed_white_std: float
    latency: float
    acceleration_process_std: float

    @property
    def noise(self) -> CorrelatedGaussianNoiseConfig:
        return CorrelatedGaussianNoiseConfig(
            correlation=self.correlation,
            position_bias_innovation_std=self.position_bias_innovation_std,
            position_white_std=self.position_white_std,
            speed_bias_innovation_std=self.speed_bias_innovation_std,
            speed_white_std=self.speed_white_std,
        )
REALISM_CANDIDATES = (
RealismCandidate("exact", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5),
RealismCandidate("low", 0.80, 0.30, 0.20, 0.12, 0.10, 0.5, 1.0),
RealismCandidate(
        FROZEN_DEVELOPMENT_REALISM_CANDIDATE_NAME,
        FROZEN_CORRELATED_NOISE_CONFIG.correlation,
        FROZEN_CORRELATED_NOISE_CONFIG.position_bias_innovation_std,
        FROZEN_CORRELATED_NOISE_CONFIG.position_white_std,
        FROZEN_CORRELATED_NOISE_CONFIG.speed_bias_innovation_std,
        FROZEN_CORRELATED_NOISE_CONFIG.speed_white_std,
        FROZEN_OBSERVATION_LATENCY_CONFIG.latency,
        FROZEN_KALMAN_ACCELERATION_PROCESS_STD,
    ),
RealismCandidate("high", 0.95, 0.468, 0.50, 0.187, 0.25, 1.0, 1.5),
)
COARSE_BASE_CLEARANCES = (0.0, 0.5, 1.0, 1.5, 2.0)

COARSE_UNCERTAINTY_MULTIPLIERS = (0.0, 0.025, 0.05, 0.075, 0.1)

@dataclass(frozen=True, order=True)
class AdaptiveCandidate:
    """One development-only ``(b_0, k)`` pair."""

    base_clearance: float
    uncertainty_multiplier: float

    def __post_init__(self) -> None:
        if not isfinite(self.base_clearance) or self.base_clearance < 0:
            raise ValueError("base_clearance must be finite and non-negative")
        if (
            not isfinite(self.uncertainty_multiplier)
            or self.uncertainty_multiplier < 0
        ):
            raise ValueError(
                "uncertainty_multiplier must be finite and non-negative"
            )
NOISE_BY_NAME = {candidate.name: candidate for candidate in REALISM_CANDIDATES}
