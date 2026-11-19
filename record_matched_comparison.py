"""Record the matched fixed-clearance exact-success and strong-degradation collision examples."""

from __future__ import annotations
from video_rendering import RecordingSpec, ReplayDiagnostics, DiagnosticOverlay, _display, _sensing_configuration, _make_environment, _update_diagnostics

import argparse

import csv


from math import isclose

from pathlib import Path

from statistics import mean

from time import perf_counter





from gymnasium.wrappers import RecordVideo

from fixed_gap_controller import ControllerState, ego_has_cleared_conflict_zone

from experiment_config import EVALUATION_EGO_HARD_BRAKING_THRESHOLD

from experiment_config import FIXED_GAP, PREDICTIVE

from predictive_fixed_margin_controller import FROZEN_REALISTIC_FIXED_CLEARANCE, PredictiveFixedMarginConfig, PredictiveFixedMarginController



from safety_metrics import PET_DEFINITION, GroundTruthSafetyTracker, SafetyMetricConfig, capture_policy_step_safety_metrics, observed_conflict_zone_ttc

from state_estimation import make_shared_kalman_observation_provider

from trajectory_prediction import DEFAULT_PREDICTION_CONFIG


from vehicle_observations import ObservationLatencyConfig

def _make_provider(spec: RecordingSpec):
    noise, latency, acceleration_std, _ = _sensing_configuration(
        spec.sensing_profile
    )
    return make_shared_kalman_observation_provider(
        noise=noise,
        latency=ObservationLatencyConfig(latency),
        acceleration_process_std=acceleration_std,
        seed=spec.seed,
    )

def _make_controller(spec, provider):
    if spec.controller != PREDICTIVE:
        raise ValueError("The matched comparison records Controller 2 only.")
    return PredictiveFixedMarginController(
        PredictiveFixedMarginConfig(
            clearance=FROZEN_REALISTIC_FIXED_CLEARANCE,
            prediction=DEFAULT_PREDICTION_CONFIG,
        ),
        observation_provider=provider,
    )

def _controller_parameter_line(spec: RecordingSpec) -> str:
    if spec.controller == FIXED_GAP:
        return "Controller 1 | critical gap=5.00 s (g=0.00 s)"
    if spec.controller == PREDICTIVE:
        return "Controller 2 | fixed clearance b=1.50 m"
    return (
        f"Controller 3 | b0={spec.base_clearance:.2f} m | "
        f"k={spec.uncertainty_multiplier:.3f}"
    )

def _overlay_lines(
    spec: RecordingSpec,
    decision: object,
    diagnostics: ReplayDiagnostics,
    *,
    simulation_time: float,
    waiting_time: float,
) -> tuple[str, ...]:
    _, latency, _, correlation = _sensing_configuration(spec.sensing_profile)
    state = getattr(decision, "state", None)
    state_text = getattr(state, "value", state)
    common = (
        f"ILLUSTRATIVE REPLAY | {spec.name}",
        _controller_parameter_line(spec),
        f"Sensing={spec.sensing_profile} | rho={correlation:.2f} | "
        f"latency={latency:.2f} s",
        f"t={simulation_time:.0f} s | state={state_text} | "
        f"action={getattr(decision, 'action_name', 'n/a')} | "
        f"wait={waiting_time:.0f} s",
    )
    if spec.controller == FIXED_GAP:
        occupied = diagnostics.conflict_zone_occupied
        occupied_text = "n/a" if occupied is None else str(bool(occupied))
        detail = (
            f"Available temporal gap={_display(diagnostics.available_gap)} s",
            f"Critical gap=5.00 s | conflict occupied={occupied_text}",
        )
    else:
        sigma_text = (
            "not used"
            if spec.controller == PREDICTIVE
            else f"{_display(diagnostics.maximum_conflict_sigma)} m"
        )
        detail = (
            f"Conflict-window sigma={sigma_text}",
            f"Required={_display(diagnostics.required_clearance)} m | "
            f"predicted min={_display(diagnostics.minimum_predicted_clearance)} m",
            f"Clearance slack={_display(diagnostics.clearance_slack)} m",
        )
    expected = (
        "Reviewed outcome recorded in manifest"
        if spec.expected_outcome is None
        else f"Reviewed outcome={spec.expected_outcome}"
    )
    return (*common, *detail, expected)

def _prefix(spec: RecordingSpec) -> str:
    spawn = str(spec.spawn_probability).replace(".", "p")
    return (
        f"{spec.name}-{spec.controller}-{spec.sensing_profile}-"
        f"seed-{spec.seed}-spawn-{spawn}"
    )

def _validate_reviewed_result(
    spec: RecordingSpec,
    *,
    outcome: str,
    waiting_time: float,
    required_clearance: float | None,
) -> None:
    if spec.expected_outcome is not None and outcome != spec.expected_outcome:
        raise RuntimeError(
            f"{spec.name} reproduced as {outcome}, not the reviewed "
            f"{spec.expected_outcome}"
        )
    if (
        spec.expected_waiting_time is not None
        and not isclose(
            waiting_time,
            spec.expected_waiting_time,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError(
            f"{spec.name} waiting time changed: {waiting_time} != "
            f"{spec.expected_waiting_time}"
        )
    if spec.expected_required_clearance is not None:
        if required_clearance is None or not isclose(
            required_clearance,
            spec.expected_required_clearance,
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            raise RuntimeError(
                f"{spec.name} required clearance changed: "
                f"{required_clearance!r} != approximately "
                f"{spec.expected_required_clearance}"
            )

def record_spec(
    spec: RecordingSpec,
    *,
    video_directory: Path,
    overwrite: bool,
) -> dict[str, object]:
    """Record one selected replay and return its manifest row."""
    prefix = _prefix(spec)
    video_path = video_directory / f"{prefix}-episode-0.mp4"
    if video_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"video already exists: {video_path}; pass --overwrite to "
                "replace this specific illustrative replay"
            )
        video_path.unlink()

    base_env = _make_environment(spec)
    provider = _make_provider(spec)
    controller = _make_controller(spec, provider)
    overlay_env = DiagnosticOverlay(base_env)
    recorded_env = RecordVideo(
        overlay_env,
        video_folder=str(video_directory),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=prefix,
        disable_logger=True,
    )
    base_env.set_record_video_wrapper(recorded_env)
    tracker = GroundTruthSafetyTracker(
        base_env,
        config=SafetyMetricConfig(
            hard_braking_threshold=EVALUATION_EGO_HARD_BRAKING_THRESHOLD
        ),
    )
    diagnostics = ReplayDiagnostics()
    waiting_time = 0.0
    commit_time: float | None = None
    clear_time: float | None = None
    commit_minimum_clearance: float | None = None
    commit_required_clearance: float | None = None
    commit_clearance_slack: float | None = None
    commit_maximum_sigma: float | None = None
    decision_times_ms: list[float] = []
    all_background_vehicles: set[object] = set()

    try:
        recorded_env.reset(seed=spec.seed)
        base_env.road.record_history = True
        controller.reset()
        controller.state = ControllerState.APPROACH
        terminated = False
        truncated = False
        while not (terminated or truncated):
            decision_start = perf_counter()
            decision = controller.decide(base_env)
            decision_times_ms.append(
                (perf_counter() - decision_start) * 1000.0
            )
            if decision.state is ControllerState.WAIT:
                waiting_time += 1.0 / float(
                    base_env.config["policy_frequency"]
                )

            _update_diagnostics(spec, decision, diagnostics)
            if decision.state is ControllerState.COMMIT and commit_time is None:
                commit_time = float(base_env.time)
                commit_minimum_clearance = (
                    diagnostics.minimum_predicted_clearance
                )
                commit_required_clearance = diagnostics.required_clearance
                commit_clearance_slack = diagnostics.clearance_slack
                commit_maximum_sigma = diagnostics.maximum_conflict_sigma

            overlay_env.set_lines(
                _overlay_lines(
                    spec,
                    decision,
                    diagnostics,
                    simulation_time=float(base_env.time),
                    waiting_time=waiting_time,
                )
            )
            if controller.last_observation is None:
                raise RuntimeError("controller did not retain its observation")
            step_start_time = float(base_env.time)
            perceived_ttc = observed_conflict_zone_ttc(
                base_env,
                controller.last_observation,
            )
            background_vehicles = tracker.associate_background_vehicles(
                base_env
            )
            all_background_vehicles.update(
                vehicle for _, vehicle in background_vehicles
            )
            tracker.observe(
                absolute_time=step_start_time,
                ego=base_env.vehicle,
                background_vehicles=background_vehicles,
                ego_committed=decision.state is ControllerState.COMMIT,
                env=base_env,
                perceived_ttc=perceived_ttc,
                controller_state=decision.state.value,
                action_name=decision.action_name,
            )
            _, _, terminated, truncated, _ = recorded_env.step(decision.action)
            capture_policy_step_safety_metrics(
                env=base_env,
                tracker=tracker,
                step_start_time=step_start_time,
                background_vehicles=background_vehicles,
                ego_committed=decision.state is ControllerState.COMMIT,
                perceived_ttc_at_step_start=perceived_ttc,
                controller_state=decision.state.value,
                action_name=decision.action_name,
            )
            if (
                ego_has_cleared_conflict_zone(base_env.vehicle)
                and clear_time is None
            ):
                clear_time = float(base_env.time)

        collision = bool(base_env.vehicle.crashed)
        success = bool(base_env.has_arrived(base_env.vehicle)) and not collision
        timeout = bool(truncated) and not collision and not success
        outcome = (
            "collision"
            if collision
            else "success"
            if success
            else "timeout"
            if timeout
            else "other"
        )
        episode_time = float(base_env.time)
        safety = tracker.result()
        background_collision = any(
            bool(getattr(vehicle, "crashed", False))
            for vehicle in all_background_vehicles
        )
    finally:
        recorded_env.close()

    _validate_reviewed_result(
        spec,
        outcome=outcome,
        waiting_time=waiting_time,
        required_clearance=commit_required_clearance,
    )
    _, latency, acceleration_std, correlation = _sensing_configuration(
        spec.sensing_profile
    )
    return {
        "case": spec.name,
        "description": spec.description,
        "controller": spec.controller,
        "sensing_profile": spec.sensing_profile,
        "correlation": correlation,
        "configured_latency": latency,
        "acceleration_process_std": acceleration_std,
        "seed": spec.seed,
        "spawn_probability": spec.spawn_probability,
        "driver_shift": spec.driver_shift,
        "base_clearance": spec.base_clearance,
        "uncertainty_multiplier": spec.uncertainty_multiplier,
        "outcome": outcome,
        "success": success,
        "collision": collision,
        "background_collision": background_collision,
        "timeout": timeout,
        "episode_time": episode_time,
        "waiting_time": waiting_time,
        "commit_time": commit_time,
        "clear_time": clear_time,
        "minimum_predicted_clearance_at_commit": commit_minimum_clearance,
        "required_clearance_at_commit": commit_required_clearance,
        "clearance_slack_at_commit": commit_clearance_slack,
        "maximum_conflict_window_position_std_at_commit": (
            commit_maximum_sigma
        ),
        "minimum_pet": safety.minimum_pet,
        "simultaneous_zone_occupancy": (
            safety.simultaneous_zone_occupancy
        ),
        "ego_hard_braking_events": safety.ego_hard_braking_events,
        "mean_decision_time_ms": mean(decision_times_ms),
        "maximum_decision_time_ms": max(decision_times_ms),
        "pet_definition": PET_DEFINITION,
        "video_path": str(video_path),
        "artifact_role": "post_hoc_illustrative_replay",
    }

RECORDING_SPECS = (
RecordingSpec(
        name="comparison_c2_exact_success",
        description="Controller 2 success before sensing degradation",
        controller=PREDICTIVE,
        sensing_profile="exact",
        seed=10025,
        spawn_probability=0.70,
        driver_shift="fast_heavy_assertive",
        expected_outcome="success",
        expected_waiting_time=1.0,
        expected_required_clearance=1.5,
    ),
RecordingSpec(
        name="comparison_c2_high_noise_collision",
        description="Controller 2 collision after sensing degradation",
        controller=PREDICTIVE,
        sensing_profile="rho0p93_latency1p25",
        seed=10025,
        spawn_probability=0.70,
        driver_shift="fast_heavy_assertive",
        expected_outcome="collision",
        expected_waiting_time=0.0,
        expected_required_clearance=1.5,
    ),
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).resolve().parent / "videos" / "matched_comparison")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for spec in RECORDING_SPECS:
            print(f"{spec.name}: {spec.description}")
        return
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "recording_manifest.csv"
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists: {manifest}; choose another directory or use --overwrite")
    rows = []
    for spec in RECORDING_SPECS:
        print(f"Recording {spec.name}", flush=True)
        rows.append(record_spec(spec, video_directory=output, overwrite=args.overwrite))
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
