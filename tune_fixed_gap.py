"""Coarse and optional local-refinement search for fixed-gap on development seeds."""

from __future__ import annotations
from tuning_common import DEVELOPMENT_SEEDS, SAFETY_EPSILON, PROJECT_DIRECTORY, frozen_realism_candidate, _optional_float, safety_exclusion_reasons, is_safety_eligible

import argparse

import csv

from concurrent.futures import ProcessPoolExecutor, as_completed

from decimal import Decimal

from math import inf, isfinite

from pathlib import Path

from typing import Iterable, Mapping

from experiment_config import EVALUATION_RAW_RESULTS_PATH, EVALUATION_SEEDS, EVALUATION_SUMMARY_RESULTS_PATH

from experiment_runner import PilotTask, run_pilot_task, summarize_group, write_csv
from experiment_config import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_SPAWN_PROBABILITIES, FIXED_GAP, FROZEN_CROSSING_TIME

COARSE_MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

FROZEN_BRAKING_SUFFIX = "3p0"

def ranking_key(row: Mapping[str, object]) -> tuple[float, float, float, float]:
    """Rank safety-eligible rows by efficiency and then the smaller margin."""
    completion_time = _optional_float(row["mean_successful_completion_time"])
    return (
        float(row["timeout_rate"]),
        completion_time if completion_time is not None else inf,
        float(row["mean_waiting_time"]),
        float(row["safety_margin"]),
    )

def select_candidate(
    summary_rows: Iterable[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Select the leading eligible equal-load row, or None if none qualify."""
    eligible = [
        row
        for row in summary_rows
        if row["traffic_profile"] == "equal_load_mix"
        and is_safety_eligible(row)
    ]
    return min(eligible, key=ranking_key) if eligible else None

def refinement_margins(
    leader: float,
    coarse_margins: Iterable[float],
) -> tuple[float, ...]:
    """Create the predeclared 0.1-second local grid around a coarse leader."""
    coarse = tuple(sorted(set(float(value) for value in coarse_margins)))
    if not coarse:
        raise ValueError("coarse margin set is empty")
    if leader >= coarse[-1] - SAFETY_EPSILON:
        raise ValueError(
            "coarse leader is the upper search boundary; expand the coarse "
            "range before refinement"
        )
    if leader <= coarse[0] + SAFETY_EPSILON and coarse[0] > 0.0:
        raise ValueError(
            "coarse leader is a positive lower boundary; expand downward "
            "before refinement"
        )

    centre = Decimal(str(leader))
    if abs(leader) <= SAFETY_EPSILON:
        lower = Decimal("0.0")
        upper = Decimal("0.5")
    else:
        lower = max(Decimal("0.0"), centre - Decimal("0.4"))
        upper = centre + Decimal("0.4")
    step = Decimal("0.1")
    count = int((upper - lower) / step)
    return tuple(float(lower + step * index) for index in range(count + 1))

def build_tasks(
    *,
    margins: Iterable[float],
    seed: int,
    episodes: int,
    duration: float,
    initial_vehicle_count: int,
) -> list[PilotTask]:
    """Build paired development tasks for every margin and traffic load."""
    candidate = frozen_realism_candidate()
    return [
        PilotTask(
            candidate=candidate,
            controller=FIXED_GAP,
            seed=episode_seed,
            duration=duration,
            initial_vehicle_count=initial_vehicle_count,
            spawn_probability=spawn_probability,
            safety_margin=margin,
        )
        for margin in margins
        for spawn_probability in DEFAULT_SPAWN_PROBABILITIES
        for episode_seed in range(seed, seed + episodes)
    ]

def _decorate_raw_row(
    row: dict[str, object],
    *,
    stage: str,
) -> dict[str, object]:
    margin = float(row["safety_margin"])
    row.update(
        {
            "optimization_stage": stage,
            "crossing_time": FROZEN_CROSSING_TIME,
            "critical_gap": FROZEN_CROSSING_TIME + margin,
        }
    )
    return row

def run_tasks(
    tasks: list[PilotTask],
    *,
    stage: str,
    workers: int,
    progress_every: int,
) -> list[dict[str, object]]:
    """Run independent development episodes and print regular progress."""
    rows: list[dict[str, object]] = []
    total = len(tasks)
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            rows.append(_decorate_raw_row(run_pilot_task(task), stage=stage))
            if completed % progress_every == 0 or completed == total:
                print(f"Completed {completed}/{total} episodes", flush=True)
        return rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_pilot_task, task): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(
                _decorate_raw_row(future.result(), stage=stage)
            )
            if completed % progress_every == 0 or completed == total:
                print(f"Completed {completed}/{total} episodes", flush=True)
    return rows

def summarize_search(
    raw_rows: list[dict[str, object]],
    *,
    margins: Iterable[float],
    stage: str,
) -> list[dict[str, object]]:
    """Build per-load and equally weighted summaries for every margin."""
    summaries: list[dict[str, object]] = []
    for margin in margins:
        margin_rows = [
            row
            for row in raw_rows
            if abs(float(row["safety_margin"]) - margin) <= SAFETY_EPSILON
        ]
        for spawn_probability in DEFAULT_SPAWN_PROBABILITIES:
            profile_rows = [
                row
                for row in margin_rows
                if abs(float(row["spawn_probability"]) - spawn_probability)
                <= SAFETY_EPSILON
            ]
            summaries.append(
                summarize_group(
                    profile_rows,
                    traffic_profile=f"spawn_{spawn_probability:.1f}",
                    spawn_probability=spawn_probability,
                )
            )
        summaries.append(
            summarize_group(
                margin_rows,
                traffic_profile="equal_load_mix",
                spawn_probability="mixed",
            )
        )

    for row in summaries:
        margin = float(row["safety_margin"])
        reasons = safety_exclusion_reasons(row)
        row.update(
            {
                "optimization_stage": stage,
                "crossing_time": FROZEN_CROSSING_TIME,
                "critical_gap": FROZEN_CROSSING_TIME + margin,
                "safety_eligible": not reasons,
                "safety_exclusion_reason": "; ".join(reasons),
            }
        )
    return summaries

def _format_optional(value: object) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{parsed:.2f}"

def print_summary(rows: list[dict[str, object]], *, stage: str) -> None:
    """Print outcome, surrogate-safety, and braking diagnostics."""
    print(f"\nFixed-gap {stage} search", flush=True)
    print(
        "g_s  profile       success collision timeout sim_zone min_PET "
        "min_TTC commit_TTC brakes/ep weak_brakes/ep eligible",
        flush=True,
    )
    for row in rows:
        print(
            f"{float(row['safety_margin']):3.1f}  "
            f"{str(row['traffic_profile']):12s} "
            f"{float(row['success_rate']):7.1%} "
            f"{float(row['collision_rate']):9.1%} "
            f"{float(row['timeout_rate']):7.1%} "
            f"{float(row['simultaneous_zone_occupancy_rate']):8.1%} "
            f"{_format_optional(row['minimum_pet']):>7s} "
            f"{_format_optional(row['minimum_actual_ttc']):>7s} "
            f"{_format_optional(row['minimum_actual_ttc_during_commit']):>10s} "
            f"{float(row[f'mean_ego_hard_braking_events_{FROZEN_BRAKING_SUFFIX}']):9.3f} "
            f"{float(row[f'mean_ego_hard_braking_actual_ttc_gt_3s_or_none_{FROZEN_BRAKING_SUFFIX}']):14.3f} "
            f"{str(bool(row['safety_eligible'])):>8s}",
            flush=True,
        )

def print_ranking(rows: list[dict[str, object]], *, stage: str) -> float | None:
    """Print the predeclared safety-first ranking and return its leader."""
    equal_rows = [
        row for row in rows if row["traffic_profile"] == "equal_load_mix"
    ]
    eligible = [row for row in equal_rows if is_safety_eligible(row)]
    excluded = [row for row in equal_rows if not is_safety_eligible(row)]

    print("\nSafety gate", flush=True)
    print(
        "A candidate must have zero observed ego collisions, zero "
        "simultaneous conflict-zone occupancy, and no PET=0 event.",
        flush=True,
    )
    for row in excluded:
        print(
            f"  EXCLUDED g={float(row['safety_margin']):.1f} s: "
            + ", ".join(safety_exclusion_reasons(row)),
            flush=True,
        )
    if not eligible:
        print(
            "No candidate in the tested range passed the safety gate.",
            flush=True,
        )
        return None

    ranked = sorted(eligible, key=ranking_key)
    print(
        "\nEligible ranking: timeout, successful completion time, waiting "
        "time, then smaller g",
        flush=True,
    )
    print(
        "rank  g_s  success collision timeout success_time mean_wait "
        "min_PET sim_zone",
        flush=True,
    )
    for index, row in enumerate(ranked, start=1):
        print(
            f"{index:4d} {float(row['safety_margin']):4.1f} "
            f"{float(row['success_rate']):7.1%} "
            f"{float(row['collision_rate']):9.1%} "
            f"{float(row['timeout_rate']):7.1%} "
            f"{_format_optional(row['mean_successful_completion_time']):>12s} "
            f"{float(row['mean_waiting_time']):9.2f} "
            f"{_format_optional(row['minimum_pet']):>7s} "
            f"{float(row['simultaneous_zone_occupancy_rate']):8.1%}",
            flush=True,
        )
    leader = float(ranked[0]["safety_margin"])
    print(
        f"\nLeading {stage} candidate: g={leader:.1f} s. "
        "This script does not freeze it automatically.",
        flush=True,
    )
    return leader

def load_coarse_summary(path: Path) -> list[dict[str, object]]:
    """Read only the development coarse summary needed for refinement."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("coarse summary is empty")
    required = {
        "safety_margin",
        "traffic_profile",
        "collision_rate",
        "timeout_rate",
        "mean_successful_completion_time",
        "mean_waiting_time",
        "minimum_pet",
        "simultaneous_zone_occupancy_rate",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            "coarse summary is missing columns: " + ", ".join(sorted(missing))
        )
    return rows

def _default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    duration_label = f"{args.duration:g}".replace(".", "p")
    stem = (
        f"fixed_gap_realism_{args.stage}_seed{args.seed}_"
        f"{args.episodes}eps_duration{duration_label}"
    )
    raw = args.raw_output or DEFAULT_OUTPUT_DIRECTORY / f"{stem}_raw.csv"
    summary = (
        args.summary_output
        or DEFAULT_OUTPUT_DIRECTORY / f"{stem}_summary.csv"
    )
    return raw.expanduser().resolve(), summary.expanduser().resolve()

def _reject_protected_outputs(paths: Iterable[Path]) -> None:
    protected = {
        (PROJECT_DIRECTORY / EVALUATION_RAW_RESULTS_PATH).resolve(),
        (PROJECT_DIRECTORY / EVALUATION_SUMMARY_RESULTS_PATH).resolve(),
    }
    if any(path in protected for path in paths):
        raise ValueError("fixed-gap tuning cannot write a held-out result path")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise Controller 1 g on frozen development realism.",
    )
    parser.add_argument(
        "--stage",
        choices=("coarse", "refine"),
        default="coarse",
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        help="Explicit margins; otherwise use the stage default.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--initial-vehicles", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--coarse-summary",
        type=Path,
        help="Development coarse summary used to derive refinement margins.",
    )
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing fixed-gap tuning development output files.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    seeds = set(range(args.seed, args.seed + args.episodes))
    if not seeds.issubset(DEVELOPMENT_SEEDS):
        parser.error("fixed-gap tuning seeds must stay within development seeds 0-49")
    if seeds.intersection(EVALUATION_SEEDS):
        parser.error("fixed-gap tuning cannot use held-out seeds")
    if not isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.initial_vehicles < 0:
        parser.error("--initial-vehicles must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.margins is not None:
        if any(not isfinite(value) or value < 0 for value in args.margins):
            parser.error("all margins must be finite and non-negative")
        if len(set(args.margins)) != len(args.margins):
            parser.error("--margins must not contain duplicates")
    return args

def _coarse_summary_path(args: argparse.Namespace) -> Path:
    if args.coarse_summary is not None:
        return args.coarse_summary.expanduser().resolve()
    duration_label = f"{args.duration:g}".replace(".", "p")
    return (
        DEFAULT_OUTPUT_DIRECTORY
        / (
            f"fixed_gap_realism_coarse_seed{args.seed}_"
            f"{args.episodes}eps_duration{duration_label}_summary.csv"
        )
    ).resolve()

def margins_for_stage(args: argparse.Namespace) -> tuple[float, ...]:
    """Resolve explicit, coarse-default, or summary-derived margins."""
    if args.margins is not None:
        return tuple(sorted(args.margins))
    if args.stage == "coarse":
        return COARSE_MARGINS

    coarse_path = _coarse_summary_path(args)
    if not coarse_path.is_file():
        raise FileNotFoundError(
            f"coarse summary not found: {coarse_path}; run the coarse stage first"
        )
    coarse_rows = load_coarse_summary(coarse_path)
    leader = select_candidate(coarse_rows)
    if leader is None:
        raise ValueError(
            "coarse summary has no safety-eligible leader; expand the coarse "
            "search before refinement"
        )
    coarse_margins = {
        float(row["safety_margin"])
        for row in coarse_rows
        if row["traffic_profile"] == "equal_load_mix"
    }
    return refinement_margins(
        float(leader["safety_margin"]),
        coarse_margins,
    )

def main() -> None:
    args = parse_args()
    margins = margins_for_stage(args)
    raw_path, summary_path = _default_output_paths(args)
    _reject_protected_outputs((raw_path, summary_path))
    if raw_path == summary_path:
        raise ValueError("raw and summary output paths must differ")
    existing = [path for path in (raw_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "fixed-gap tuning output already exists; use --overwrite only for an "
            "intentional development rerun: "
            + ", ".join(str(path) for path in existing)
        )

    tasks = build_tasks(
        margins=margins,
        seed=args.seed,
        episodes=args.episodes,
        duration=args.duration,
        initial_vehicle_count=args.initial_vehicles,
    )
    print(
        f"Running fixed-gap tuning {args.stage} search: {len(margins)} margins x "
        f"{len(DEFAULT_SPAWN_PROBABILITIES)} traffic loads x "
        f"{args.episodes} paired development seeds = {len(tasks)} episodes",
        flush=True,
    )
    print(
        "Frozen realism: medium_q1p5, equal slow/normal/fast driver mixture, "
        "1 Hz controller, 40 s default duration",
        flush=True,
    )
    print(f"Margins: {', '.join(f'{value:.1f}' for value in margins)} s")
    print(f"CPU workers: {args.workers}; GPU: not used", flush=True)
    if args.episodes < 50:
        print(
            "NOTE: fewer than 50 seeds is diagnostic only and cannot freeze g.",
            flush=True,
        )
    print("Evaluation seeds and result paths are excluded from tuning.\n", flush=True)

    raw_rows = run_tasks(
        tasks,
        stage=args.stage,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    raw_rows.sort(
        key=lambda row: (
            float(row["safety_margin"]),
            float(row["spawn_probability"]),
            int(row["seed"]),
        )
    )
    summaries = summarize_search(
        raw_rows,
        margins=margins,
        stage=args.stage,
    )
    saved_raw = write_csv(raw_path, raw_rows)
    saved_summary = write_csv(summary_path, summaries)

    print_summary(summaries, stage=args.stage)
    leader = print_ranking(summaries, stage=args.stage)
    if leader is not None and args.stage == "coarse":
        if abs(leader - max(margins)) <= SAFETY_EPSILON:
            print(
                "The leading coarse candidate is at the upper boundary "
                "of the tested range.",
                flush=True,
            )
        else:
            print(
                "Coarse search complete. Optional --stage refine uses the "
                "same seed and episode settings.",
                flush=True,
            )
    elif leader is not None:
        print(
            "Refined temporal-margin search complete.",
            flush=True,
        )

    print(f"\nSaved raw development results: {saved_raw}", flush=True)
    print(f"Saved aggregate summary:       {saved_summary}", flush=True)

if __name__ == "__main__":
    main()
