# Robust Intersection Decision-Making Under Noisy Vehicle Observations

Experiment and recording code for the CS5478 report. The simulation compares:

1. **C1 — fixed temporal gap:** commit when the estimated arrival time exceeds a 5 s critical gap.
2. **C2 — fixed predictive clearance:** check predicted vehicle footprints with a fixed 1.5 m clearance.
3. **C3 — uncertainty-aware predictive clearance:** add a buffer proportional to propagated position uncertainty.

The repository contains the final fixed-controller evaluation, the parameter searches described in the report, and the video recorders. It requires no API keys or external dataset. Traffic and observations are generated locally.

## Installation

Use Python 3.11. The dependency versions match the environment used for the report.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Experiments run on CPUs. Set `--workers` to the number of processes appropriate for your machine. Run commands from this repository's root.

## Reproduce the report experiments

### Fixed-controller evaluation

```bash
python run_baselines.py --workers 8
```

The default matrix evaluates C1 and C2 on two driver populations, three traffic loads, three sensing conditions, and 50 seeds per condition: **900 stochastic episodes per controller**. Six deterministic hazards are also run for each controller, giving **1,812 episodes** in total.

Stochastic episodes start in `APPROACH`; hazard episodes place a stationary ego vehicle at the stop line and start in `WAIT`. The selected controller parameters remain fixed.

Outputs:

- `results/baselines_raw.csv`
- `results/baselines_summary.csv`
- `results/baselines_checkpoint.csv`

Interrupted runs resume missing task IDs from the checkpoint. Completed raw and summary files are protected from replacement; use another `--output-directory` for another run.

For a small execution check, use one seed per cell:

```bash
python run_baselines.py --episodes-per-cell 1 --workers 2 \
  --output-directory results/smoke_baselines
```

This runs 36 stochastic episodes and 12 hazards. It is a diagnostic subset, not the report's complete matrix.

### Controller 3 parameter study

```bash
python run_uncertainty_study.py --workers 8
```

This performs the declared coarse search and one local refinement independently for the Exact, Low, Medium (`medium_q1p5`), and High sensing profiles. Each candidate uses 300 stochastic episodes and six hazards by default. Candidates are ranked by safety outcomes before efficiency.

Outputs:

- `results/controller3_noise_study_raw.csv`
- `results/controller3_noise_study_candidate_summary.csv`
- `results/controller3_noise_study_selected_parameters.csv`
- `results/controller3_noise_study_checkpoint.csv`

The full search is substantially larger than the fixed-controller evaluation. Completed episodes are checkpointed. Use a different `--output-prefix` for an independent run. Checkpoints with different seed counts or study signatures are rejected.

C3 parameters are selected and evaluated on the same study scenarios. This reproduces the report's parameter-selection and sensitivity study. Its sensing profiles differ from those used for the frozen C1/C2 evaluation.

## Reproduce parameter selection

C1 and C2 tuning uses development seeds 0–49, three development traffic loads, and the frozen medium sensing pipeline. These seeds are disjoint from the default fixed-controller evaluation seeds 10000–10049.

```bash
python tune_fixed_gap.py --stage coarse --workers 8
python tune_fixed_gap.py --stage refine --workers 8
python tune_predictive_clearance.py --stage coarse --workers 8
```

C1 searches temporal margins, and C2 searches 13 clearances from 0 to 6 m. Search outputs are written under `results/`; the refinement stage reads the corresponding coarse summary. These commands do not edit frozen controller parameters.

The report selected C2's 1.5 m clearance from the coarse grid. Its optional `--stage refine` mode was not used for the reported result. C3's parameter selection is already included in `run_uncertainty_study.py`.

For short tuning checks, `--episodes 1` can be combined with explicit candidates, such as `--margins 0` or `--clearances 1.5`. Such subsets do not reproduce the complete parameter search.

## Record supplementary videos

Install the optional recording dependencies:

```bash
python -m pip install -r requirements-video.txt
```

The recorders render offscreen. On a machine without a display, set:

```bash
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
```

After the two report experiments have generated their CSVs, inspect and record the evidence selection:

```bash
python record_evidence_videos.py --list
python record_evidence_videos.py
```

The default selection uses up to two stochastic successes, collisions, and timeouts per controller/profile, plus the six hazards. The report's completed results provide **68 clips**. Outcome categories without matching episodes are omitted. `--collection` selects `c1_c2_stochastic`, `c1_c2_hazards`, `c3_stochastic`, or `c3_hazards`; repeated `--case` arguments select individual case IDs.

Videos and their manifests are saved under `videos/experiment_evidence/`. Recording resumes completed clips and verifies replay outcomes, waiting times, episode times, PET, and applicable clearance values against the source rows. Source CSV hashes and task IDs are retained in the manifest.

Custom input CSVs can be supplied with `--baseline-results`, `--study-results`, and `--selected-parameters`. Use `--output-directory` to change the recording destination. `--overwrite` replaces only the selected recordings.

Record the two additional matched C2 examples:

```bash
python record_matched_comparison.py --list
python record_matched_comparison.py
```

These use the same seed 10025, fast-heavy assertive traffic, and spawn probability 0.70. Exact sensing produces a successful crossing; strong degradation produces a collision. Outputs go to `videos/matched_comparison/`. Their outcomes, waiting times, and clearance are checked against the reviewed values.

Recordings are post-hoc illustrations. Use the complete raw and summary CSVs for numerical performance claims.

## File purposes

| File | Purpose |
|---|---|
| README.md | Explain installation, experiment reproduction, recording, and the purpose of each repository file. |
| `run_baselines.py` | Schedule and checkpoint the final C1/C2 evaluation; write raw and aggregate metrics. |
| `run_uncertainty_study.py` | Run C3's four-profile coarse/refined search and select parameters. |
| `tune_fixed_gap.py` | Select C1's temporal margin on development episodes. |
| `tune_predictive_clearance.py` | Select C2's fixed spatial clearance on development episodes. |
| `record_evidence_videos.py` | Select examples from completed experiment CSVs and record validated replays. |
| `record_matched_comparison.py` | Record the two reviewed C2 sensing-comparison clips. |
| `experiment_config.py` | Declare driver populations, sensing profiles, hazard settings, search grids, and output paths. |
| `scenarios.py` | Construct development traffic, shifted evaluation traffic, and deterministic hazards. |
| `experiment_runner.py` | Execute and aggregate the shared development episodes used by C1/C2 tuning. |
| `tuning_common.py` | Share development-seed restrictions and safety-first candidate-selection helpers. |
| `video_rendering.py` | Share recording environments, sensing setup, and diagnostic overlays. |
| `fixed_gap_controller.py` | Implement temporal-gap decisions, the shared state machine, and the straight-through traffic environment. |
| `predictive_fixed_margin_controller.py` | Implement fixed-clearance predictive crossing decisions. |
| `uncertainty_aware_predictive_controller.py` | Implement covariance propagation and adaptive-clearance decisions. |
| `vehicle_observations.py` | Define observation interfaces, vehicle snapshots, correlated noise, and latency. |
| `state_estimation.py` | Estimate lane-aligned states and persistent biases with Kalman filtering. |
| `trajectory_prediction.py` | Predict background-vehicle trajectories at constant velocity. |
| `ego_trajectory_prediction.py` | Roll out candidate ego crossings without modifying the real environment. |
| `fixed_clearance_collision_checker.py` | Check oriented footprints and calculate predicted edge-to-edge clearance. |
| `safety_metrics.py` | Measure ground-truth PET, TTC, occupancy, and braking for evaluation. |
| `requirements.txt` | Pin the experiment runtime's direct dependencies. |
| `requirements-video.txt` | Add the optional video-encoding dependencies. |
| `.gitignore` | Exclude generated outputs, recordings, caches, local environments, and credential files. |

`scenarios.py` provides the shared environment and hazard constructors. Generated results, recordings, caches, and local environments are excluded by `.gitignore`.
