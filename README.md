# CRc

MATLAB and Python analysis code for [brief one-line description of the project/paper — add before publishing].

The MATLAB code (repo root, `Functions/`, `BehaviorOnly/`) covers everything **downstream of rate-map generation** (`BatchRateMaps.mlx`): KL-divergence-based stability, place-field/spatial-information comparisons, behavioral performance, reward-aligned firing, and population decoding. The raw data processing pipeline (spike sorting output → rate maps) is not included here.

The Python code (`RLModel/`) is separate and self-contained: it quantifies per-trial Radial Arm Maze behaviour and fits a tabular Q-learning model to it. Nothing in `RLModel/` calls the MATLAB code or vice versa.

## Contents

Each MATLAB analysis script is provided both as a MATLAB Live Script (`.mlx`, what was actually run) and as a plain-text `.m` export (for readable diffs in git — `.mlx` files are zip archives under the hood and don't diff meaningfully). **The `.mlx` file is the source of truth**; if you edit a script, re-export the `.m` copy (or regenerate it) so the two don't drift apart.

```
CRc/
├── README.md
├── ComparePerformance.mlx / .m
├── ComparePlaceFields.mlx / .m
├── CompareSI_FR.mlx / .m
├── CompileSpatialFeatures.mlx / .m
├── Decoding.mlx / .m              # bootstrapped decoder — used for published results
├── DecodingResults.mlx / .m
├── KL_distance.mlx / .m
├── KL_distance_plots.mlx / .m      # day-pooled cumulative KL plots + per-day LMM stats
├── KL_distance_reward_cells.mlx / .m
├── RewardRasters.mlx / .m
├── BehaviorOnly/                  # Radial Arm Maze / T-maze / habituation — separate data root, see "Data paths"
│   ├── HabituationRAM.mlx
│   ├── Radial_Arm.mlx
│   └── Tmaze.mlx                  # .mat data files are gitignored, not stored in this repo
├── RLModel/                       # Python — Q-learning model of Radial Arm Maze behaviour
│   ├── requirements.txt
│   ├── utility_functions/
│   │   └── utility_functions.py    # shared constants (zone colours) + session loaders
│   └── scripts/
│       ├── real_data_basic_features.py                      # (1) behaviour → pop_basic_data.pkl
│       ├── q_learning_optimisation_and_plotting.py          # (2) model definition + per-animal DE fits
│       ├── q_learning_eps_beta_loss_grid_search_w_noise.py  # (3) (ε, β, q_noise) grid search + AUL
│       └── rl_functions.py         # plotting params, entry extraction, legacy arm-to-arm model
├── CRc.prj                        # MATLAB Project — open this to auto-set the path
├── local_config_template.m        # copy to local_config.m and set your data paths (see "Data paths")
├── local_config.m                 # machine-specific, gitignored — not in the repo history
├── Functions/
│   ├── get_data_root.m             # reads local_config.m; main Analysis root
│   ├── get_behavior_root.m         # reads local_config.m; BehaviorOnly data root
│   ├── read_local_config.m         # shared helper behind the two functions above
│   ├── SE.m
│   ├── save_figure.m
│   ├── linspecer.m
│   ├── run_decoder.m
│   ├── plot_group_comparison.m
│   ├── plot_si_vs_fr.m
│   ├── plot_grouped_boxplot_stats.m
│   ├── sigstar.m
│   ├── shadedErrorBar.m           # vendored from raacampbell/shadedErrorBar — see file header
│   ├── centroid.m                 # vendored from BNT/externals/geom2d — see file header
│   └── average_matrix.m           # found in Ingvild's behavioral testing folder
└── resources/project/             # MATLAB Project metadata (created by CRc.prj)
```

The `RLModel/` nesting is not cosmetic: the scripts do `sys.path.append("..")` and resolve their data paths as `../../...`, both relative to the working directory. Run them from `RLModel/scripts/` or the imports and paths break.

## Requirements

**MATLAB** (developed/run in R2023a; not tested on older versions)

- **Statistics and Machine Learning Toolbox** — required almost everywhere: `fitlme`/`anova` (linear mixed-effects models, used in most comparison scripts), `fitcecoc`/`templateLinear`/`predict` (the SVM decoder in `run_decoder.m`), plus `ttest2`, `ranksum`, `signrank`, `friedman`, `anovan`/`multcompare`, `confusionchart`, `perfcurve`, `randsample`. Without this toolbox, most of these scripts will error out.
- **Parallel Computing Toolbox** — optional. `Decoding.mlx` runs its 100 bootstrap iterations in a `parfor` loop. Without this toolbox MATLAB automatically falls back to running it serially (no error, just slower) — fine for testing, but expect it to take a while on a full run.
- Everything else used (`readtable`, `boxplot`, `histogram`, `smoothdata`, `normalize`, `rescale`, `inpolygon`, `boundary`, etc.) ships with base MATLAB.

**Python** (3.11+)

- Packages are pinned in `RLModel/requirements.txt` (`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `numba`, `tqdm`).
- **`tkinter`** — `rl_functions.set_plotting_params()` reads screen DPI through Tk and is called at *import* time, so these scripts need a display. On Debian/Ubuntu: `sudo apt install python3-tk`. They will not import headless (see "Known issues").
- 6+ CPU cores recommended. `n_cpus = 6` is hardcoded near the top of each `__main__` block. The grid search is the expensive one — 11,600 simulations of 500 agents × 64 trials.

## Setup

**MATLAB**

1. Clone this repo.
2. Open `CRc.prj` in MATLAB (double-click it, or `openProject('path/to/CRc')`) — this automatically adds the repo root and `Functions/` to your MATLAB path for as long as the project is open, and removes them again when you close it, so it doesn't leak into other projects on your machine.
3. Copy `local_config_template.m` to `local_config.m` (same folder) and set the two paths inside it — see "Data paths" below.

**Python**

```bash
cd RLModel && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd scripts
python real_data_basic_features.py                     # (1) writes pop_basic_data.pkl
python q_learning_optimisation_and_plotting.py         # (2) per-animal (eps, beta) fits
python q_learning_eps_beta_loss_grid_search_w_noise.py # (3) grid search + AUL
```

Run in that order — (2) and (3) both read the output of (1). There are no command-line arguments; every switch (`fit`, `plot`, `ndays`, `q_noise`, `make_aul_plots`, `n_cpus`, …) is set by editing the `if __name__ == "__main__":` block at the bottom of each script.

## The RL model

An 8-arm radial maze, one state per arm plus a centre state (`0`), and 9 actions (eight arm directions plus "stay"). Arms are numbered as in the behavioural data: treats in arms 3, 4 and 7; start arm 2 on days 1–8 and arm 5 on days 9–14, which the model reproduces via `start_arm_arr` indexed by `episode // 64`. Q is indexed by **(previous state, current state, action)** — a one-step state history, hence `1hist` in the output paths — with an independent table per simulated agent (500 by default). Errors per trial are (arm entries) − 3, matching the definition in `real_data_basic_features.py`. `q_noise` is the s.d. of Gaussian noise added to Q **once per simulation step per active agent**, so its effect accumulates with trial length rather than being fixed per trial.

> `epsilon` is the **learning rate**, not an ε-greedy exploration probability. Action selection is softmax with inverse temperature `beta`. State this explicitly in the methods — the conventional reading of "epsilon" is wrong for this model.

Both fitting scripts minimise the same loss: the mean absolute difference between simulated and observed mean errors per day (8 trials/day), averaged over days. The DE fits use 14 days; the grid search uses 8. Genotype comparisons are two-sided Mann–Whitney *U* on n = 4 vs n = 4 animals, where the smallest attainable *p* is 0.029 — report exact *p* values rather than the significance stars the plotting helper draws.

## Data paths

### MATLAB

There are two separate data roots, since the main Ephys analysis data and the behavior-only data live in different folders:

- **`get_data_root()`** — the main Analysis root, used by every script in the repo root.
- **`get_behavior_root()`** — the Radial Arm Maze / T-maze / habituation data root, used by the scripts in `BehaviorOnly/`. Kept separate deliberately: these are large raw `.mat` files (tens of MB) that stay off GitHub entirely (gitignored — see `BehaviorOnly/*.mat` in `.gitignore`), unlike the main Analysis data which at least has its folder structure documented here.

Both functions read from the same `local_config.m`, a single machine-specific file that is gitignored — never overwritten by a pull, never pushed to GitHub. To set up a new machine or clone:

1. Copy `local_config_template.m` → `local_config.m` (repo root).
2. Edit the two paths inside, e.g.:
   ```matlab
   paths.analysis = '/Volumes/quattrocolo/crhip/ingvild/Analysis';
   paths.behavior = '/Volumes/quattrocolo/crhip/ingvild/Radial Arm Maze';
   ```

Everything under `get_data_root()` is expected to follow this structure:

- `<root>/Metadata.xlsx` and `<root>/<animal_ID>/<date>/<session>/data/Processed/*.mat` — raw/per-session data (`postprocessed.mat`, `behavior.mat`, `Position_tuning.mat`, etc.)
- `<root>/CompiledData/` — the compiled `.mat` files scripts load/save between each other (`CompiledSpatialStats.mat`, `KLdistance.mat`, `Decoding.mat`, `All_1D_spatial_maps.mat`, etc.)
- `<root>/Figures/Ingvild/Ephys/...` — where every script saves its figures, via `save_figure(...)`

`get_behavior_root()` is flatter — `HabituationData.mat`, `DataRAM.mat`, `PostHoc.mat`, and `DataTmaze.mat` sit directly under that one folder.

If a script needs a path other than the root itself, it builds it with `fullfile(get_data_root(), ...)` or `fullfile(get_behavior_root(), ...)` rather than hardcoding it, so moving to a new machine only ever requires editing `local_config.m`.

### RLModel

The Python scripts do not use `local_config.m`. Their paths are hardcoded and resolve against the working directory, so with `RLModel/scripts/` as the working directory they land at the repo root:

| Path in the scripts | Resolves to | Contents |
|---|---|---|
| `../../Data` | `CRc/Data/` | per-mouse position/zone pickles, `{genotype}_{mouse_id}` — **input**, gitignored |
| `../../Output/RL/real_behaviour_spatial` | `CRc/Output/RL/real_behaviour_spatial/` | `pop_basic_data.pkl`, `pop_entry_data.pkl`, behaviour figures |
| `../../Output/RL/theoretical_spatial_1hist` | `CRc/Output/RL/theoretical_spatial_1hist/` | fits, loss surfaces, model figures |

`Data/` and `Output/` are gitignored — nothing under them is in the repo. `pop_basic_data.pkl` is the interface between the behavioural and modelling halves: one row per trial (896 rows = 8 mice × 14 days × 8 trials), with columns `animal_id`, `genotype` (1 = control/cre−, 2 = experimental/cre+), `experiment_type` (actually the *trial block*, 1 or 2), `day`, `trial_idx`, `num_errors`, `t_trial`, `err_r1`/`err_r2`/`err_r3`, `performance`. The fitting scripts read `num_errors` in file order and reshape it into blocks of 8, which is safe only because `real_data_basic_features.py` writes rows sorted by genotype → animal → day → trial with exactly 8 trials per day.

## Known issues / before you publish this

- **Upstream data files aren't included.** `CompileSpatialFeatures.mlx` produces `CompiledSpatialStats.mat`, which several other scripts load — but the pipeline that produces the *inputs* to `CompileSpatialFeatures.mlx` (`postprocessed.mat`, `Position_tuning.mat`, `behavior.mat`, etc.) lives outside this repo. The same applies to `RLModel/`: the per-mouse pickles under `Data/` that `real_data_basic_features.py` reads are produced upstream. Worth documenting the expected structure/fields of those files somewhere so the repo is usable without the full pipeline.
- **`penalize_return` appears to have no effect.** In `sim_q_learning_agents_vectorized` it writes −10 only to `(prev_state, state)` pairs where both are arms *and* the arms differ. Arms are reachable only via the centre, so that pair never occurs — an exhaustive walk of the transition graph finds zero overlap with the reachable pairs, for any `BINS_PER_ARM`. If the intent was "penalise re-entering the arm you just left", the inner loop needs to include `state = 0`. Check this before the methods text claims the penalty does anything; it is also encoded in the output directory names (`_penalizedReturn`).
- **The differential evolution search is unseeded.** Per-parameter simulation RNGs are seeded deterministically (`_seed_from_params`, `seed_for_grid`), but `differential_evolution` is called without `seed=`, so its population draws come from the global NumPy state and the published (ε, β) cannot be regenerated exactly. Pass a fixed seed — recent SciPy releases are migrating this keyword to `rng=`.
- **`rl_functions.py` mixes two models.** Everything except `set_plotting_params`, `get_screen_dpi` and the `get_*entry_df` helpers belongs to an earlier arm-to-arm Q-learning model that used the global NumPy RNG and did not produce the published results. Worth splitting or deleting so a reader can tell which model the paper describes. The `tkinter` call in `set_plotting_params` is also what makes the scripts fail headless — a try/except fallback to fixed rcParams would fix that.
- **No licence or DOI.** *Nature Neuroscience*'s code-availability policy expects an explicit licence; add a `LICENSE` file (MIT or BSD-3-Clause), keep the headers of the vendored files (`shadedErrorBar.m`, `centroid.m`) intact, and archive a release on Zenodo so the reviewed version stays retrievable.
