# CRc

MATLAB analysis code for [brief one-line description of the project/paper — add before publishing].

This repository covers everything **downstream of rate-map generation** (`BatchRateMaps.mlx`): KL-divergence-based stability, place-field/spatial-information comparisons, behavioral performance, reward-aligned firing, and population decoding. The raw data processing pipeline (spike sorting output → rate maps) is not included here.

## Contents

Each analysis script is provided both as a MATLAB Live Script (`.mlx`, what was actually run) and as a plain-text `.m` export (for readable diffs in git — `.mlx` files are zip archives under the hood and don't diff meaningfully). **The `.mlx` file is the source of truth**; if you edit a script, re-export the `.m` copy (or regenerate it) so the two don't drift apart.

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
├── CRc.prj                        # MATLAB Project — open this to auto-set the path
├── local_config_template.m        # copy to local_config.m and set your data path (see "Data paths")
├── local_config.m                 # machine-specific, gitignored — not in the repo history
├── Functions/
│   ├── get_data_root.m             # reads local_config.m; every script calls this for its data root
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

## Requirements

- MATLAB (developed/run in R2023a; not tested on older versions)
- **Statistics and Machine Learning Toolbox** — required almost everywhere: `fitlme`/`anova` (linear mixed-effects models, used in most comparison scripts), `fitcecoc`/`templateLinear`/`predict` (the SVM decoder in `run_decoder.m`), plus `ttest2`, `ranksum`, `signrank`, `friedman`, `anovan`/`multcompare`, `confusionchart`, `perfcurve`, `randsample`. Without this toolbox, most of these scripts will error out.
- **Parallel Computing Toolbox** — optional. `Decoding.mlx` runs its 100 bootstrap iterations in a `parfor` loop. Without this toolbox MATLAB automatically falls back to running it serially (no error, just slower) — fine for testing, but expect it to take a while on a full run.
- Everything else used (`readtable`, `boxplot`, `histogram`, `smoothdata`, `normalize`, `rescale`, `inpolygon`, `boundary`, etc.) ships with base MATLAB.

## Setup

1. Clone this repo.
2. Open `CRc.prj` in MATLAB (double-click it, or `openProject('path/to/CRc')`) — this automatically adds the repo root and `Functions/` to your MATLAB path for as long as the project is open, and removes them again when you close it, so it doesn't leak into other projects on your machine.
3. Copy `local_config_template.m` to `local_config.m` (same folder) and set the one path inside it to your local Analysis folder — see "Data paths" below.

## Data paths

Every script gets its data root by calling `get_data_root()` instead of hardcoding a path. That function reads `local_config.m`, a single machine-specific file that is gitignored — so it's never overwritten by a pull and never pushed to GitHub. To set up a new machine or clone:

1. Copy `local_config_template.m` → `local_config.m` (repo root).
2. Edit the one line inside to point at your local Analysis folder, e.g.:
   ```matlab
   root = '/Volumes/quattrocolo/crhip/ingvild/Analysis';
   ```

Everything is expected to live under that single root:

- `<root>/Metadata.xlsx` and `<root>/<animal_ID>/<date>/<session>/data/Processed/*.mat` — raw/per-session data (`postprocessed.mat`, `behavior.mat`, `Position_tuning.mat`, etc.)
- `<root>/CompiledData/` — the compiled `.mat` files scripts load/save between each other (`CompiledSpatialStats.mat`, `KLdistance.mat`, `Decoding.mat`, `All_1D_spatial_maps.mat`, etc.)
- `<root>/Figures/Ingvild/Ephys/...` — where every script saves its figures, via `save_figure(...)`

If a script needs a path other than the root itself, it builds it with `fullfile(get_data_root(), ...)` rather than hardcoding it, so moving to a new machine only ever requires editing `local_config.m`.

## Known issues / before you publish this

- **Upstream `.mat` files aren't included.** `CompileSpatialFeatures.mlx` produces `CompiledSpatialStats.mat`, which several other scripts load — but the pipeline that produces the *inputs* to `CompileSpatialFeatures.mlx` (`postprocessed.mat`, `Position_tuning.mat`, `behavior.mat`, etc.) lives outside this repo. Worth documenting the expected structure/fields of those files somewhere so the repo is usable without the full pipeline.
- **Not everything from the original analysis is in here yet.** `Decoding2.mlx` (earlier, non-bootstrapped decoder), `Process_reward_zones.mlx`, `Process_behavior_only.mlx`, `StabilityQuantification.mlx`, `StabilityQuantificationOccCorrected.mlx`, and `TrialDynamics.mlx` are not currently included in this repo.
