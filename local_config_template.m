function paths = local_config()
%LOCAL_CONFIG Machine-specific data roots.
%   Copy this file to local_config.m (same folder, repo root) and set the
%   paths below.
%
%   paths.analysis -- raw session data, Metadata.xlsx, CompiledData/, and
%   Figures/, matching the structure documented in README.md.
%
%   paths.behavior -- the Radial Arm Maze / T-maze / habituation .mat
%   files used by the scripts in BehaviorOnly/ (kept out of the repo and
%   out of git since they're raw data).
%
%   local_config.m is gitignored, so each machine/clone keeps its own
%   values and it never gets overwritten or pushed.

    paths.analysis = 'REPLACE_WITH_YOUR_ANALYSIS_PATH';
    paths.behavior = 'REPLACE_WITH_YOUR_BEHAVIOR_DATA_PATH';
end
