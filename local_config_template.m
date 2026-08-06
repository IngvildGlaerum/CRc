function root = local_config()
%LOCAL_CONFIG Machine-specific path to the Analysis data root.
%   Copy this file to local_config.m (same folder, repo root) and set the
%   path below to your local Analysis folder. Everything -- raw session
%   data, Metadata.xlsx, CompiledData/, and Figures/ -- is expected to
%   live under this one root, matching the structure documented in
%   README.md.
%
%   local_config.m is gitignored, so each machine/clone keeps its own
%   value and it never gets overwritten or pushed.

    root = 'REPLACE_WITH_YOUR_DATA_PATH';
end
