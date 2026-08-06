function root = get_data_root()
%GET_DATA_ROOT Return this machine's local path to the Analysis data root.
%   Every analysis script calls this instead of hardcoding a path, so the
%   whole repo can be pointed at a new machine by editing a single file.
%   See also GET_BEHAVIOR_ROOT.
%
%   Setup (once per machine/clone):
%     1. Copy local_config_template.m (in the repo root) to local_config.m
%        (same folder).
%     2. Edit the paths inside local_config.m to point at your local
%        folders.
%   local_config.m is gitignored, so it is never overwritten by git and
%   never pushed -- each machine/clone keeps its own value.

    paths = read_local_config();
    root = paths.analysis;
end
