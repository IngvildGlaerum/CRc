function root = get_data_root()
%GET_DATA_ROOT Return this machine's local path to the Analysis data root.
%   Every analysis script calls this instead of hardcoding a path, so the
%   whole repo can be pointed at a new machine by editing a single file.
%
%   Setup (once per machine/clone):
%     1. Copy local_config_template.m (in the repo root) to local_config.m
%        (same folder).
%     2. Edit the path inside local_config.m to point at your local
%        Analysis folder.
%   local_config.m is gitignored, so it is never overwritten by git and
%   never pushed -- each machine/clone keeps its own value.

    repo_root = fileparts(fileparts(mfilename('fullpath')));  % Functions/ -> repo root
    local_config_path = fullfile(repo_root, 'local_config.m');

    if ~isfile(local_config_path)
        error(['No local_config.m found in the repo root. Copy local_config_template.m ' ...
               'to local_config.m and set your data path there.']);
    end

    root = local_config();
end
