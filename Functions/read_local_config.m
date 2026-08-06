function paths = read_local_config()
%READ_LOCAL_CONFIG Load this machine's local_config.m and return its paths struct.
%   Shared helper used by GET_DATA_ROOT and GET_BEHAVIOR_ROOT. Not meant
%   to be called directly from analysis scripts.

    repo_root = fileparts(fileparts(mfilename('fullpath')));  % Functions/ -> repo root
    local_config_path = fullfile(repo_root, 'local_config.m');

    if ~isfile(local_config_path)
        error(['No local_config.m found in the repo root. Copy local_config_template.m ' ...
               'to local_config.m and set your data paths there.']);
    end

    paths = local_config();
end
