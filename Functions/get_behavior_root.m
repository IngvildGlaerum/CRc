function root = get_behavior_root()
%GET_BEHAVIOR_ROOT Return this machine's local path to the behavior-only data root.
%   Used by the scripts in BehaviorOnly/ (Radial Arm Maze, T-maze,
%   habituation). Kept as a separate root from GET_DATA_ROOT since this
%   raw data lives in a different folder than the main Analysis data and
%   is not tracked in git.
%
%   Setup: same local_config.m as GET_DATA_ROOT -- see that function's
%   help for setup steps.

    paths = read_local_config();
    root = paths.behavior;
end
