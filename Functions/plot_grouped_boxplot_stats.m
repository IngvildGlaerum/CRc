function bFig = plot_grouped_boxplot_stats(grouped, yLabel, varargin)
% PLOT_GROUPED_BOXPLOT_STATS Plot grouped boxplots with optional stats annotations
%
% Usage:
%   hFig = plot_grouped_boxplot_stats(grouped, yLabel, 'ParamName', ParamValue, ...)
%
% Inputs:
%   grouped  - Cell array {1,2}, each cell is data matrix (rows=samples, cols=groups)
%   yLabel   - Label for y-axis
%
% Optional Name-Value Pairs:
%   'CtrlColor'       - RGB triplet for Control box color (default [0 0 0.5])
%   'ExpColor'        - RGB triplet for Experimental box color (default [0.5 0 0])
%   'FigurePath'        - String path to save figure (default: no save)
%   'SaveName'        - Filename for saving (default: no save)
%   'ShowSignificance'- Logical to show/hide significance stars (default: true)
%
% Output:
%   hFig - Figure handle

% Parse inputs
p = inputParser;
addRequired(p, 'grouped', @(x) iscell(x) && size(x,2)==2);
addRequired(p, 'yLabel', @(x) ischar(x) || isstring(x));
addParameter(p, 'CtrlColor', [0 0 0.5], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'ExpColor',  [0.5 0 0], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'FigurePath', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'SaveName', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'ShowSignificance', true, @(x) islogical(x) || isnumeric(x));

parse(p, grouped, yLabel, varargin{:});
opts = p.Results;

ctrlColor = opts.CtrlColor;
expColor = opts.ExpColor;

numCols = size(grouped{1,1}, 2);

% Flatten data for boxplot
allData = [];
groupIdx = [];
for col = 1:numCols
    % Control
    allData  = [allData; grouped{1,1}(:,col)];
    groupIdx = [groupIdx; repmat((col-1)*2+1, size(grouped{1,1}(:,col)))];
    % Experimental
    allData  = [allData; grouped{1,2}(:,col)];
    groupIdx = [groupIdx; repmat((col-1)*2+2, size(grouped{1,2}(:,col)))];
end

% Create figure and boxplot
bFig = figure; 
h = boxplot(allData, groupIdx, 'Colors', 'k');
hold on;

% Color each box
hBox = findobj(gca, 'Tag', 'Box');
for i = 1:length(hBox)
    if mod(length(hBox)-i, 2) == 0  % Control
        patch(get(hBox(i), 'XData'), get(hBox(i), 'YData'), ctrlColor, 'FaceAlpha', 0.5);
    else  % Experimental
        patch(get(hBox(i), 'XData'), get(hBox(i), 'YData'), expColor, 'FaceAlpha', 0.5);
    end
end

% Scatter raw data points with jitter
rng(1); % reproducible jitter
for g = unique(groupIdx)'
    xData = g + (rand(sum(groupIdx==g),1)-0.5)*0.2; % jitter
    yData = allData(groupIdx==g);
    if mod(g, 2) == 1  % Control
        scatter(xData, yData, 25, ctrlColor, 'filled', 'MarkerFaceAlpha', 0.6);
    else  % Experimental
        scatter(xData, yData, 25, expColor, 'filled', 'MarkerFaceAlpha', 0.6);
    end
end

% Set x-axis labels (e.g. R1, R2, R3)
xticks(1.5:2:2*numCols-0.5);
xticklabels(arrayfun(@(x) sprintf('R%d', x), 1:numCols, 'UniformOutput', false));
ylabel(yLabel);
box on;

% === STATISTICS & SIGNIFICANCE ANNOTATIONS ===
if opts.ShowSignificance
    % Between-groups Wilcoxon rank-sum
    p_between = nan(1, numCols);
    for col = 1:numCols
        ctrlVals = grouped{1,1}(:,col);
        expVals  = grouped{1,2}(:,col);
        p_between(col) = ranksum(ctrlVals, expVals);
    end

    % Within-groups Friedman test
    p_within_control = friedman(grouped{1,1}, 1, 'off');
    p_within_exp     = friedman(grouped{1,2}, 1, 'off');

    % Post-hoc within Control (signrank)
    pairs_ctrl = {};
    pvals_ctrl = [];
    for c1 = 1:numCols-1
        for c2 = c1+1:numCols
            p = signrank(grouped{1,1}(:,c1), grouped{1,1}(:,c2));
            pairs_ctrl{end+1} = [ (c1-1)*2+1, (c2-1)*2+1 ];
            pvals_ctrl(end+1) = p;
        end
    end

    % Post-hoc within Experimental (signrank)
    pairs_exp = {};
    pvals_exp = [];
    for c1 = 1:numCols-1
        for c2 = c1+1:numCols
            p = signrank(grouped{1,2}(:,c1), grouped{1,2}(:,c2));
            pairs_exp{end+1} = [ (c1-1)*2+2, (c2-1)*2+2 ];
            pvals_exp(end+1) = p;
        end
    end

    % Use sigstar or similar function to add significance stars on plot
    % (Make sure sigstar.m is on your path)
    try
        % Between-group significance stars
        for col = 1:numCols
            if p_between(col) < 0.05
                sigstar({[(col-1)*2+1, (col-1)*2+2]}, p_between(col));
            end
        end

        % Within-group significance stars for Control
        for i = 1:length(pvals_ctrl)
            if pvals_ctrl(i) < 0.05
                sigstar({pairs_ctrl{i}}, pvals_ctrl(i));
            end
        end

        % Within-group significance stars for Experimental
        for i = 1:length(pvals_exp)
            if pvals_exp(i) < 0.05
                sigstar({pairs_exp{i}}, pvals_exp(i));
            end
        end
    catch ME
        warning('sigstar function not found or error in plotting significance:', '%s', ME.message);
    end

    % Display stats in Command Window
    disp('Between-groups p-values:');
    disp(p_between);

    disp('Within-group Friedman p-values:');
    disp(['Control: ', num2str(p_within_control)]);
    disp(['Experimental: ', num2str(p_within_exp)]);
end

% Save figure if requested
if ~isempty(opts.FigurePath) && ~isempty(opts.SaveName)
    saveas(bFig, fullfile(opts.FigurePath, opts.SaveName));
end

end