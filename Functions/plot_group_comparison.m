function plot_group_comparison(data, control, experimental, cell_numb, edges, colors, save_path_base,save_path, ylabel_text, x_label_text, save_tag)
% plot_group_comparison
% Plots boxplot, scatter, and histogram comparisons between control and experimental groups
%
% Parameters:
%   data            - Input data matrix (e.g., SI or sparsity)
%   control         - Logical or index array for control columns
%   experimental    - Logical or index array for experimental columns
%   cell_numb       - 1x2 vector with number of neurons in [control, experimental]
%   edges           - Predefine the range for the histogram
%   colors          - Cell array with color definitions {color_control, color_experimental}
%   save_path_base  - Path to save the figures (e.g., 'E:\path\to\figures\')
%   ylabel_text     - Label for y-axis (e.g., 'Spatial Information (bit/spike)')
%   x_label_text    - Label for x-axis in histogram (e.g., same as ylabel_text)
%   save_tag        - String tag used in filename (e.g., 'SI')

% Flatten and clean control data
data_cont = data(:,control);
flat_cont = data_cont(:);
flat_cont = flat_cont(~cellfun(@isempty, flat_cont))';
data_all_cont = cell2mat(flat_cont);

% Flatten and clean experimental data
data_exp = data(:,experimental);
flat_exp = data_exp(:);
flat_exp = flat_exp(~cellfun(@isempty, flat_exp))';
data_all_exp = cell2mat(flat_exp);

% Combine into matrix
% cell_numb_max = max([numel(data_all_cont),numel(data_all_exp)]);
data_all = nan(max(cell_numb),2);
data_all(1:length(data_all_cont),1) = data_all_cont;
data_all(1:length(data_all_exp),2) = data_all_exp;

% Boxplot + Scatter
group_name = {'Control','Experimental'};
figure
boxplot(data_all, group_name)
hold on
scatter(1, data_all(:,1), 'filled', 'jitter','on','jitterAmount',0.08, 'MarkerFaceAlpha', 0.5);
% s.MarkerFaceAlpha = 0.7;
scatter(2, data_all(:,2), 'filled', 'jitter','on','jitterAmount',0.08, 'MarkerFaceAlpha', 0.5);
% s.MarkerEdgeAlpha = 0.7;
ylabel(ylabel_text)
print(gcf, '-dsvg', '-painters', fullfile(save_path,'Paper Prep',[save_tag '_box.svg']));
% save_figure(fullfile(save_path_base, save_tag), [save_tag '_box']);
fullfile(save_path_base,[save_tag '_box.svg'])
% t-test
[h,p] = ttest2(data_all_cont, data_all_exp);
disp(['t-test p-value: ' num2str(p)])
% Histogram
figure
h1 = histogram(data_all_cont, edges, 'FaceAlpha', 0.9, 'FaceColor', colors{1});
hold on
h2 = histogram(data_all_exp, edges, 'FaceAlpha', 0.7, 'FaceColor', colors{2});
ylabel('# Neurons')
xlabel(x_label_text)
save_figure(fullfile(save_path_base, save_tag), [save_tag '_hist']);
end