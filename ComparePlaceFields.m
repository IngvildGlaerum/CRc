%% Compare Place Cells
% Converted from MATLAB Live Script (.mlx) to plain .m for version control.
% Source of truth is the .mlx file; regenerate this export if the .mlx changes.

clear; close all; clc;
animalID = [65622,63383,67330,67456,66922,65588,67324,67141];
animals = length(animalID);

main_dir = get_data_root();
meta_table = readtable(fullfile(main_dir,'Metadata.xlsx'),"Sheet",'Data ID');
load_path = fullfile(main_dir,'CompiledData');
figure_path = fullfile(main_dir,'Figures/Ingvild/Ephys/PlaceFields');

%% Define Variables

% % check what is relevant here
gen = [1,2]; %(1 = control, 2 = exp for indexing)
colors = {'#284B5E','#BC554C'};
session = 6;
zones = 9;
control = 1:4;
experimental = 5:8;
threshold = 20;

% Load data
load(fullfile(load_path,"CompiledSpatialStats.mat"));
group_id = {'Control','Experimental'};

%% Plot FR over SI

fr_cont = [];
fr_exp = [];
si_cont_trials = [];
si_exp_trials = [];

for a = 1:animals

    for s = 1:session

        firing_rate = {};
        info = [];

        firing_rate = FR{s,a};
        if isempty(firing_rate)
            continue
        end
        if ismember(a,control)
            f = cellfun(@mean,firing_rate);
            fr_cont = [fr_cont,f];
            info = si_trials{s,a}';
            si_cont_trials = [si_cont_trials,info];
        else
            f = cellfun(@mean,firing_rate);
            fr_exp = [fr_exp,f];
            info = si_trials{s,a}';
            si_exp_trials = [si_exp_trials,info];
        end

    end
end

% Plot scatter per trial

for s = 1:session
    figure('Visible','off')
    scatter(fr_cont(s,:),si_cont_trials(s,:),'Filled','MarkerFaceColor',colors{1},'MarkerEdgeColor',colors{1})
    hold on
    scatter(fr_exp(s,:),si_exp_trials(s,:),'Filled','MarkerFaceColor',colors{2},'MarkerEdgeColor',colors{2})
    ylabel('Spatial Information')
    xlabel('Firing Rate (Hz)')
    title(sprintf('Day %d',s))
    close;
end

firing_rate_contr = mean(fr_cont,'omitnan');
cont_idx = firing_rate_contr<= threshold;
firing_rate_exp = mean(fr_exp,'omitnan');
exp_idx = firing_rate_exp<= threshold;

si_cont = mean(si_cont_trials);
si_exp = mean(si_exp_trials);

figure('Visible','off')
scatter(firing_rate_contr(cont_idx),si_cont(cont_idx),'Filled','MarkerFaceColor',colors{1},'MarkerEdgeColor','k')
hold on
p = scatter(firing_rate_exp(exp_idx),si_exp(exp_idx),'Filled','MarkerFaceColor',colors{2},'MarkerEdgeColor','k');
p.MarkerFaceAlpha = 0.8;
ylabel('Spatial Information')
xlabel('Firing Rate (Hz)')
figure_handle = 'SIvsFR';
save_figure(fullfile(main_dir,'Figures/Ingvild/Ephys/FR/Comparison'),figure_handle)

%% Plot Field locations

control = find(genotype_idx == 1);
experimental = find(genotype_idx == 2);
edges = linspace(-0.5,8.5,10);


field_zone_cont = animals_peak_zone(:,control);
missing_contr = cellfun(@isempty, field_zone_cont);
if sum(missing_contr(:))>= 1
    convert_mat = field_zone_cont(:);
    fz_cont = convert_mat(~missing_contr(:));
else
    fz_cont = field_zone_cont(:);
end

fz_control = cell2mat(cellfun(@(m) m(:,1), fz_cont', 'UniformOutput', false)');
fields_control = fz_control(~isnan(fz_control));

% Map each integer into the center of a 9-bin circle
angles = (fields_control + 0.5) / 9 * 2*pi;   % place each value at its bin-center (radians)
edges  = linspace(0, 2*pi, 10);     % 9 bins -> 10 edges

% Get counts for each bin
[counts, ~] = histcounts(angles, edges);
total = sum(counts);
perc  = 100 * counts / max(1,total);  % avoid divide-by-zero

% Plot the rose
figure;
p = polarhistogram(angles, 'BinEdges', edges);   % use explicit edges
p.FaceColor   = 'blue';        % one color for all slices
p.EdgeColor   = 'black';       % <-- black outline
p.EdgeAlpha   = 1;             % fully opaque edges
p.LineWidth   = 1;             % make outline visible
p.DisplayStyle = 'bar';        % bar style (classic rose)
title('Distribution of values (Rose plot)')

% Put ticks/labels (0..8) at the center of each bin
centers = (edges(1:end-1) + edges(2:end)) / 2;   % radians
ax = gca;
ax.ThetaTick = rad2deg(centers);                 % ThetaTick expects degrees
ax.ThetaTickLabel = arrayfun(@num2str, 0:8, 'UniformOutput', false);


% Histogram
figure
h = histogram(fields_control,edges,'FaceAlpha',0.9);
title('Control Animals')
ylabel('# Neurons')
ylim([0 140])
yticks(0:20:140)
xlabel('Zone')
h.FaceColor = colors{1};
figure_handle = 'All Sessions Control Animals';
save_figure(fullfile(figure_path,'ZoneLocation'),figure_handle)

field_zone_exp = animals_peak_zone(:,experimental);
missing_exp = cellfun(@isempty, field_zone_exp);
if sum(missing_exp(:))>= 1
    convert_mat = field_zone_exp(:);
    fz_exp = convert_mat(~missing_exp(:));
else
    fz_exp = field_zone_exp(:);
end

fz_experimental = cell2mat(cellfun(@(m) m(:,1), fz_exp', 'UniformOutput', false)');
fields_exp = fz_experimental(~isnan(fz_experimental));

% Map each integer into the center of a 9-bin circle
angles = (fields_exp + 0.5) / 9 * 2*pi;   % place each value at its bin-center (radians)
edges  = linspace(0, 2*pi, 10);     % 9 bins -> 10 edges

% Get counts for each bin
[counts, ~] = histcounts(angles, edges);
total = sum(counts);
perc  = 100 * counts / max(1,total);  % avoid divide-by-zero

% Plot the rose
figure;
p = polarhistogram(angles, 'BinEdges', edges);   % use explicit edges
p.FaceColor   = 'blue';        % one color for all slices
p.EdgeColor   = 'black';       % <-- black outline
p.EdgeAlpha   = 1;             % fully opaque edges
p.LineWidth   = 1;             % make outline visible
p.DisplayStyle = 'bar';        % bar style (classic rose)
title('Distribution of values (Rose plot)')

% Put ticks/labels (0..8) at the center of each bin
centers = (edges(1:end-1) + edges(2:end)) / 2;   % radians
ax = gca;
ax.ThetaTick = rad2deg(centers);                 % ThetaTick expects degrees
ax.ThetaTickLabel = arrayfun(@num2str, 0:8, 'UniformOutput', false);


figure
h = histogram(fields_exp,'FaceAlpha',0.9);
ylim([0 140])
yticks(0:20:140)
title('Experimental Animals')
ylabel('# Neurons')
xlabel('Zone')
h.FaceColor = colors{2};
figure_handle = 'All Sessions Experimental Animals';
save_figure(fullfile(figure_path,'ZoneLocation'),figure_handle)



% % Separate into TB1 and TB2
% TB_idx = [1:3;4:6];
% 
% for i = 1:2
%     TB = TB_idx(i,:);
% 
%     % Control animals
%     field_zone_cont = animals_peak_zone(TB',control);
%     missing_contr = cellfun(@isempty, field_zone_cont);
%     if sum(missing_contr(:))>= 1
%         convert_mat = field_zone_cont(:);
%         fz_cont = convert_mat(~missing_contr(:));
%     else
%         fz_cont = field_zone_cont(:);
%     end
% 
%     fz_control = cell2mat(cellfun(@(m) m(:,1), fz_cont', 'UniformOutput', false)');
% 
%     figure('Visible','off')
%     h = histogram(fz_control,edges,'Normalization','pdf','FaceAlpha',0.9);
%     title(sprintf('TB %d', i))
%     ylabel('PDF Field Location')
%     ylim([0 0.25])
%     yticks(0:0.05:0.25)
%     xlabel('Zone')
%     h.FaceColor = colors{1};
%     figure_handle = sprintf('Control Animals TB %d', i);
%     save_figure(fullfile(figure_path,'ZoneLocation'),figure_handle)
% 
%     % Experimental animals
%     field_zone_exp = animals_peak_zone(TB',experimental);
%     missing_exp = cellfun(@isempty, field_zone_exp);
%     if sum(missing_exp(:))>= 1
%         convert_mat = field_zone_exp(:);
%         fz_exp = convert_mat(~missing_exp(:));
%     else
%         fz_exp = field_zone_exp(:);
%     end
% 
%     fz_experimental = cell2mat(cellfun(@(m) m(:,1), fz_exp', 'UniformOutput', false)');
% 
%     figure('Visible','off')
%     h = histogram(fz_experimental,edges,'Normalization','pdf','FaceAlpha',0.9);
%     ylim([0 0.25])
%     yticks(0:0.05:0.25)
%     title(sprintf('TB %d', i))
%     ylabel('PDF Field Location')
%     xlabel('Zone')
%     h.FaceColor = colors{2};
%     figure_handle = sprintf('Experimental Animals TB %d', i);
%     save_figure(fullfile(figure_path,'ZoneLocation'),figure_handle)
% 
% end
% 
% 
% % Look at field location for different days 
% for s = 1:session
%     field_zone_cont = animals_peak_zone(s,control);
%     field_zone_exp = animals_peak_zone(s,experimental);
% 
%     missing_contr = cellfun(@isempty, field_zone_cont);
%     missing_exp = cellfun(@isempty, field_zone_exp);
%     if sum(missing_contr)>= 1
%         field_zone_cont = field_zone_cont(~missing_contr);
%     end
% 
%     if sum(missing_exp)>= 1
%         field_zone_exp = field_zone_exp(~missing_exp);
%     end
% 
%     field_zone_control = cell2mat(cellfun(@(m) m(:,1), field_zone_cont, 'UniformOutput', false)');
%     field_zone_experimental = cell2mat(cellfun(@(m) m(:,1), field_zone_exp, 'UniformOutput', false)');
% 
%     figure('Visible','off')
%     h = histogram(field_zone_control,edges,'Normalization','pdf','FaceAlpha',0.9);
%     title(sprintf('Control Animals Day %d',s))
%     h.FaceColor = colors{1};
%     ylim([0 0.25])
%     yticks(0:0.05:0.25)
%     ylabel('PDF Field Location')
%     xlabel('Zone')
%     figure_handle = sprintf('Control Animals Day %d',s);
%     save_figure(fullfile(figure_path,'ZoneLocation'),figure_handle)
% 
%     figure('Visible','off')
%     h = histogram(field_zone_experimental,edges,'Normalization','pdf','FaceAlpha',0.9);
%     title(sprintf('Experimental Animals Day %d',s))
%     ylim([0 0.25])
%     yticks(0:0.05:0.25)
%     h.FaceColor = colors{2};
%     ylabel('PDF Field Location')
%     xlabel('Zone')
%     figure_handle = sprintf('Experimental Animals Day %d',s);
%     save_figure(fullfile(figure_path,'ZoneLocation'),figure_handle)
% end

%% Plot peak FR

peak_fr_cont = animals_peak(:,control);
convert_mat = peak_fr_cont(:);
missing_contr = cellfun(@isempty, convert_mat);
fr_cont = convert_mat(~missing_contr(:));

fr_control = cell2mat(cellfun(@(m) m(:,1), fr_cont', 'UniformOutput', false)');


peak_fr_exp = animals_peak(:,experimental);

convert_mat = peak_fr_exp(:);
missing_exp = cellfun(@isempty, convert_mat);
fr_exp = convert_mat(~missing_exp(:));

fr_experimental = cell2mat(cellfun(@(m) m(:,1), fr_exp', 'UniformOutput', false)');
cell_numb = [length(fr_control),length(fr_experimental)];

peak_fr_all = nan(max(cell_numb),2);
peak_fr_all(1:length(fr_control),1) = fr_control;
peak_fr_all(1:length(fr_experimental),2) = fr_experimental;

figure
boxplot(peak_fr_all,group_id,'OutlierSize',0.5)
hold on
scatter(1,peak_fr_all(:,1),'filled','jitter','on','jitterAmount',0.08)
scatter(2,peak_fr_all(:,2),'filled','jitter','on','jitterAmount',0.08)
ylabel('Firing Rate')
title('Peak of Place Field')
figure_handle = 'All Sessions Peak FR';
save_figure(fullfile(figure_path,'PlaceField'),figure_handle)

%% Compare Field Number

% run this for each day as well 

field_nb = cell(1,animals);
for a = 1:animals
    for s = 1:session
        field_numb = animals_field_size{s,a};  
        field_nb{s,a} = sum(~isnan(field_numb),2);
    end
end

fields_cont = field_nb(:,control);

convert_mat = fields_cont(:);
missing_contr = cellfun(@isempty, convert_mat);
fields_cont = convert_mat(~missing_contr(:));

field_number_control = cell2mat(cellfun(@(m) m(:,1), fields_cont', 'UniformOutput', false)');


fields_exp = field_nb(:,experimental);

convert_mat = fields_exp(:);
missing_exp = cellfun(@isempty, convert_mat);
fields_exp = convert_mat(~missing_exp(:));

field_number_experimental = cell2mat(cellfun(@(m) m(:,1), fields_exp', 'UniformOutput', false)');

field_number_all = nan(max(cell_numb),2);
field_number_all(1:cell_numb(1),1) = field_number_control;
field_number_all(1:cell_numb(2),2) = field_number_experimental;

% figure
% boxplot(field_number_all,group_id)
% hold on
% scatter(1,field_number_all(:,1),'filled','jitter','on','jitterAmount',0.08)
% scatter(2,field_number_all(:,2),'filled','jitter','on','jitterAmount',0.08)
% ylabel('# Place Fields')
% title('Field Number')
% figure_handle = 'All Sessions Field Number';
% save_figure(fullfile(figure_path,'PlaceField'),figure_handle)

% I think A histogram is better
max_field_nb = max(field_number_all,[],'all');
edges = linspace(-0.5,max_field_nb+0.5,15);

figure
h = histogram(field_number_all(:,1),edges,'FaceAlpha',0.9);
h.FaceColor = colors{1};
hold on
h = histogram(field_number_all(:,2),edges,'FaceAlpha',0.7);
h.FaceColor = colors{2};
ylabel('# Neurons')
xlabel('# Place Fields')
title('Field Distribution')
figure_handle = 'All Sessions Field Distribution';
save_figure(fullfile(figure_path,'PlaceField'),figure_handle)



% Define bin edges (0 up to max number of fields)
edges = 0:max(field_number_all(:));

% Count neurons per bin for each group
counts1 = histcounts(field_number_all(:,1), [edges, edges(end)+1]);
counts2 = histcounts(field_number_all(:,2), [edges, edges(end)+1]);

% Plot side-by-side bars
figure
b = bar(edges, [counts1(:), counts2(:)], 'grouped', 'BarWidth', 0.9);
b(1).FaceColor = colors{1};
b(2).FaceColor = colors{2};

xlabel('# Place Fields')
ylabel('# Neurons')
legend({'Group 1','Group 2'}, 'Location','northeast')
title('Field Distribution (Side-by-side)')




% make table
[ctrl_counts, ctrl_values, ctrl_percentages] = groupcounts(field_number_control);
[exp_counts, exp_values, exp_percentages] = groupcounts(field_number_experimental);

% Combine all unique values from both groups
all_values = union(ctrl_values, exp_values);

% Initialize arrays for aligned counts and percentages
ctrl_counts_aligned = zeros(size(all_values));
ctrl_percent_aligned = zeros(size(all_values));
exp_counts_aligned = zeros(size(all_values));
exp_percent_aligned = zeros(size(all_values));

% Align control values
[~, idx_ctrl] = ismember(ctrl_values, all_values);
ctrl_counts_aligned(idx_ctrl) = ctrl_counts;
ctrl_percent_aligned(idx_ctrl) = ctrl_percentages;

% Align experimental values
[~, idx_exp] = ismember(exp_values, all_values);
exp_counts_aligned(idx_exp) = exp_counts;
exp_percent_aligned(idx_exp) = exp_percentages;

% Make final table
T = table(all_values, ...
          ctrl_counts_aligned, ctrl_percent_aligned, ...
          exp_counts_aligned, exp_percent_aligned, ...
          'VariableNames', {'Value', 'Control_Count', 'Control_Percent', 'Experimental_Count', 'Experimental_Percent'});

% Create a figure
f = figure('Position', [100 100 800 400]);

% Display table using uitable
t = uitable(f, 'Data', T{:,:}, 'ColumnName', T.Properties.VariableNames, ...
            'Units', 'Normalized', 'Position', [0 0 1 1]);

figure_handle = 'Table Field Distribution';
save_figure(fullfile(figure_path,'PlaceField'),figure_handle)
