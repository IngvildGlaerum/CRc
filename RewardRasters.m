%% Raster Plots - Reward Alignment
% Converted from MATLAB Live Script (.mlx) to plain .m for version control.
% Source of truth is the .mlx file; regenerate this export if the .mlx changes.

clear; close all; clc;
animalID = [65622,63383,67330,67456,66922,65588,67324,67141];
animals = length(animalID);
warning("off","all")


main_dir = get_data_root();
meta_table = readtable(fullfile(main_dir,'Metadata.xlsx'),"Sheet",'Data ID');
% Define variables
reward_zones = [3,4,7]; 
window = 2;             % 8 or 2 seconds
binSize = 0.2;          % 20 ms
edges_pop = -window:binSize:window;
nBins = numel(edges_pop)-1;


for a = 1:length(animalID)

    identity = [meta_table.animal_id,meta_table.experiment_type];
    date = cell2mat(meta_table.date);

    exp = find((identity(:,1) == animalID(a) & identity(:,2) == 2)+(identity(:,1)== animalID(a) & identity(:,2) == 3));
    [session,~] = size(date(exp,:));
    validity = meta_table.validity(exp);
    d = date(exp,:);


    for s = 1:session
        if validity(s) == 1

        d = date(exp(s),:);
        else

            continue
        end

        session_num = 'A';
        animal_ID = num2str(animalID(a));

        % load variables
        path = char(fullfile(main_dir,animal_ID,d,session_num,'data/Processed'));
        load(fullfile(path, 'postprocessed.mat'));
        load(fullfile(path, 'behavior.mat'));

        % filter for valid cells (QM)
        trial_spikes = trial_spikes(2:end,valid_cells); 
        [trials,cell_numb] = size(trial_spikes);

        index = find(strcmp(meta_table.date, d) == 1 & meta_table.animal_id == str2double(animal_ID));
        g = meta_table.genotype(index); % 1 = control, 2 = exp

        figure_path = fullfile(main_dir,"Figures/Ingvild/Ephys/RasterPlots");

        for t = 1:trials
            z = trial_zones{t+1,1};
    
            [t_idx,zones] = find(z);
            trajectory = zeros(length(z),1);
            trajectory(t_idx) = zones;
            
            % The first reward
            array = all_paths{1,t};
            
            matches = ismember(array, reward_zones);
            match_idx = find(matches);
            matched_values = unique(array(match_idx),'stable');

            reward1 = matched_values(1);
            reward2 = matched_values(2);
            reward3 = matched_values(3);

         
            r1_idx = find(trajectory == reward1,1,'first') + 5;
            r2_idx = find(trajectory == reward2,1,'first') + 5;
            r3_idx = find(trajectory == reward3,1,'first') + 5;
            
            % Get the exact entry point and position
            reward1_pos = trial_position{t+1,1}(r1_idx,:);
            reward2_pos = trial_position{t+1,1}(r2_idx,:);
            reward3_pos = trial_position{t+1,1}(r3_idx,:);
            time_reward(t,1) = reward1_pos(1,1);
            time_reward(t,2) = reward2_pos(1,1);
            time_reward(t,3) = reward3_pos(1,1);


            % velocity_reward(t,1) = velocity{t}(r1_idx,:);
            % velocity_reward(t,2) = velocity{t}(r2_idx,:);
            % velocity_reward(t,3) = velocity{t}(r3_idx,:);
        end

       % get spiking +- 8 sec before the zero point
       for r = 1:3
           reward_name = sprintf(' Reward %d',r);
           pop_matrix = nan(cell_numb, nBins); % each row = neuron
           for c = 1:cell_numb
            
                all_rel_spikes = []; % store all aligned spikes
                all_counts = zeros(trials, nBins); % trial × bin
            
                figure('Visible','off');
                
                % --- Raster plot ---
                subplot(2,1,1);
                hold on;
                
                for t = 1:trials
                    % Get spike times for this trial (absolute times in seconds)
                    trial_abs_spikes = trial_spikes{t,c};
                    
                    % Align to reward time for this trial
                    rel_spikes = trial_abs_spikes - time_reward(t,r);
                    
                    % Keep only spikes in ±window
                    rel_spikes = rel_spikes(rel_spikes >= -window & rel_spikes <= window);
                    
                    % Store for histogram
                    all_rel_spikes = [all_rel_spikes; rel_spikes(:)];

                    % Bin spikes
                    counts_pop = histcounts(rel_spikes, edges_pop);
                    % Convert to firing rate (Hz)
                    fr = counts_pop / binSize;
                    
                    all_counts(t,:) = fr;
                    
                    % Plot each spike as a short vertical line
                    for sp = 1:length(rel_spikes)
                        line([rel_spikes(sp), rel_spikes(sp)], [t-0.4, t+0.4], 'Color', 'k');
                    end
                end
                
                % Formatting raster
                xlabel('Time from reward entry (s)');
                ylabel('Trial');
                xlim([-window, window]);
                ylim([0.5, trials + 0.5]);
                yline(0, 'k'); % time zero
                set(gca, 'YDir', 'reverse'); % trial 1 at top
                name = sprintf('Cell %d',c);
                title(strcat(name,' - Spike Raster aligned to',reward_name));
                
                % --- Peri-event histogram ---
                subplot(2,1,2);
                edges = -window:binSize:window;
                counts = histcounts(all_rel_spikes, edges);
                
                % Convert counts to firing rate (Hz)
                firing_rate = counts / (trials * binSize);
                
                bar(edges(1:end-1) + binSize/2, firing_rate, 'k', 'BarWidth', 1);
                
                xlabel('Time from reward entry (s)');
                ylabel('Firing rate (Hz)');
                xlim([-window, window]);
                yline(0, 'k');
                title('Peri-event Time Histogram');

                % Save figure
                figure_handle = strcat(name,' Raster ',reward_name);
                save_figure(fullfile(figure_path,animal_ID,d),figure_handle)
                close

                % Population Average across trials
                mean_fr = mean(all_counts, 1);
    
                % Z-score per neuron
                z_fr = (mean_fr - mean(mean_fr)) / std(mean_fr);
                
                pop_matrix(c,:) = z_fr;

           end

           % Sort neurons by latency of peak firing
           [~, peak_idx] = max(pop_matrix, [], 2);
           [~, sort_order] = sort(peak_idx);
           sorted_matrix = pop_matrix(sort_order, :);
            
           % Plot heatmap
           figure('Visible','off');
           imagesc(edges_pop(1:end-1) + binSize/2, 1:cell_numb, sorted_matrix);
           colormap jet;
           colorbar;
           xlabel(sprintf('Time from reward %d (s)',r));
           ylabel('Neurons');
           title('Population Peri-Event Response (Z-scored)');
           yline(0, 'k', 'LineWidth', 1);
           xline(0,'w','LineWidth',1)
           set(gca, 'YDir', 'normal');

           % Save population figure
           day = sprintf(' Day %d',s);
           figure_handle = strcat(animal_ID , day,reward_name);
           save_figure(figure_path,figure_handle)
           close
      
        end

    end
end
