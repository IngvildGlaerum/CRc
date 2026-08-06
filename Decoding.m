%% Decoding with bootstrapping
% Converted from MATLAB Live Script (.mlx) to plain .m for version control.
% Source of truth is the .mlx file; regenerate this export if the .mlx changes.

clear; close all; clc;
%NB! this was changed to run on a cluster, but the computations are the same
% -------- USER PARAMETERS --------
animalID = [65622,63383,67330,67456,66922,65588,67324,67141];
main_dir = get_data_root();
meta_table = readtable(fullfile(main_dir,'Metadata.xlsx'),"Sheet",'Data ID');

% Decoding parameters
bin_size = 2;                                                       % 1 = 20 ms bins | 2 = 40 ms bins
decode_modes = {'reward_vs_nonreward','reward_zone','all_zones'};   % modes to run
oversample = true;                                                  % true = oversample (with replacement), false = subsample
min_percent = 0.05;                                                 % minimum percent coverage
n_shuffles = 1000;                                                  % number of shuffels
bootstraps = 100;                                                   % number of bootstrap replicates
N_min = 17;                                                         % minimum number of cells
reward_zones = [3,4,7];
colors = {'#284B5E','#BC554C'};                                     % standard group colors 
warning("off","all")
animals = length(animalID);
c_colors = linspecer(9);                                            % returns 9 RGB triplets
class_colors = [c_colors(end,:);c_colors(1:8,:)];                   % keep colors consitant across analysis
fixed_classes = 0:8;                                                % all possible class labels


% Output variables
decoding = cell(6,8);

%%%%%%%%%%% MAIN LOOP %%%%%%%%%%
for a = 1:animals
    identity = [meta_table.animal_id,meta_table.experiment_type];
    date = cell2mat(meta_table.date);
    exp = find((identity(:,1) == animalID(a) & identity(:,2) == 2) + (identity(:,1)== animalID(a) & identity(:,2) == 3));
    [session,~] = size(date(exp,:));
    validity = meta_table.validity(exp);
    d = date(exp,:);

    for s = 1:session 
        if validity(s) ~= 1
            continue
        end
        d = date(exp(s),:);
        session_num = 'A';
        animal_ID = num2str(animalID(a));

        % Load variables
        path = char(fullfile(main_dir,animal_ID,d,session_num,'data/Processed'));
        load(fullfile(path, 'postprocessed.mat'));

        % Filter valid cells
        trial_spikes = trial_spikes(2:end,valid_cells); 
        [trials,cell_numb] = size(trial_spikes);

        index = find(strcmp(meta_table.date, d) == 1 & meta_table.animal_id == str2double(animal_ID));
        g = meta_table.genotype(index); % 1 = control, 2 = exp
        day = sprintf('Day %d',s);

        % Time binning
        time_start = find(trial_position{1,1}(:,1) == trial_position{2,1}(1,1));
        time_end = find(trial_position{1,1}(:,1) == trial_position{trials+1,1}(end,1));
        time = trial_position{1,1}(time_start:time_end,1);
        
        nbins = floor(length(time)/ bin_size);
        time_binned = linspace(time(1,1),time(end,1),nbins)';
        time_edges = linspace(time(1), time(end), nbins+1)'; % note nbins+1 edges % the same?


        binned_spikes = zeros(cell_numb,numel(time_binned)-1);

        % Bin spike trains
        for c = 1:cell_numb
            spike = cell2mat(trial_spikes(:,c)');
            binned_spikes(c,:) = histcounts(spike,time_binned);
        end


        % Identify trial and zone labels
        trial_indx = nan(numel(time),1);
        zone_indx = nan(numel(time),1);
        for i = 1:trials
            trial_indx(start(i):stop(i)) = i;

            z  = trial_zones{i+1,1};
            [zones, t_idx] = find(z');
            zones_time = zeros(length(z),1);
            zones_time(t_idx) = zones;
            zone_indx(start(i):stop(i)) = zones_time;
        end
        between_trials = isnan(trial_indx);
        trial_indx(between_trials) = 0; % between-trials label 0
        trial_id = trial_indx(time_start:time_end,1); 
        
        zone_indx(between_trials) = 0;
        zone_id = zone_indx(time_start:time_end,1);
        

        % Bin labels correct time_bins
        bin_idx = discretize(time, time_edges); % indices 1:nbins
        
        % Bin trial_id using mode
        binned_trial_id = accumarray(bin_idx(~isnan(bin_idx)), trial_id(~isnan(bin_idx)),[nbins, 1], @(x) mode(x), 0);
        
        % Bin zones_time using mode
        binned_zone_time = accumarray(bin_idx(~isnan(bin_idx)), zone_id(~isnan(bin_idx)),[nbins, 1], @(x) mode(x), 0);

        % Align label data length with spiking data
        t_labels = binned_trial_id(1:end-1)';
        z_labels = binned_zone_time(1:end-1)';
        r_labels = ismember(z_labels,reward_zones);             % Should I only consider the center zone here?

        % Create reward identity labels
        r_identity_labels = zeros(1,length(r_labels));
        r_identity_labels(r_labels) = z_labels(r_labels);

        % remove time inbetween trials 
        valid_idx = t_labels > 0; 
        all_data = binned_spikes(:,valid_idx)'; % obs x features
        zone_labels = z_labels(valid_idx);
        reward_labels = r_labels(valid_idx);
        reward_identity_labels = r_identity_labels(valid_idx);
        trial_labels = t_labels(valid_idx);


        % Try 5-fold testing: 20 trian | 5 test
        fold_vec = 1:5:30;
        for f = 1:5
            vec = fold_vec(f):1:fold_vec(f+1)-1;
            new_vec = ismember(trial_labels,vec);
            trial_labels(new_vec) = f;
        end
        
        
        % RUN DECODING FOR EACH MODE
        output = cell(7,numel(decode_modes));
        for dm = 1:numel(decode_modes)
            mode_decoding = decode_modes{dm};

            switch mode_decoding
                case 'reward_vs_nonreward'
                    labels = reward_labels;             % 0|1
                    n_classes = 2;
                    mode_classes = [0,1];
                case 'reward_zone'
                    labels = reward_identity_labels;    % 0|3|4|7
                    n_classes = 4;
                    mode_classes = [0,3,4,7];
                case 'all_zones'
                    labels = zone_labels;               % 0|1|2|3|4|5|6|7|8
                    n_classes = 9;
                    mode_classes = [0,1,2,3,4,5,6,7,8];
            end
            
            % smooth data 

            smoothed_data = all_data; % prealloc
            for t = 1:max(trial_labels)
                idx = (trial_labels==t);
                if any(idx)
                    smoothed_data(idx,:) = smoothdata(all_data(idx,:), 1, 'gaussian', 40);
                end
            end

            %%%%%%%%%% Bootstrap cells %%%%%%%%%% 
           
            boot_true = cell(bootstraps,1);
            boot_pred = cell(bootstraps,1);
            boot_true_idx = cell(bootstraps,max(trial_labels));
            boot_acc = nan(bootstraps, max(trial_labels));
            boot_shuffle = cell(bootstraps,1);
            boot_margins = cell(bootstraps,1);
            boot_margins_shuffle = cell(bootstraps,n_shuffles);
            boot_weights = cell(bootstraps,1);
            boot_weights_idx = nan(N_min,bootstraps);

            % Parallelizing the bootstraps 
            parfor b = 1:bootstraps
                rand_idx = randsample(cell_numb, N_min, true);
                subsampled_data = smoothed_data(:, rand_idx);
                
                % Run decoder on subsampled data
                [acc, true_labels, pred_labels, test_label_idx, neuron_weights, margins,shuffle_acc,margins_shuffle] = ...
                run_decoder(subsampled_data, labels, trial_labels, n_classes, oversample, min_percent,n_shuffles);
                
                boot_acc(b,:) = acc;
                boot_shuffle{b} = shuffle_acc;
                boot_true{b} = true_labels;
                boot_pred{b} = pred_labels;
                boot_true_idx(b,:) = test_label_idx;
                boot_margins{b}= margins;
                boot_margins_shuffle(b,:) = margins_shuffle;
                boot_weights{b,1}= neuron_weights;
                boot_weights_idx(:,b)= rand_idx;            % keep track of which cells were selected

            end


            output{1,dm} = boot_acc;
            output{2,dm} = boot_shuffle;
            output{3,dm} = boot_true;
            output{4,dm} = boot_pred;
            output{5,dm} = boot_true_idx;
            output{6,dm} = conf_avg;
            output{7,dm} = boot_margins;

        end
        % save output for comparison
        save(fullfile(path,'Decoding.mat'),"output");
        decoding{s,a} = output; %in case I have to run this in segments

    end
end

save(fullfile(main_dir,'CompiledData','Decoding.mat'),"decoding");
