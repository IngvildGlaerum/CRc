%% Basic Features: comparison
% Converted from MATLAB Live Script (.mlx) to plain .m for version control.
% Source of truth is the .mlx file; regenerate this export if the .mlx changes.

clear; close all; clc;
animalID = [65622,63383,67330,67456,66922,65588,67324,67141];
animals = length(animalID);
days = 6;

main_dir = get_data_root();
meta_table = readtable(fullfile(main_dir,'Metadata.xlsx'),"Sheet",'Data ID');

% Define output variables
SI = cell(days,animals);
sparsity = cell(days,animals);
si_rate = cell(days,animals);
FR = cell(days,animals);
genotype_idx = nan(1,animals);
si_trials = cell(days,animals);
animals_field_size = cell(days,animals);
animals_peak = cell(days,animals);
animals_peak_zone = cell(days,animals);
animals_field_fr = cell(days,animals);

% scaling factors
flipped_idx = [7,6,5,4,3,2,1,8];

save_path = fullfile(main_dir,'CompiledData');

%% Compile data

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
            session_num = 'A';
            animal_ID = num2str(animalID(a));
    
            % load variables
            animal_dir = char(fullfile(main_dir,animal_ID,d));
            path = char(fullfile(main_dir,animal_ID,d,session_num,'data/Processed'));
            load(fullfile(path, 'postprocessed.mat'));
            load(fullfile(path, 'behavior.mat'));
            load(fullfile(path, 'Position_tuning.mat'));
            
            % Get session specifications
            [trials,cell_numb] = size(trial_spikes(2:end,valid_cells));
            index = find(strcmp(meta_table.date, d) == 1 & meta_table.animal_id == str2double(animal_ID));
            g = meta_table.genotype(index); % 1 = control, 2 = exp
            genotype_idx(1,a) = g; 
    
            % Spatial Features
            SI{s,a} = spatial_info_content;
            sparsity{s,a} = sparsity_all_cells;
            si_rate{s,a} = spatial_info_rate;
            FR{s,a} = firing_rate;
            si_trials{s,a} = trial_spatial_information;
    
            place_fields.animal(a).experiment(s).name = field_stats;

            %loop through zones -- NB! This is not the way to do this!!
            zone_x_coord = nan(2,8);
            zone_y_coord = nan(2,8);
            for z = 1:8
                zone_dir = fullfile(animal_dir,sprintf('Arm%d.csv',z));
                zone = table2array(readtable(zone_dir));

                % scale zone coordinates                               
                x_zone_min = ceil(min(zone(:,1))*0.1)-5;
                x_zone_max = ceil(max(zone(:,1))*0.1)+5;
                y_zone_min = ceil(min(zone(:,2))*0.1)-5;
                y_zone_max = ceil(max(zone(:,2))*0.1)+5;

                zone_x_coord(:,z) = [x_zone_min;x_zone_max];
                zone_y_coord(:,z) = [y_zone_min;y_zone_max];
            end

      
            field_size = nan(cell_numb,20); % pick a hight field number for nans
            peak = nan(cell_numb,20);
            in_zone = nan(cell_numb,20);
            fr_field = nan(cell_numb,20);

            for c = 1:cell_numb

                stats = field_stats(c).place_field;  
    
                field_numb = length(stats);
                for r = 1:field_numb
                    field_size(c,r) = stats(r).size;
                    peak(c,r) = stats(r).peak;
                    fr_field(c,r) = stats(r).meanRate;
    
                    % Identify arm
                    peak_x = stats(r).peakX;
                    peak_y = stats(r).peakY;
                    
                    % could have done this in the other loop - whatever
                    is_within = zeros(1,8);
                    for i = 1:8
                        is_within(1,i) = (peak_x >=zone_x_coord(1,i) && peak_x <=zone_x_coord(2,i)) && ...
                            (peak_y >=zone_y_coord(1,i) && peak_y <=zone_y_coord(2,i));
                    end

                    % flip the zone coord to the correct arms
                    flp_corrected = is_within(flipped_idx);

                    if sum(flp_corrected) == 0

                        % zero = the center zone
                        in_zone(c,r) = 0;
                    elseif sum(flp_corrected) == 2 
                        % then it is in a corner = place field in center
                        in_zone(c,r) = 0;
                    else
                        in_zone(c,r) = find(flp_corrected);
                    end
    
                end
                
            end
            % start over next session
            clear("field_stats")  % this wasn't the problem, but can't hurt

            animals_field_size{s,a} = field_size; % sparsity is another measure for this
            animals_peak{s,a} = peak;
            animals_peak_zone{s,a} = in_zone;
            animals_field_fr{s,a} = fr_field;
        else
            % If not valid session
            place_fields.animal(a).experiment(s).name = [];
        end
    end
end

% Save the compiled data
save(fullfile(save_path,'CompiledSpatialStats.mat'),"FR","si_rate","SI","sparsity","genotype_idx","si_trials","place_fields","animalID","animals_field_size","animals_peak","animals_peak_zone","animals_field_fr");
