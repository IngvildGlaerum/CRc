function sab = SAB(entries,animals,arm_numb)

     [idx,~] = size(entries);
     for t = 1:idx
        for a = 1:animals
            habituation_trials = entries{t,a};

            %the transpose will determine what it is sorted by
            [arm, t_idx] = find(habituation_trials');

            %make sure you get re-entry into same arm
            arm_time = zeros(length(entries),1);
            arm_time(t_idx) = arm;
            % invert the diff function so you get a true everytime the diff is not 0
            change = [true, diff(arm_time')~=0,true];
            change_idx = find(change);
            
            % use index to get the consequtive arm choises
            traject = arm_time(change_idx(1:end-1),:);
            
            %how many unique enties out of every 3rd or eight' entry
            test = ceil(length(traject)/arm_numb);
            tot_entry = length(traject);
            j = 1;
            alteration = [];
            for i = 1:test
    
                if j>tot_entry-arm_numb
                    window = traject(j:end);
                    alter = numel(unique(window));%-1 to get alteration between
                else
                    window = traject(1*j:arm_numb*i);
                    alter = numel(unique(window));
                j = j+arm_numb;
                end
                alteration(i,1) = alter;
                
            end
            sab(t,a) = sum(alteration)/(tot_entry-1);
        end
        
    end
end