function [entry_error,rm_error,wm_error] = EntryError(path,correct,dim,incorrect)
% INPUT VARIABLES
%path = a vector with the arm index numbers of all entries during the trial
%correct = a vector with the identification numbers for correct entry arm choice
%dim = specifies if you want the errors made before (1) the first treat was consumed or after (2) or after (3)

% OUTPUT VARIABLES
    if dim == 1
        
        correct_indx = find(path==correct(1) | path==correct(2) | path == correct (3));

        % number of errors before the first treat is consumed
        entry_error = length(path(1:correct_indx(1)-1)); % -1 to not include the correct entry
        
        % Are the errors WM errors or RM errors
        if exist('incorrect','var')
  
           rm_error = sum(ismember(incorrect,path(1:correct_indx(1))));
           
           replicates = groupcounts(path(1:correct_indx(1)-1));
           
           wm_error = sum(replicates-1);
        end
        
    elseif dim == 2
        % Analyze secondary errors = errors made between consuming bait 1 and 2
        
        %get the index all correct arm entries
        correct_indx = find(path==correct(1) | path==correct(2) | path == correct (3));
        
        correct2 = find(path(correct_indx)~=path(correct_indx(1)));
        counts = groupcounts(path(correct_indx(1):correct_indx(correct2(1)))); %Include the first correct entry because you are subtracting it later
        
        % Remove the first entry of the two baited arms - the remaining is all wrong entries
        entry_error = sum(counts)-2;
        
        if exist('incorrect','var')
           
           initial_rm = ismember(incorrect,path(correct_indx(1):correct_indx(correct2(1))));
           % control that the RM error wasn't already made in E1
           check_e1 = ismember(incorrect(initial_rm),path(1:correct_indx(1)));
           rm_error = sum(initial_rm)-sum(check_e1);
           
           replicates = groupcounts(path(correct_indx(1):correct_indx(correct2(1))));
           %subtract the first entry of the arm, but add whatever arm was also visited in E1
           arms = [incorrect,correct];
           initial_wm = ismember(arms,path(correct_indx(1):correct_indx(correct2(1))));
           check_wm = ismember(arms(initial_wm),path(1:correct_indx(1)-1)); %don't include the last here (it will be considered twice)
           wm_error = sum(replicates-1)+sum(check_wm); 
        end
        
        
    else
        % analyze errors after consumption of treat 2
        correct_indx = find(path==correct(1) | path==correct(2) | path == correct (3));
        
        correct2 = find(path(correct_indx)~=path(correct_indx(1)));


        %get size of mat beween baited arm 2 and baited arm 3
        counts = length(path(correct_indx(correct2(1)):end));

        %check if animal completed the task
        a = path(correct_indx)~= path(correct_indx(1)); % indicate when animal is not in first correct entry
        b = path(correct_indx)~= path(correct_indx(correct2(1))); % indicate when animal is not in second correct entry 
        c = a.*b; % get the intersection
        
        % if completed the task, subtract the last correct entry
        if sum(c)<1 
            entry_error = sum(counts)-1; %if not completed, subtract bait 2 not bait 3
        else
            entry_error = sum(counts)-2; %if completed, subtract bait 2&3
        end
        
        if exist('incorrect','var')
           
           initial_rm = ismember(incorrect,path(correct_indx(correct2(1)):end));
           % control that the RM error wasn't already made before
           check_it = ismember(incorrect(initial_rm),path(1:correct_indx(correct2(1)))); %is this indexing correct?
           rm_error = sum(initial_rm)-sum(check_it);
           



           %subtract the first entry of the arm, but add whatever arm was also visited before
          
           % arms = [incorrect,correct];
           % initial_wm = ismember(arms,path(correct_indx(correct2(1)):end));
           % 
           e3 = path(correct_indx(correct2(1)):end); 

           %has the arms been visited before?
           visit_prior = ismember(e3,path(1:correct_indx(correct2(1))-1));
           initial_visit = length(unique(e3(visit_prior))); % get rid of replicates here (avoid double counting)


           %are there replicates within the subset?
            
           replicates = groupcounts(path(correct_indx(correct2(1)):end))-1; 
           % replicates_adj = sum(replicates)-length(replicates); % this gets rid of all first entries in this section

           wm_error = sum(replicates) + initial_visit;

        end
        
    end
end