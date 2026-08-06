
function [trajectory,completed] = traject(entries,goal)
    
    %the ' will determine what it is sorted by
    [arm, t_idx] = find(entries');

    %make sure you get re-entry into same arm
    arm_time = zeros(length(entries),1);
    arm_time(t_idx) = arm;

    % invert the diff function so you get a true everytime the diff is not 0
    change = [true, diff(arm_time')~=0,true];
    change_idx = find(change);

    % use index to get the consequtive arm choises
    tr = arm_time(change_idx(1:end-1),:); 
    trajectory = nonzeros(tr);
    
    % % control for start arm (don't record it)
    % if exist('start_arm','var')
    %     if trajectory(1) == start_arm
    % 
    %         trajectory = trajectory(2:end);
    %     end
    % end
    
    % snip the trajectory after all goal arms were entered once
    if exist('goal','var')
       
        if sum(ismember(goal,trajectory))<3
            trajectory = trajectory(1:end);
            completed = length(arm_time); %set to last sampling time
        else
           [idx1,~] = find(trajectory == goal(1));
           [idx2,~] = find(trajectory == goal(2));
           [idx3,~] = find(trajectory == goal(3));

           idx = [idx1(1),idx2(1),idx3(1)];

           %Completed only shows up if you specify the output variable
           [midx,x] = max(idx);
           [complete,~] = find(arm_time == goal(x));
           completed = complete(1,1);
           trajectory = trajectory(1:midx);
       
        end
    end
end