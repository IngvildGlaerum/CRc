function score_t_maze = score_mat(data, animal_numb, test_onset)
    % if the column index of sample and test are the same then make error for
    % that trial. If they are not, then they have alternated and succeeded
    sample = data(test_onset:2:end-1,:);
    test = data((test_onset+1):2:end,:);
    
    score_t_maze = zeros(size(sample));
    for j = 1:animal_numb
        sc = sum([sample(:,j), test(:,j)],2);
        
        [row,~] = find(sc==3);
        score_t_maze(row,j) = 1; 
    end

end

