function sum_trials = sum_matrix(matrix,animals,n)

% Separate data into sessions/days
% n should reflect the elements to create the sum over
    s1 = size(matrix,1); 
    m = s1 - mod(s1,n);
    
    for s = 1:animals
        y = reshape(matrix(1:m,s), n, []);
        
        Avg = transpose(sum(y,1));
        sum_trials(:,s) = Avg;
    end
end