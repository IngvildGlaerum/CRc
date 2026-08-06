function ste = SE(data)
% Calculate the standard error of the mean and omitnans

    ste = std(data,'omitnan')/sqrt(length(data));
    
end