function [average,SE] = genotype(variable,negative,positive)
    % always keep control genotype as the first value in "gen"

    average = [mean(variable(:,negative),2), mean(variable(:,positive),2)];
    SE = [std(variable(:,negative),0,2)/sqrt(length(variable(:,negative))),std(variable(:,positive),0,2)/sqrt(length(variable(:,positive)))];
end