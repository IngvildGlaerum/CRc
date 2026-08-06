function save_figure(path,figure_title)
% path cannot be in a cell format
    
    if ~exist(path,'dir')
        mkdir(path)
    end
    saveas(gcf, fullfile(path,strcat(figure_title, '.png')));
end