function plot_si_vs_fr(si,fr, control, experimental, cell_numb, colors, save_path, ylabel_text, xlabel_text, save_tag)

    % plot_group_comparison
    % Plots boxplot, scatter, and histogram comparisons between control and experimental groups
    %
    % Parameters:
    %   data            - Input data matrix (e.g., SI or sparsity)
    %   control         - Logical or index array for control columns
    %   experimental    - Logical or index array for experimental columns
    %   cell_numb       - 1x2 vector with number of neurons in [control, experimental]
    %   edges           - Predefine the range for the histogram
    %   colors          - Cell array with color definitions {color_control, color_experimental}
    %   save_path_base  - Path to save the figures (e.g., 'E:\path\to\figures\')
    %   ylabel_text     - Label for y-axis (e.g., 'Spatial Information (bit/spike)')
    %   x_label_text    - Label for x-axis in histogram (e.g., same as ylabel_text)
    %   save_tag        - String tag used in filename (e.g., 'SI')

    % Flatten and clean control data SI
    si_cont = si(:,control);
    flat_cont = si_cont(:);
    flat_cont = flat_cont(~cellfun(@isempty, flat_cont))';
    si_all_cont = cell2mat(flat_cont);
    
    % Flatten and clean experimental data SI
    si_exp = si(:,experimental);
    flat_exp = si_exp(:);
    flat_exp = flat_exp(~cellfun(@isempty, flat_exp))';
    si_all_exp = cell2mat(flat_exp);
 
    % Combine into matrix
    % cell_numb_max = max([numel(data_all_cont),numel(data_all_exp)]);
    si_all = nan(max(cell_numb),2);
    si_all(1:length(si_all_cont),1) = si_all_cont;
    si_all(1:length(si_all_exp),2) = si_all_exp;



     % Flatten and clean control data FR
    fr_cont = fr(:,control);
    flat_cont = fr_cont(:);
    flat_cont = flat_cont(~cellfun(@isempty, flat_cont))';
    fr_all_cont = cell2mat(flat_cont);
    
    % Flatten and clean experimental data FR
    fr_exp = fr(:,experimental);
    flat_exp = fr_exp(:);
    flat_exp = flat_exp(~cellfun(@isempty, flat_exp))';
    fr_all_exp = cell2mat(flat_exp);
    
    % Combine into matrix
    % cell_numb_max = max([numel(data_all_cont),numel(data_all_exp)]);
    fr_all = nan(max(cell_numb),2);
    fr_all(1:length(fr_all_cont),1) = fr_all_cont;
    fr_all(1:length(fr_all_exp),2) = fr_all_exp;

    figure
    scatter(fr_all(:,1), si_all(:,1), 'filled','MarkerFaceColor',colors{1});
    hold on
    s = scatter(fr_all(:,2), si_all(:,2), 'filled','MarkerFaceColor',colors{2});
    s.MarkerFaceAlpha = 0.65;
    ylabel(ylabel_text)
    xlabel(xlabel_text)
    print(gcf, '-dsvg', '-painters', fullfile(save_path,[save_tag '_box.svg']));

end