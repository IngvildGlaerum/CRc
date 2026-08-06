function [accuracy_per_trial, all_true, all_pred, true_label_idx, neuron_weights, margins, shuffle_acc] = run_decoder(all_data, labels, trial_labels, n_classes, oversample, min_percent, n_shuffles)
% RUN_DECODER
%   Train/test linear SVM decoder with cross-trial folds.
%   Optionally compute shuffle distribution by permuting training labels.
%
% Inputs:
%   all_data      - [samples x neurons] firing rates or binned spikes
%   labels        - [samples x 1] true class labels
%   trial_labels  - [samples x 1] trial assignment (for cross-validation)
%   n_classes     - number of unique classes expected
%   oversample    - true = oversample minority classes; false = undersample
%   min_percent   - minimum fraction of samples for class to be included in test
%   n_shuffles    - number of shuffles (optional)
%
% Outputs:
%   accuracy_per_trial - [trials x 1] accuracy per held-out trial
%   all_true           - vector of all true test labels
%   all_pred           - vector of all predicted labels
%   neuron_weights     - [trials x neurons] mean abs weight per neuron
%   margins            - decision values aligned to n_classes
%   shuffle_acc        - [n_shuffles x trials] shuffle accuracies
%
% -------------------------------------------------------------------------

if nargin < 6 || isempty(min_percent)
    min_percent = 0.05; % default: 5%
end
if nargin < 7
    n_shuffles = 0;
end

all_classes = unique(labels);
trials = max(trial_labels);

accuracy_per_trial = nan(trials,1);
all_pred   = [];
all_true   = [];
margins    = [];
margins_shuf = [];
true_label_idx = cell(1,trials);
n_neurons  = size(all_data,2);
neuron_weights = nan(trials, n_neurons);

% Preallocate shuffle accuracy if needed
if n_shuffles > 0
    shuffle_acc = nan(n_shuffles, trials);
else
    shuffle_acc = [];
end

for t = 1:trials
    test_trial = t;
    train_idx = trial_labels ~= test_trial;
    test_idx  = trial_labels == test_trial;

    train_data   = all_data(train_idx,:);
    train_labels = labels(train_idx);
    test_data    = all_data(test_idx,:);
    test_labels  = labels(test_idx);

    %%% Balance training set (real) %%%
    classes = unique(train_labels);
    counts = arrayfun(@(x) sum(train_labels==x), classes);

    if oversample
        n_target = max(counts);
        balanced_idx = [];
        for ci = 1:numel(classes)
            idx = find(train_labels==classes(ci));
            idx_os = randsample(idx, n_target, true);
            balanced_idx = [balanced_idx; idx_os];
        end
    else
        n_target = min(counts);
        balanced_idx = [];
        for ci = 1:numel(classes)
            idx = find(train_labels==classes(ci));
            idx_ss = randsample(idx, n_target);
            balanced_idx = [balanced_idx; idx_ss];
        end
    end

    train_data   = train_data(balanced_idx,:);
    train_labels = train_labels(balanced_idx);

    % Ensure column vectors
    train_labels = train_labels(:);
    test_labels = test_labels(:);

    %%% Filter underrepresented classes in TEST %%%

    classes_test = unique(test_labels);
    counts_test  = arrayfun(@(x) sum(test_labels==x), classes_test);
    total_test   = numel(test_labels);
    percentages  = counts_test / total_test;
    valid_classes = classes_test(percentages >= min_percent);

    if numel(valid_classes) < 2
        continue; % skip trial if not enough valid classes
    end

    % Balance test set (optional: this reduces test size!)
    counts_valid = arrayfun(@(x) sum(test_labels==x), valid_classes);
    n_target = min(counts_valid);
    balanced_test = [];
    for ci = 1:numel(valid_classes)
        idx = find(test_labels==valid_classes(ci));
        idx_t = randsample(idx, n_target);
        balanced_test = [balanced_test; idx_t];
    end
    balanced_test_data = test_data(balanced_test,:);
    test_labels = test_labels(balanced_test);
    true_label_idx{t} = balanced_test(:);

    % (Optional) Z-score features
    % ============================
    % mu = mean(train_data,1);
    % sigma = std(train_data,[],1);
    % sigma(sigma==0) = 1;
    % train_data = (train_data - mu) ./ sigma;
    % balanced_test_data = (balanced_test_data - mu) ./ sigma;
    % ============================

    %%% Train model %%%
    tLIN = templateLinear('Learner','svm', 'Regularization','ridge');
    Mdl  = fitcecoc(train_data, train_labels, 'Learners', tLIN, ...
                    'Coding','onevsall');

    % Extract neuron weights
    betas = cellfun(@(b) b.Beta, Mdl.BinaryLearners, 'UniformOutput', false);
    betas = cell2mat(betas'); 
    neuron_weights(t,:) = mean(abs(betas), 2)';

    %%% Predict %%%
    [predicted_labels, score] = predict(Mdl, balanced_test_data);
    accuracy_per_trial(t) = mean(predicted_labels == test_labels);
    all_pred = [all_pred; predicted_labels(:)];
    all_true = [all_true; test_labels(:)];

    % Align scores to all_classes
    score_aligned = nan(size(score,1), n_classes);
    train_classes = Mdl.ClassNames;
    [~, idx_in_all] = ismember(train_classes, all_classes);
    score_aligned(:, idx_in_all) = score;
    margins = [margins; score_aligned];

    %%% Shuffle control %%%
    if n_shuffles > 0
        raw_train_data   = all_data(train_idx,:);
        raw_train_labels = labels(train_idx);
       
        for sh = 1:n_shuffles
            % Shuffle BEFORE balancing, use raw labels
            shuffled_labels = raw_train_labels(randperm(numel(raw_train_labels)));
    
            % Balance shuffled training labels
            classes_shuf = unique(shuffled_labels);
            counts_shuf  = arrayfun(@(x) sum(shuffled_labels==x), classes_shuf);
    
            if oversample
                n_target = max(counts_shuf);
                balanced_idx = [];
                for ci = 1:numel(classes_shuf)
                    idx = find(shuffled_labels==classes_shuf(ci));
                    idx_os = randsample(idx, n_target, true);
                    balanced_idx = [balanced_idx; idx_os];
                end
            else
                n_target = min(counts_shuf);
                balanced_idx = [];
                for ci = 1:numel(classes_shuf)
                    idx = find(shuffled_labels==classes_shuf(ci));
                    idx_ss = randsample(idx, n_target);
                    balanced_idx = [balanced_idx; idx_ss];
                end
            end
    
            shuf_train_data   = raw_train_data(balanced_idx,:);
            shuf_train_labels = shuffled_labels(balanced_idx);
    
            % Make sure labels are column vector
            shuf_train_labels = shuf_train_labels(:);
    
            % Train shuffled model
            sh_tLIN = templateLinear('Learner','svm', 'Regularization','ridge');
            Mdl_shuf = fitcecoc(shuf_train_data, shuf_train_labels, ...
                                'Learners', sh_tLIN, 'Coding','onevsall');
    
            % Predict on balanced test set
            pred_shuf = predict(Mdl_shuf, balanced_test_data);
            shuffle_acc(sh,t) = mean(pred_shuf == test_labels);

        end
    end
end
end