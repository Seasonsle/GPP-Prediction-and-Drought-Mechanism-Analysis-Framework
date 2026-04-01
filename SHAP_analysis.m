% =========================================================================
% Copyright (c) 2024-2025 Le Seasons. All rights reserved.
% 
% This source code is released under the MIT License.
% It is provided explicitly for academic review and reproducibility of the paper:
% "GPP Prediction and Drought Mechanism Analysis in the Huai River Basin"
% 
% NOTE: If you use or adapt this code in your research, you MUST cite the 
% original paper mentioned above.
% =========================================================================

clc; clear; close all;

%% ================= 0. Environment Setup =================
rng(42); % Set random seed for reproducibility

%% ================= 1. Configuration and Loading =================
modelFile = fullfile(pwd, 'models', 'Final_Model_With_PSO-v1.mat');
dataFile  = fullfile(pwd, 'data', 'sample_Train_3D_v1.mat');

fprintf('Status: Loading model and dataset...\n');
load(modelFile, 'net');
load(dataFile, 'x_train', 'x_test', 'y_test');

[nFeat, nTime, nTrain] = size(x_train);
X_Train_2D = reshape(x_train, nFeat, []);
[~, ps_in] = mapminmax(X_Train_2D, 0, 1);
clear x_train X_Train_2D; 

% Define original feature names
featureNames = {'Precip', 'PET', 'SPEI', 'MaxTemp', 'soil', 'Lat', 'Lon', 'SinMo', 'CosMo'};

%% ================= 2. Sample Selection (Stratified) =================
n_strata = 3;            
% Cap samples per strata based on the size of the test set
samples_per_strata = min(200, floor(size(x_test,3)/n_strata)); 
lat_channel_idx = 6;    

fprintf('Status: Performing latitude-based stratified sampling...\n');
Lat_All = squeeze(x_test(lat_channel_idx, 1, :));
[~, sorted_indices] = sort(Lat_All);
total_test = length(Lat_All);
chunk_size = floor(total_test / n_strata);
final_shap_idx = [];

for i = 1:n_strata
    start_ptr = (i-1) * chunk_size + 1;
    if i == n_strata, end_ptr = total_test; else, end_ptr = i * chunk_size; end
    layer_indices = sorted_indices(start_ptr : end_ptr);
    rand_sub_idx = randperm(length(layer_indices), samples_per_strata);
    final_shap_idx = [final_shap_idx; layer_indices(rand_sub_idx)];
end

rand_idx = final_shap_idx;
n_sample_shap = length(rand_idx);

fprintf('Status: Constructing SHAP analysis subset (%d samples)...\n', n_sample_shap);
X_Source = x_test(:, :, rand_idx);
X_Cell_Shap = cell(n_sample_shap, 1);
for k = 1:n_sample_shap
    temp_2d = X_Source(:,:,k);
    temp_2d = mapminmax('apply', temp_2d, ps_in);
    X_Cell_Shap{k} = temp_2d;
end

Ref_Matrix = mean(cat(3, X_Cell_Shap{:}), 3);

%% ================= 3. SHAP Value Calculation =================
fprintf('Status: Initiating SHAP value computation...\n');
% Note: This relies on the custom function shapley_lstm_3d.m
shap_results = shapley_lstm_3d(net, X_Cell_Shap, Ref_Matrix, featureNames);

%% ================= 3.5 Feature Filtering =================
% Remove non-interpretable cyclical encodings from visual output
remove_list = {'SinMo', 'CosMo'}; 
keep_mask = ~ismember(featureNames, remove_list);

shap_results = shap_results(:, keep_mask);
featureNames = featureNames(keep_mask);

n_show_feats = length(featureNames);
fprintf('Status: Excluded features [%s]. Remaining features: %d\n', strjoin(remove_list, ', '), n_show_feats);

%% ================= 4. Visualization =================
fprintf('Status: Generating analytical plots...\n');

% --- Plot 1: Global Feature Importance ---
figure('Name', 'Global Feature Importance', 'Color', 'w', 'Position', [100, 100, 700, 500]);
global_impact = mean(abs(shap_results), 1); 
[sorted_vals, idx] = sort(global_impact, 'ascend');

b = barh(sorted_vals, 'FaceColor', [0.2, 0.4, 0.6], 'FaceAlpha', 0.7);

yticks(1:n_show_feats);
yticklabels(featureNames(idx));
xlabel('Mean |SHAP Value| (Impact on GPP)');
title('Global Feature Importance Ranking');
grid on;

for i = 1:n_show_feats
    text(sorted_vals(i), i, sprintf(' %.4f', sorted_vals(i)), 'VerticalAlignment', 'middle');
end

% --- Plot 2: SHAP Beeswarm Plot ---
figure('Name', 'SHAP Summary Plot', 'Color', 'w', 'Position', [800, 100, 700, 600]);
hold on; xline(0, 'k--');
colors = jet(100); 

for i = 1:n_show_feats
    feat_idx = idx(i); 
    shap_vals_feat = shap_results(:, feat_idx);
    
    curr_name = featureNames{feat_idx}; 
    if strcmp(curr_name, 'Precip'), raw_col=1;
    elseif strcmp(curr_name, 'PET'), raw_col=2;
    elseif strcmp(curr_name, 'SPEI'), raw_col=3;
    elseif strcmp(curr_name, 'MaxTemp'), raw_col=4;
    elseif strcmp(curr_name, 'Wind'), raw_col=5; 
    elseif strcmp(curr_name, 'Lat'), raw_col=6;
    elseif strcmp(curr_name, 'Lon'), raw_col=7;
    else, raw_col=1; 
    end
    
    raw_vals = zeros(n_sample_shap, 1);
    for k=1:n_sample_shap
        raw_vals(k) = mean(X_Cell_Shap{k}(raw_col, :));
    end
    
    norm_raw = (raw_vals - min(raw_vals)) / (max(raw_vals) - min(raw_vals) + 1e-6);
    color_indices = round(norm_raw * 99) + 1;
    
    y_pos = i + (rand(size(shap_vals_feat))-0.5)*0.3;
    
    scatter(shap_vals_feat, y_pos, 30, colors(color_indices,:), 'filled', 'MarkerEdgeAlpha', 0.6);
end

yticks(1:n_show_feats);
yticklabels(featureNames(idx));
xlabel('SHAP Value (Impact on Model Output)');
title('SHAP Summary Plot');
colormap(jet); c = colorbar; c.Label.String = 'Feature Value (Low to High)';
grid on;

fprintf('[OK] SHAP Analysis successfully completed.\n');