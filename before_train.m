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

%% ================= 1. Data Loading =================
% [Security] Use a relative path to load the sample dataset, 
% rather than the complete basin-wide data.
filePath = fullfile(pwd, 'data', 'sample_Train_3D.mat');

if ~isfile(filePath)
    error('[!] File not found: %s. Please ensure your current directory is the project root.', filePath);
end

fprintf('------------------------------------------------------\n');
fprintf('Loading data (please monitor Task Manager for memory usage)...\n');
t_start = tic;
load(filePath); 
t_load = toc(t_start);
fprintf('Data loaded successfully. Time elapsed: %.2f seconds\n', t_load);

%% ================= 2. Memory Estimation =================
fprintf('\n========== [1] Memory Requirement Assessment ==========\n');

% 1. Calculate actual memory footprint of the loaded dataset
vars = whos('x_train', 'y_train', 'x_test', 'y_test');
totalBytes = sum([vars.bytes]);
dataGB = totalBytes / (1024^3);

fprintf('Current dataset size (RAM): \t%.2f GB\n', dataGB);

% 2. Retrieve system physical memory info (Windows only)
try
    [userView, systemView] = memory;
    totalRAM = systemView.PhysicalMemory.Total / (1024^3);
    availableRAM = systemView.PhysicalMemory.Available / (1024^3);
    fprintf('Total physical memory:      \t%.2f GB\n', totalRAM);
    fprintf('Available memory:           \t%.2f GB\n', availableRAM);
catch
    % Fallback for Linux/Mac
    totalRAM = 16; % Assumed baseline
    fprintf('Unable to retrieve system memory info. Assuming 16 GB total.\n');
end

% 3. Estimate peak memory during training
% Rule of thumb: Peak RAM ~= Dataset Size * 1.5 (internal copies) + 2GB (overhead)
estimatedPeak = dataGB * 1.5 + 2.0; 

fprintf('------------------------------------------------------\n');
fprintf('[Training Feasibility Analysis]\n');
fprintf('Estimated peak memory required: \t~%.2f GB\n', estimatedPeak);

if estimatedPeak < totalRAM * 0.9
    fprintf('[OK] Status: Safe. Sufficient memory available.\n');
elseif estimatedPeak < totalRAM
    fprintf('[!] Status: Marginal. Consider closing other heavy applications.\n');
else
    fprintf('[X] Status: Critical! High risk of Out-Of-Memory error.\n');
    fprintf('    Recommendation: Reduce BatchSize or increase spatial sampling stride.\n');
end
fprintf('------------------------------------------------------\n');

%% ================= 3. Data Dimension & Feature Check =================
fprintf('\n========== [2] Data Dimension Check ==========\n');
[nFeat, nTime, nTrain] = size(x_train);
fprintf('Number of Features: %d\n', nFeat);
fprintf('Time Steps:         %d\n', nTime);
fprintf('Training Samples:   %d\n', nTrain);

if any(isnan(x_train(:))) || any(isnan(y_train(:)))
    warning('Fail: Dataset contains NaN values! Training will not converge.');
else
    fprintf('[OK] Data integrity verified (No NaNs found).\n');
end

%% ================= 4. Visual Inspection =================
fprintf('\n========== [3] Visual Inspection ==========\n');
sample_idx = randi(nTrain); 
fprintf('Plotting waveforms for Sample #%d...\n', sample_idx);

if exist('times', 'var'), t_axis = times; else, t_axis = 1:nTime; end

figure('Name', 'Data Inspection', 'Color', 'w', 'Position', [100, 100, 800, 600]);

% Subplot 1: GPP (Target)
subplot(3,1,1); 
plot(t_axis, squeeze(y_train(1, :, sample_idx)), 'g-', 'LineWidth', 1.5);
title(sprintf('Target Variable: GPP (Sample %d)', sample_idx)); 
ylabel('Value'); grid on; axis tight;

% Subplot 2: Meteorological Features
subplot(3,1,2);
colors = {'b', 'r', 'k', 'm', 'c'};
hold on;
plot_count = min(3, nFeat-2); 
for i = 1:plot_count
    data_trace = squeeze(x_train(i, :, sample_idx));
    norm_trace = (data_trace - min(data_trace)) / (max(data_trace) - min(data_trace) + 0.001);
    plot(t_axis, norm_trace, 'Color', colors{i}, 'LineWidth', 1);
end
title(sprintf('Top %d Input Features (Normalized Trend)', plot_count));
legend(arrayfun(@(x) sprintf('Feat %d', x), 1:plot_count, 'UniformOutput', false));
grid on; axis tight;

% Subplot 3: Seasonality Encoding
subplot(3,1,3);
ts_sin = squeeze(x_train(end-1, :, sample_idx)); 
ts_cos = squeeze(x_train(end, :, sample_idx));
plot(t_axis, ts_sin, '--'); hold on; plot(t_axis, ts_cos, ':');
title('Spatiotemporal Encoding (Cyclic)'); legend('Sin', 'Cos'); grid on; axis tight;

fprintf('Inspection complete. If no errors appear, you may proceed to training.\n');