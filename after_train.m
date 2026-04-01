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

%% ================= 1. Load Model and Test Data =================
modelFile = fullfile(pwd, 'models', 'Final_Model_Result.mat');
dataFile  = fullfile(pwd, 'data', 'sample_Train_3D.mat'); 

if ~isfile(modelFile) || ~isfile(dataFile)
    error('File not found! Please verify the directory structure (models/ and data/ folders).');
end

fprintf('Loading model and test data...\n');
load(modelFile, 'net', 'ps_in', 'ps_out', 'info');
load(dataFile, 'x_test', 'y_test'); 

[nFeat, nTime, nSample] = size(x_test);
fprintf('Model loaded successfully. Test samples: %d\n', nSample);

%% ================= 2. Review Training History =================
figure('Name', 'Training Process Review', 'Color', 'w', 'Position', [100, 100, 800, 400]);

% RMSE Plot
subplot(1, 2, 1);
plot(info.TrainingRMSE, 'b-', 'LineWidth', 1.5); hold on;
if ~isempty(info.ValidationRMSE)
    val_rmse = info.ValidationRMSE;
    x_val = find(~isnan(val_rmse));
    plot(x_val, val_rmse(x_val), 'k--o', 'LineWidth', 1.5, 'MarkerSize', 4);
    legend('Training RMSE', 'Validation RMSE');
else
    legend('Training RMSE');
end
title('RMSE Descent Curve'); xlabel('Iterations'); ylabel('RMSE'); grid on;

% Loss Plot
subplot(1, 2, 2);
plot(info.TrainingLoss, 'r-', 'LineWidth', 1);
title('Loss Curve'); xlabel('Iterations'); ylabel('Loss'); grid on;

%% ================= 3. Execute Prediction =================
fprintf('Executing model predictions on the test set...\n');

% Normalize input data
x_test_2D = reshape(x_test, nFeat, []);
x_test_2D = mapminmax('apply', x_test_2D, ps_in);
x_test_Norm_3D = reshape(single(x_test_2D), nFeat, nTime, nSample);

fprintf('Formatting data for inference...\n');
x_cell = cell(nSample, 1);
for k = 1:nSample
    x_cell{k} = x_test_Norm_3D(:, :, k); 
end

fprintf('Predicting...\n');
Y_Pred_Cell = predict(net, x_cell, 'MiniBatchSize', 1024);

if iscell(Y_Pred_Cell)
    Y_Pred_Norm = cat(3, Y_Pred_Cell{:});
else
    Y_Pred_Norm = Y_Pred_Cell;
end

% Reverse normalization to obtain actual physical GPP values
Y_Pred_2D = reshape(Y_Pred_Norm, 1, []);
Y_Pred_Real_2D = mapminmax('reverse', Y_Pred_2D, ps_out);
Y_Pred_Real = reshape(Y_Pred_Real_2D, 1, nTime, nSample);
Y_True = y_test; 

fprintf('Prediction complete.\n');

%% ================= 4. Calculate Evaluation Metrics =================
all_pred = double(Y_Pred_Real(:));
all_true = double(Y_True(:));

% Filter out NaNs if any
valid_idx = ~isnan(all_pred) & ~isnan(all_true);
all_pred = all_pred(valid_idx);
all_true = all_true(valid_idx);

% RMSE
rmse_val = sqrt(mean((all_pred - all_true).^2));
% R-squared
SSR = sum((all_pred - all_true).^2);
SST = sum((all_true - mean(all_true)).^2);
r2_val = 1 - SSR/SST;
% Pearson Correlation
corr_val = corr(all_pred, all_true);

fprintf('\n========== Model Performance ==========\n');
fprintf('RMSE : %.4f\n', rmse_val);
fprintf('R^2  : %.4f\n', r2_val);
fprintf('Corr : %.4f\n', corr_val);
fprintf('=======================================\n');

%% ================= 5. Visual Assessment =================
figure('Name', 'Prediction Assessment', 'Color', 'w', 'Position', [100, 100, 1000, 500]);

% Scatter Plot
subplot(1, 2, 1);
sample_scatter_idx = randperm(length(all_pred), min(5000, length(all_pred)));
scatter(all_true(sample_scatter_idx), all_pred(sample_scatter_idx), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'k--', 'LineWidth', 2); 
xlabel('Observed GPP');
ylabel('Predicted GPP');
title(sprintf('Scatter Plot (R^2 = %.2f)', r2_val));
grid on; axis square;

% Time Series Plot
subplot(1, 2, 2);
pixel_idx = randi(nSample);
ts_true = squeeze(Y_True(1, :, pixel_idx));
ts_pred = squeeze(Y_Pred_Real(1, :, pixel_idx));

if exist('times', 'var'), t_axis = times; else, t_axis = 1:nTime; end

plot(t_axis, ts_true, 'k-', 'LineWidth', 1.5); hold on;
plot(t_axis, ts_pred, 'r--', 'LineWidth', 1.5);
legend('Observation', 'Prediction');
title(sprintf('Time Series Comparison (Sample #%d)', pixel_idx));
xlabel('Time'); ylabel('GPP');
grid on;