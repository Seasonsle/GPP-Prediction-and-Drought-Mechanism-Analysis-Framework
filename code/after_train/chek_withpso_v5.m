%% ================= 模型性能评估 (v5.1 Hybrid | 核心生长季 4-10月) =================
% 目标：基于 Final_Model_Huaihe_v5 (12通道) 进行全量测试集评估
% 更新内容：
%   1. 全局评估(Global Metrics)
%   2. 最佳像元挖掘(Best Pixel Analysis)
%   3. [新增] 空间误差分布图 (Spatial Error Map)
% 更新时间：2025-12-05
clc; clear; close all;

%% ================= 0. 基础配置 =================
rng(42); 

scriptDir = fileparts(mfilename('fullpath'));
codeDir = fileparts(scriptDir);
projectRoot = fileparts(codeDir);
addpath(genpath(fullfile(projectRoot, 'code')));

% 路径配置 (请确保路径正确)
baseFactorDir = fullfile(projectRoot, "data");
baseTrainDir  = fullfile(projectRoot, "models");

% 1. 模型文件 (v5)
modelFile = fullfile(baseTrainDir, "Final_Model_Huaihe_v5.mat"); 
% 2. 数据文件 (v5)
dataFile  = fullfile(baseFactorDir, "Ready_for_Train_3D_Huaihe_v5.mat");
% 3. 地理坐标文件
geoFile   = fullfile(baseFactorDir, "Pixel_Data_GPP.mat");

%% ================= 1. 加载资源 =================
fprintf('Status: 正在加载 v5 模型与数据...\n');

if ~exist(modelFile, 'file') || ~exist(dataFile, 'file') || ~exist(geoFile, 'file')
    error('❌ 文件缺失！请检查路径:\n%s\n%s', modelFile, dataFile);
end

% 加载模型与参数
load(modelFile, 'net', 'ps_in', 'ps_out'); 
% 加载测试数据
load(dataFile, 'x_test', 'y_test', 'idx_Test_Global', 'times');
% 加载地理信息
load(geoFile, 'Anchor_Lat', 'Anchor_Lon');

[nFeat, nTime, nTest] = size(x_test);
fprintf('  -> 测试集样本数: %d\n', nTest);
fprintf('  -> 特征通道数: %d\n', nFeat);

%% ================= 2. 执行预测 (显存保护模式) =================
fprintf('Status: 正在进行全量预测 (BatchSize=256)...\n');

% 1. 预处理
x_test_2D = reshape(x_test, nFeat, []);
x_test_2D = mapminmax('apply', x_test_2D, ps_in); 
x_test_Norm_3D = reshape(single(x_test_2D), nFeat, nTime, nTest);

% 2. 转换为 Cell
x_cell = cell(nTest, 1);
for k = 1:nTest, x_cell{k} = x_test_Norm_3D(:, :, k); end

% 3. 预测
try
    Y_Pred_Cell = predict(net, x_cell, 'MiniBatchSize', 256, 'ExecutionEnvironment', 'gpu');
catch
    fprintf('⚠️ GPU 显存不足，自动切换至 CPU 模式...\n');
    Y_Pred_Cell = predict(net, x_cell, 'MiniBatchSize', 256, 'ExecutionEnvironment', 'cpu');
end

% 4. 整理结果
if iscell(Y_Pred_Cell), Y_Pred_Norm = cat(3, Y_Pred_Cell{:}); else, Y_Pred_Norm = Y_Pred_Cell; end

% 5. 反归一化
Y_Pred_Real = reshape(mapminmax('reverse', reshape(Y_Pred_Norm, 1, []), ps_out), 1, nTime, nTest);
Y_True = y_test;

% 转换为矩阵方便计算 [Time x Pixels]
Mat_Pred = squeeze(Y_Pred_Real); 
Mat_True = squeeze(Y_True);

%% ================= 3. 空间格局评估 (含误差矩阵计算) =================
fprintf('Status: 正在评估空间格局 (Spatial Pattern)...\n');
Err_Mat  = Mat_Pred - Mat_True; % 预测 - 观测

% 空间 R2 计算
sse_spatial = sum(Err_Mat.^2, 2);
sst_spatial = sum((Mat_True - mean(Mat_True, 2)).^2, 2); 
r2_spatial = 1 - (sse_spatial ./ sst_spatial);

% 寻找最佳月份 (用于空间绘图)
if exist('times', 'var')
    month_list = month(times);
else
    month_list = repmat(1:12, 1, ceil(nTime/12)); month_list = month_list(1:nTime)';
end
% 仅在生长季(4-10月)内寻找空间表现最好的月份
gs_mask = (month_list >= 4) & (month_list <= 10);
pool_r2 = r2_spatial; pool_r2(~gs_mask) = -Inf;
[max_spatial_r2, idx_best_spatial] = max(pool_r2);

% 提取该月份的真实日期用于标题
if exist('times', 'var')
    best_date_str = datestr(times(idx_best_spatial), 'yyyy-mm');
else
    best_date_str = sprintf('Time Index %d', idx_best_spatial);
end

%% ================= 4. 全局综合评估 (Global Metrics) =================
fprintf('Status: 正在计算全局综合指标 (Global R2 & RMSE)...\n');

% 将所有像元、所有时间展平为一维向量
Vec_True = Mat_True(:);
Vec_Pred = Mat_Pred(:);

% 1. Global RMSE
global_mse = mean((Vec_Pred - Vec_True).^2);
global_rmse = sqrt(global_mse);

% 2. Global R2
global_sse = sum((Vec_Pred - Vec_True).^2);
global_sst = sum((Vec_True - mean(Vec_True)).^2);
global_r2  = 1 - (global_sse / global_sst);

fprintf('\n==================================================\n');
fprintf('🌍 Global Performance (Overall)\n');
fprintf('==================================================\n');
fprintf('   - Global R^2  : %.4f ⭐\n', global_r2);
fprintf('   - Global RMSE : %.4f gC/m2/mon\n', global_rmse);
fprintf('==================================================\n');

%% ================= 5. 最佳像元挖掘 (Temporal Analysis) =================
fprintf('Status: 正在挖掘最佳单点表现 (Temporal Best Pixel)...\n');

% 计算每个像元的时间维 R2
sse_time = sum((Mat_Pred - Mat_True).^2, 1); 
sst_time = sum((Mat_True - mean(Mat_True, 1)).^2, 1);
r2_time_pixel = 1 - (sse_time ./ sst_time);

% 找到 R2 最高的像元索引
[best_pixel_r2, idx_best_pixel] = max(r2_time_pixel);

% 获取该点的经纬度
best_lat = Anchor_Lat(idx_Test_Global(idx_best_pixel));
best_lon = Anchor_Lon(idx_Test_Global(idx_best_pixel));

fprintf('📍 Best Pixel Info:\n');
fprintf('   - Index: %d\n', idx_best_pixel);
fprintf('   - Location: (Lat: %.2f, Lon: %.2f)\n', best_lat, best_lon);
fprintf('   - Time-Series R^2: %.4f\n', best_pixel_r2);

%% ================= 6. 可视化绘图 (核心修改部分) =================

% --- 图1: 空间格局综合图 (Observed | Predicted | Error) ---
% 宽度增加到 1500 以容纳 3 个子图
figure('Name', 'Spatial Pattern & Error', 'Color', 'w', 'Position', [50, 50, 1500, 350]);

% 准备绘图数据
Val_True = Mat_True(idx_best_spatial, :); 
Val_Pred = Mat_Pred(idx_best_spatial, :);
Val_Err  = Err_Mat(idx_best_spatial, :); % 误差数据
Test_Lon = Anchor_Lon(idx_Test_Global); 
Test_Lat = Anchor_Lat(idx_Test_Global);

% 统一所有图的 Marker 大小
mk_size = 15;

% Subplot 1: Observed
subplot(1,3,1); 
%figure 
scatter(Test_Lon, Test_Lat, mk_size, Val_True, 'filled'); 
title({['Observed GPP (' best_date_str ')']}, 'FontSize', 12); 
axis equal; grid on; box on; colorbar;
caxis([0, max(Val_True)]); % 锁定颜色范围

% Subplot 2: Predicted
subplot(1,3,2);
%figure 
scatter(Test_Lon, Test_Lat, mk_size, Val_Pred, 'filled'); 
title({sprintf('Predicted GPP (Spatial R^2=%.3f)', max_spatial_r2)}, 'FontSize', 12); 
axis equal; grid on; box on; colorbar;
caxis([0, max(Val_True)]); % 保持与观测值一致的量纲

% Subplot 3: Spatial Error (新增)
subplot(1,3,3); 
%figure 
scatter(Test_Lon, Test_Lat, mk_size, Val_Err, 'filled'); 
title({'Prediction Error (Pred - Obs)', 'Red: Over | Blue: Under'}, 'FontSize', 12); 
axis equal; grid on; box on; 
cb = colorbar; cb.Label.String = 'Error (gC/m^2/mon)';

% 【关键技巧】设置对称的颜色轴，使 0 (白色/黄色) 居中
err_limit = max(abs(Val_Err)); 
caxis([-err_limit, err_limit]); 
colormap(subplot(1,3,3), 'jet'); % 使用 Jet 颜色条 (Blue-Cyan-Yellow-Orange-Red)

% --- 图2: 最佳像元深度分析 (保持不变) ---
figure('Name', 'Best Pixel Analysis', 'Color', 'w', 'Position', [100, 200, 1000, 400]);

% 准备该像元的数据
ts_true = Mat_True(:, idx_best_pixel);
ts_pred = Mat_Pred(:, idx_best_pixel);
max_val = max([ts_true; ts_pred]) * 1.1;

% 子图1: 时间序列对比
subplot(1, 2, 1);
plot(times, ts_true, 'k-', 'LineWidth', 1.2, 'DisplayName', 'Observed'); hold on;
plot(times, ts_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
title({sprintf('Best Pixel Time Series (Lat: %.2f, Lon: %.2f)', best_lat, best_lon), ...
       sprintf('Temporal R^2 = %.3f', best_pixel_r2)}, 'FontSize', 11);
ylabel('GPP (gC/m^2/mon)'); xlabel('Time');
legend('Location', 'best'); grid on; box on;
ylim([0, max_val]);

% 子图2: 散点图
subplot(1, 2, 2);
scatter(ts_true, ts_pred, 30, 'MarkerFaceColor', [0.2, 0.4, 0.8], ...
    'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.5); 
hold on;
plot([0, max_val], [0, max_val], 'k--', 'LineWidth', 1.5);
pixel_rmse = sqrt(mean((ts_pred - ts_true).^2));
title({sprintf('Observed vs Predicted (Best Pixel)'), ...
       sprintf('R^2 = %.3f | RMSE = %.3f', best_pixel_r2, pixel_rmse)}, ...
       'FontSize', 11);
xlabel('Observed GPP'); ylabel('Predicted GPP'); 
axis square; grid on; box on;
xlim([0, max_val]); ylim([0, max_val]);

fprintf('\n✅ 绘图完成！请检查 "Spatial Pattern & Error" 窗口。\n');