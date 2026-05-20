%% ========================================================================
%  Script: Huaihe_Resilience_MultiFactor_Mechanism.m
%  Purpose: 
%    1. 多因素综合：同时分析 Prec(5m), Soil(5m), VPD, Temp(3m)
%    2. 滞后体现：直接从 v5 数据集提取 Lag 特征，而非原始数据
%    3. 归一化绘图：使用百分位 (Percentile) 将不同单位因子绘制在同一X轴
% =========================================================================

clc; clear; close all;

scriptDir = fileparts(mfilename('fullpath'));
codeDir = fileparts(scriptDir);
projectRoot = fileparts(codeDir);
addpath(genpath(fullfile(projectRoot, 'code')));

%% 1. 路径配置
% v5 数据集 (包含已计算好的滞后特征)
trainFile = fullfile(projectRoot, "data", "Ready_for_Train_3D_Huaihe_v5.mat");
% 恢复力结果
resFile   = fullfile(projectRoot, "outputs", "Resilience_Lite_Result.mat");

%% 2. 提取多维环境特征 (Feature Extraction)
fprintf('Step 1: 从 v5 数据集提取多维滞后特征...\n');
if ~isfile(trainFile), error('未找到 v5 数据集'); end

% 使用 matfile 为了节省内存
m = matfile(trainFile);

% 提取 Train 和 Test 数据并合并 (还原全流域空间)
% 注意: 你的 v5 数据结构是 [Channel, Time, Pixel]
xtr = m.x_train; 
xte = m.x_test;
X_Full = cat(3, xtr, xte); % 合并像素维度
clear xtr xte;

% 定义我们要分析的通道 (根据你的 data_Fusion 代码逻辑)
% Ch 5: vpd (大气需水)
% Ch 7: Ante_Prec_5m (长期水文供给 - 核心修正!)
% Ch 9: Ante_Soil_5m (长期土壤缓冲)
% Ch 10: Ante_Temp_3m (中期热量累积)
% Ch 11-12: Lat, Lon (用于对齐)

target_channels = [5, 7, 9, 10]; 
feature_names   = {'VPD (Atmos Demand)', 'Ante-Prec-5m (Hydro Supply)', ...
                   'Ante-Soil-5m (Buffer)', 'Ante-Temp-3m (Heat)'};
colors          = {'#D95319', '#0072BD', '#77AC30', '#A2142F'}; % 橙, 蓝, 绿, 红

% 提取并计算“多年平均气候态” (Climatology)
% 对 Time 维度 (dim 2) 求平均 -> 得到每个像素的背景特征
Feat_Matrix = squeeze(mean(X_Full(target_channels, :, :), 2))'; % 转置为 [Pixel x Feature]

% 提取对应的坐标
Lat_Clim = squeeze(mean(X_Full(11, :, :), 2));
Lon_Clim = squeeze(mean(X_Full(12, :, :), 2));
Coords_Clim = double([Lat_Clim, Lon_Clim]);

clear X_Full; 
fprintf('   特征提取完成。维度: %d 像素 x %d 特征\n', size(Feat_Matrix));

%% 3. 空间对齐 (Spatial Matching)
fprintf('Step 2: 与恢复力结果进行空间对齐...\n');
load(resFile, 'Mean_Recovery_Time', 'Lat_Lite', 'Lon_Lite');

% 清洗恢复力数据
valid_mask = ~isnan(Mean_Recovery_Time) & Mean_Recovery_Time > 0 & Mean_Recovery_Time < 12;
Res_Data = double(Mean_Recovery_Time(valid_mask));
Coords_Res = double([Lat_Lite(valid_mask), Lon_Lite(valid_mask)]);

% KNN 匹配
[idx, dist] = knnsearch(Coords_Clim, Coords_Res);

% 过滤匹配度差的点
good_match = dist < 0.05;
Final_Res = Res_Data(good_match);
Final_Feats = Feat_Matrix(idx(good_match), :);

fprintf('   最终有效分析样本: %d\n', length(Final_Res));

%% 4. 绘图：多因素敏感度分析 (Sensitivity Plot)
fprintf('Step 3: 绘制综合机制图...\n');

figure('Color', 'w', 'Position', [100, 100, 900, 600]);
hold on;

legend_handles = [];
smooth_window = 5; % 曲线平滑窗口

% 为了在同一张图上画不同单位的变量，我们将 X 轴标准化为 "Percentile (0-100%)"
% 0% = 该因子在该流域的最小值，100% = 该因子在该流域的最大值
x_percentiles = 5:5:95; % 采样点 (5% 到 95%)

for i = 1:length(target_channels)
    raw_feat = Final_Feats(:, i);
    
    % 分箱统计
    % 使用 prctile 动态确定分箱边界，确保每箱样本数一致
    edges = prctile(raw_feat, 0:5:100); % 20个分箱
    [~, ~, bins] = histcounts(raw_feat, edges);
    
    y_means = [];
    y_errs  = [];
    x_centers = [];
    
    for k = 1:20
        idx_bin = bins == k;
        if sum(idx_bin) > 50
            y_means(end+1) = mean(Final_Res(idx_bin));
            y_errs(end+1)  = std(Final_Res(idx_bin)) / sqrt(sum(idx_bin)) * 1.96; % 95% CI
            x_centers(end+1) = (k-1)*5 + 2.5; % 用百分位作为 X 坐标
        end
    end
    
    % 绘制带误差阴影的曲线 (Shaded Error Bar)
    % 或者简单点，绘制粗线 + 误差棒
    h = errorbar(x_centers, y_means, y_errs, 'o-', ...
        'Color', colors{i}, 'LineWidth', 2, 'MarkerFaceColor', 'w', 'CapSize', 0);
    
    legend_handles(end+1) = h;
    
    % 计算斜率 (敏感度)
    p = polyfit(x_centers, y_means, 1);
    slope_val = p(1) * 100; % 换算为：每 100% 变化导致的月数变化
    
    % 在线条末端添加文字标注
    text(x_centers(end), y_means(end), sprintf('Slope=%.2f', slope_val), ...
        'Color', colors{i}, 'FontSize', 10, 'FontWeight', 'bold');
end

%% 5. 美化与逻辑增强
grid on; box on;

% 动态调整 Y 轴 (解决“图像全在底部”的问题)
% 自动寻找数据的上下限，并留出 10% 的余量
all_y = Final_Res;
y_lower = prctile(all_y, 5) * 0.9; % 忽略极端低值
y_upper = prctile(all_y, 95) * 1.1; % 忽略极端高值
ylim([y_lower, y_upper]);

xlabel('Environmental Gradient (Percentile 0% - 100%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Mean Recovery Time (Months)', 'FontSize', 12, 'FontWeight', 'bold');
title({'Drivers of Ecosystem Resilience: Multi-Factor Gradient Analysis', ...
       'Comparison of Water Supply vs. Atmospheric Demand'}, 'FontSize', 14);

% 添加辅助区域说明
yl = ylim;
text(5, yl(2)*0.95, 'Low Stress / Low Value', 'FontSize', 10, 'Color', [0.5 0.5 0.5]);
text(85, yl(2)*0.95, 'High Stress / High Value', 'FontSize', 10, 'Color', [0.5 0.5 0.5]);

legend(legend_handles, feature_names, 'Location', 'best', 'FontSize', 11);

% 保存高清大图
%saveas(gcf, 'Figure_MultiFactor_Mechanism.png');
fprintf('绘图完成！\n');