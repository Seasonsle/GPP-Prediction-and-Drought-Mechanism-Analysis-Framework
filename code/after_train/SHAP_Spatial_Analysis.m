%% ================= SHAP 机制分析 (排除地理坐标版) =================
% 目标：分析气象/生态因子的贡献排序，排除 Lat/Lon 静态坐标干扰
% 对应模型：Final_Model_Huaihe_v5.mat
% 更新时间：2025-12-06
clc; clear; close all;

%% ================= 0. 设置与路径 =================
rng(42); 

scriptDir = fileparts(mfilename('fullpath'));
codeDir = fileparts(scriptDir);
projectRoot = fileparts(codeDir);
addpath(genpath(fullfile(projectRoot, 'code')));
baseFactorDir = fullfile(projectRoot, "data");
baseTrainDir  = fullfile(projectRoot, "models"); 

% 指定 v5 模型与数据
modelFile = fullfile(baseTrainDir, "Final_Model_Huaihe_v5.mat");
dataFile  = fullfile(baseFactorDir, "Ready_for_Train_3D_Huaihe_v5.mat");

fprintf('Status: 正在加载 v5 模型与数据...\n');
if ~exist(modelFile, 'file'), error('❌ 未找到模型文件: %s', modelFile); end
if ~exist(dataFile, 'file'), error('❌ 未找到数据文件: %s', dataFile); end

load(dataFile, 'x_test', 'fullVarNames'); 
load(modelFile, 'net', 'ps_in'); 

fprintf('  -> 原始特征列表:\n'); disp(fullVarNames);

%% ================= 1. 样本抽样 (加速计算) =================
n_samples_shap = 600; % 样本量，根据显存调整
total_test = size(x_test, 3);
if total_test < n_samples_shap, n_samples_shap = total_test; end

rand_idx = randperm(total_test, n_samples_shap);
fprintf('Status: 随机抽取 %d 个测试集样本进行解释...\n', n_samples_shap);

X_Source = x_test(:, :, rand_idx);
X_Cell_Shap = cell(n_samples_shap, 1);

% 预处理：归一化
for k = 1:n_samples_shap
    temp_2d = X_Source(:,:,k);
    temp_2d = mapminmax('apply', temp_2d, ps_in); 
    X_Cell_Shap{k} = temp_2d;
end

% 计算背景参考矩阵 (Background Reference)
Ref_Matrix = mean(cat(3, X_Cell_Shap{:}), 3);

%% ================= 2. 计算 SHAP (包含所有特征) =================
% 注意：模型预测需要所有特征，所以这里必须先算全量的 SHAP，再在绘图时剔除
fprintf('Status: 正在计算 SHAP 值 (可能需要几分钟)...\n');
shap_results = shapley_lstm_3d(net, X_Cell_Shap, Ref_Matrix, fullVarNames);

%% ================= 3. 结果整理 (【关键修改】排除 Lat/Lon) =================
fprintf('Status: 正在过滤特征，排除 Lat/Lon...\n');

% 定义要排除的特征关键词 (大小写不敏感)
% 这里涵盖了 Lat, Lon 以及你提到的 LAI (如果存在)
exclude_keywords = {'Lat', 'Lon', 'LAI', 'Latitude', 'Longitude'}; 

% 查找需要保留的特征索引
keep_mask = true(size(fullVarNames)); % 初始化全选
for i = 1:length(fullVarNames)
    for j = 1:length(exclude_keywords)
        % 如果特征名包含排除关键词（如 'Lat'），则标记为 false
        if contains(fullVarNames{i}, exclude_keywords{j}, 'IgnoreCase', true)
            keep_mask(i) = false;
            fprintf('  -> 已剔除特征: %s\n', fullVarNames{i});
            break; 
        end
    end
end

% 仅保留气象/土壤因子
shap_results_plot = shap_results(:, keep_mask); 
feat_names_plot   = fullVarNames(keep_mask);
n_show = length(feat_names_plot);

fprintf('Status: 最终参与排序的特征数: %d\n', n_show);
disp(feat_names_plot);

%% ================= 4. 可视化 (增强版：修复灰色匹配问题) =================
fprintf('Status: 正在绘制图表 (已修复 soil 颜色匹配)...\n');

% --- 4.0 定义配色 (完全来自 A3Dshiyitu.m) ---
bright_colors = [
    0.2, 0.6, 1.0;  % 1. prec (蓝色)
    0.2, 0.8, 0.2;  % 2. petPM (绿色)
    1.0, 0.6, 0.2;  % 3. tmpmean (橙色)
    1.0, 0.3, 0.3;  % 4. soil_moisture (红色/淡红) <--- 目标颜色
    0.6, 0.4, 0.8;  % 5. vpd (紫色)
    0.2, 0.8, 0.8;  % 6. Ante_prec_3m (青色)
    1.0, 0.8, 0.0;  % 7. Ante_prec_5m (金黄)
    0.8, 0.2, 0.6;  % 8. Ante_sm_3m (玫红)
    0.4, 1.0, 0.6;  % 9. Ante_sm_5m (淡青绿)
    0.6, 0.6, 0.6;  % 10. Ante_tmp_3m (灰色)
    0.4, 0.4, 1.0;  % 11. Lat
    0.8, 0.4, 0.0;  % 12. Lon
    0.0, 0.7, 0.7;  % 13. Tcos
    0.9, 0.1, 0.5   % 14. Tsin
];

% --- 4.1 建立容错映射表 ---
% 这里的 Key 是您数据中可能出现的“原始变量名”，Value 是颜色索引
nameMap = containers.Map();
colorIdxMap = containers.Map();

% 辅助函数：快速添加映射 (支持多个别名指向同一个颜色)
add_map = @(keys, dispName, cIdx) add_mapping(nameMap, colorIdxMap, keys, dispName, cIdx);

% === 1. prec ===
add_map({'prec', 'precipitation'}, 'prec', 1);

% === 2. petPM ===
add_map({'petPM', 'pet', 'PET'}, 'petPM', 2);

% === 3. tmpmean ===
add_map({'tmpmean', 'tmp', 'temp'}, 'tmpmean', 3);

% === 4. soil_moisture (修复重点！) ===
% 无论原始名是 'soil' 还是 'soil_moisture'，都指向颜色 4
add_map({'soil_moisture', 'soil', 'sm', 'SOIL'}, 'soil_moisture', 4); 

% === 5. vpd ===
add_map({'vpd', 'VPD'}, 'vpd', 5);

% === 6. Ante_prec_3 ===
add_map({'Ante_prec_3', 'Ante_prec_3m'}, 'Ante_prec_3m', 6);

% === 7. Ante_prec_5 ===
add_map({'Ante_prec_5', 'Ante_prec_5m'}, 'Ante_prec_5m', 7);

% === 8. Ante_sm_3 ===
add_map({'Ante_sm_3', 'Ante_sm_3m', 'Ante_soil_3', 'Ante_soil_3m'}, 'Ante_sm_3m', 8);

% === 9. Ante_sm_5 ===
add_map({'Ante_sm_5', 'Ante_sm_5m', 'Ante_soil_5', 'Ante_soil_5m'}, 'Ante_sm_5m', 9);

% === 10. Ante_tmp_3 ===
add_map({'Ante_tmp_3', 'Ante_tmp_3m', 'Ante_temp_3', 'Ante_temp_3m'}, 'Ante_tmp_3m', 10);

% === 13/14 时间周期 ===
add_map({'Tcos'}, 'Tcos', 13);
add_map({'Tsin'}, 'Tsin', 14);


% --- 4.2 数据计算 ---
global_impact = mean(abs(shap_results_plot), 1); 
total_impact_sum = sum(global_impact); 
global_impact_pct = (global_impact / total_impact_sum) * 100;

[sorted_pct, idx] = sort(global_impact_pct, 'ascend');
sorted_raw_names = feat_names_plot(idx); 

% --- 4.3 生成绘图属性 (带 Debug 信息) ---
sorted_display_names = cell(size(sorted_raw_names));
sorted_colors = zeros(length(sorted_raw_names), 3);

fprintf('\n--- 颜色匹配检查 ---\n');
for i = 1:length(sorted_raw_names)
    raw_name = sorted_raw_names{i};
    
    % 尝试精确匹配
    if isKey(nameMap, raw_name)
        sorted_display_names{i} = nameMap(raw_name);
        c_idx = colorIdxMap(raw_name);
        sorted_colors(i, :) = bright_colors(c_idx, :);
        fprintf('  ✅ "%s" -> 配色索引 %d (成功)\n', raw_name, c_idx);
    else
        % 如果失败，尝试模糊匹配 (比如包含 'soil' 就算)
        if contains(raw_name, 'soil', 'IgnoreCase', true)
            sorted_display_names{i} = ['4. ', raw_name];
            sorted_colors(i, :) = bright_colors(4, :); % 强制红色
            fprintf('  ⚠️ "%s" -> 模糊匹配到 soil (红色)\n', raw_name);
        elseif contains(raw_name, 'prec', 'IgnoreCase', true) && contains(raw_name, '5')
             sorted_display_names{i} = ['7. ', raw_name];
             sorted_colors(i, :) = bright_colors(7, :); 
             fprintf('  ⚠️ "%s" -> 模糊匹配到 Ante_prec_5 (金黄)\n', raw_name);
        elseif contains(raw_name, 'tmp', 'IgnoreCase', true) && contains(raw_name, '3')
             sorted_display_names{i} = ['10. ', raw_name];
             sorted_colors(i, :) = bright_colors(10, :); 
             fprintf('  ⚠️ "%s" -> 模糊匹配到 Ante_tmp_3 (灰色)\n', raw_name);
        else
            % 彻底失败
            sorted_display_names{i} = raw_name;
            sorted_colors(i, :) = [0.5 0.5 0.5]; % 灰色
            fprintf('  ❌ "%s" -> 未找到匹配，使用灰色\n', raw_name);
        end
    end
end

% --- 图1: 全局特征重要性 (条形图 - 百分比) ---
f1 = figure('Name', 'Global Importance (%)', 'Color', 'w', 'Position', [100, 100, 700, 500]);
b = barh(sorted_pct);
b.FaceColor = 'flat'; 
b.CData = sorted_colors; 

for i = 1:n_show
    text(sorted_pct(i) + 0.2, i, sprintf('%.1f%%', sorted_pct(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
end

yticks(1:n_show);
yticklabels(sorted_display_names);
xlabel('Relative Importance (%)', 'FontName', 'Times New Roman'); 
title({'Ecological Drivers Ranking'; '(Percentage Contribution)'}, 'FontSize', 12, 'FontName', 'Times New Roman');
grid on; ax = gca; ax.FontSize = 11; ax.FontName = 'Times New Roman';
ax.XLim = [0, max(sorted_pct) * 1.15]; 
ax.TickLength = [0.005, 0.005]; 

% --- 图2: 蜂群图 (原始单位) ---
f2 = figure('Name', 'SHAP Beeswarm', 'Color', 'w', 'Position', [800, 100, 800, 600]);
hold on; xline(0, 'k--', 'Alpha', 0.5);
colormap(jet); 
colors = colormap; 

for i = 1:n_show
    current_feat_idx_plot = idx(i);
    shap_vals = shap_results_plot(:, current_feat_idx_plot);
    
    % 获取原始值
    current_raw_name = feat_names_plot{current_feat_idx_plot};
    orig_idx = find(strcmp(fullVarNames, current_raw_name));
    raw_vals = zeros(n_samples_shap, 1);
    for k=1:n_samples_shap
        raw_vals(k) = mean(X_Cell_Shap{k}(orig_idx, :));
    end
    
    % 上色
    min_v = min(raw_vals); max_v = max(raw_vals);
    if max_v == min_v, max_v = min_v + 1e-6; end 
    norm_raw = (raw_vals - min_v) / (max_v - min_v);
    color_indices = round(norm_raw * (size(colors,1)-1)) + 1;
    point_colors = colors(color_indices, :);
    
    jitter = (rand(size(shap_vals)) - 0.5) * 0.45;
    y_pos = i + jitter;
    
    scatter(shap_vals, y_pos, 20, point_colors, 'filled', 'MarkerEdgeAlpha', 0.2);
end

yticks(1:n_show);
yticklabels(sorted_display_names);
xlabel('SHAP Value (Impact on GPP: g C m^{-2} mon^{-1})', 'FontWeight', 'bold', 'FontName', 'Times New Roman'); 
title({'Feature Contributions (Beeswarm Plot)'; 'Raw Impact Units'}, 'FontSize', 12, 'FontName', 'Times New Roman');

c = colorbar; 
c.Label.String = 'Feature Value (Low --> High)';
c.Ticks = [0, 1]; c.TickLabels = {'Low', 'High'};
c.Label.FontName = 'Times New Roman';
grid on; ax = gca; ax.FontSize = 11; ax.FontName = 'Times New Roman';
ax.TickLength = [0.005, 0.005];

fprintf('✅ 可视化修复完成：检查命令行输出以确认所有变量都已匹配颜色。\n');

% === 辅助函数定义 (必须放在脚本最后) ===
function add_mapping(nMap, cMap, keys, dispName, cIdx)
    for i = 1:length(keys)
        nMap(keys{i}) = dispName;
        cMap(keys{i}) = cIdx;
    end
end