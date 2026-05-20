%% ================= SHAP 空间格局：纯净生态驱动版 (v5.2) =================
% 目标：
% 1. 剔除 Lat/Lon 静态坐标，只关注动态环境因子。
% 2. 同类因子去冗余：针对 Precip, Temp, Soil 等，自动选择影响力最大的时间周期。
% 3. 绘制“最佳生态因子”的空间分布图 + 主导驱动分区图。
% 更新时间：2025-12-06
clc; clear; close all;

%% ================= 0. 配置与加载 =================
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
geoFile   = fullfile(baseFactorDir, "Pixel_Data_GPP.mat"); 

fprintf('Status: 正在加载 v5 模型与数据...\n');
if ~exist(modelFile, 'file'), error('❌ 未找到模型文件: %s', modelFile); end
if ~exist(dataFile, 'file'), error('❌ 未找到数据文件: %s', dataFile); end

load(modelFile, 'net', 'ps_in');
load(dataFile, 'x_test', 'idx_Test_Global', 'fullVarNames'); 
load(geoFile, 'Anchor_Lat', 'Anchor_Lon');

% 获取测试集像元的真实经纬度
Test_Lat = Anchor_Lat(idx_Test_Global);
Test_Lon = Anchor_Lon(idx_Test_Global);

fprintf('  -> 原始特征列表:\n'); disp(fullVarNames);

%% ================= 1. 空间计算策略 (Grid Sampling) =================
% 为了快速出图，建议使用采样 (step=2或4)，若要高清全图设为1
sample_step = 1; 
grid_idx = 1 : sample_step : length(idx_Test_Global);
n_map = length(grid_idx);

fprintf('Status: 构建地图样本集 (共 %d 个像元)...\n', n_map);
X_Map_Source = x_test(:, :, grid_idx);
Lat_Map = Test_Lat(grid_idx);
Lon_Map = Test_Lon(grid_idx);

% 预处理归一化
X_Cell_Map = cell(n_map, 1);
for k = 1:n_map
    temp_2d = X_Map_Source(:,:,k);
    temp_2d = mapminmax('apply', temp_2d, ps_in); 
    X_Cell_Map{k} = temp_2d;
end

% 计算基准值
Ref_Matrix = mean(cat(3, X_Cell_Map{:}), 3);

%% ================= 2. 计算 SHAP (全量特征) =================
fprintf('Status: 正在计算空间 SHAP 值 (请耐心等待)...\n');
% 输出维度: [n_map, n_features]
shap_map_raw = shapley_lstm_3d(net, X_Cell_Map, Ref_Matrix, fullVarNames);

%% ================= 3. 【核心逻辑】智能筛选最佳因子 =================
fprintf('Status: 正在进行“同类归并”筛选...\n');

% 计算每个特征的全局平均影响力 (用于评判谁是老大)
global_impact = mean(abs(shap_map_raw), 1);

% 定义物理类别关键词 (分组)
% 格式: {类别名称, 识别关键词列表}
groups = {
    'Atmosphere (VPD)',   {'vpd'};
    'Precipitation',      {'prec', 'rain'};  % 包含 Ante_Prec, prec
    'Temperature',        {'temp', 'tmp'};   % 包含 Ante_Temp, tmpmean
    'Soil Moisture',      {'soil'};          % 包含 Ante_soil, soil
    'Energy (PET)',       {'pet', 'rad'};
};

selected_feat_indices = [];
selected_feat_names   = {};
selected_feat_titles  = {};

for g = 1:size(groups, 1)
    group_name = groups{g, 1};
    keywords   = groups{g, 2};
    
    % 1. 找到该组涉及的所有特征索引
    group_member_idx = [];
    for i = 1:length(fullVarNames)
        f_name = fullVarNames{i};
        % 排除 Lat/Lon/Sin/Cos
        if contains(f_name, {'Lat','Lon','Sin','Cos'}, 'IgnoreCase', true), continue; end
        
        % 检查是否包含关键词
        for k = 1:length(keywords)
            if contains(f_name, keywords{k}, 'IgnoreCase', true)
                group_member_idx = [group_member_idx, i];
                break;
            end
        end
    end
    
    % 2. 如果该组有成员，选出 Impact 最大的那个 (Winner)
    if ~isempty(group_member_idx)
        impacts = global_impact(group_member_idx);
        [max_imp, best_sub_idx] = max(impacts);
        winner_global_idx = group_member_idx(best_sub_idx);
        winner_name = fullVarNames{winner_global_idx};
        
        fprintf('  -> [%s] 组优胜者: %s (Impact=%.4f)\n', group_name, winner_name, max_imp);
        
        selected_feat_indices(end+1) = winner_global_idx;
        selected_feat_names{end+1}   = winner_name;
        selected_feat_titles{end+1}  = sprintf('%s\n(%s)', group_name, winner_name);
    end
end

n_selected = length(selected_feat_indices);
fprintf('Status: 最终选定 %d 个核心生态因子进行绘图。\n', n_selected);

%% ================= 4. 绘图 A: 核心生态因子空间分布 =================
figure('Name', 'Ecological Drivers Spatial Pattern', 'Color', 'w', 'Position', [50, 50, 1400, 600]);

% 颜色条: 红蓝 (Red-White-Blue)
try cmap = redblue(64); catch, cmap = jet(64); end

% 自动布局计算
cols = ceil(n_selected / 2);
rows = 2;

for i = 1:n_selected
    subplot(rows, cols, i);
    
    % 提取对应特征的 SHAP 值
    feat_idx = selected_feat_indices(i);
    current_shap = shap_map_raw(:, feat_idx);
    
    scatter(Lon_Map, Lat_Map, 10, current_shap, 'filled');
    
    title(selected_feat_titles{i}, 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
    axis equal; 
    xlim([min(Lon_Map)-0.1, max(Lon_Map)+0.1]); 
    ylim([min(Lat_Map)-0.1, max(Lat_Map)+0.1]);
    
    % 颜色设置
    max_val = max(abs(current_shap));
    caxis([-max_val, max_val]); % 对称显示
    colormap(gca, cmap);
    colorbar;
    box on;
end
sgtitle('核心生态因子 SHAP 空间异质性 (排除经纬度干扰)', 'FontSize', 14);

%% ================= 5. 绘图 B: 生态驱动机制分区 (Ecological Regimes) =================
% 目标：在每个像元上，比较这就几个因子的绝对贡献，看谁是老大
figure('Name', 'Dominant Ecological Regimes', 'Color', 'w', 'Position', [100, 100, 900, 700]);

% 提取这几个精选因子的 SHAP 值矩阵
shap_selected = abs(shap_map_raw(:, selected_feat_indices));
[~, max_k] = max(shap_selected, [], 2); % 找每行的最大值索引

% 配色方案 (给每个因子一种颜色)
regime_colors = lines(n_selected); 

hold on;
for k = 1:n_selected
    mask = (max_k == k);
    if sum(mask) > 0
        scatter(Lon_Map(mask), Lat_Map(mask), 12, regime_colors(k,:), 'filled', ...
            'DisplayName', selected_feat_names{k});
    end
end

axis equal;
xlim([min(Lon_Map)-0.1, max(Lon_Map)+0.1]);
ylim([min(Lat_Map)-0.1, max(Lat_Map)+0.1]);
title({'主导生态驱动分区图 (Dominant Ecological Regimes)'; 'Based on Max |SHAP| of Selected Factors'}, 'FontSize', 14);
legend('show', 'Location', 'eastoutside', 'FontSize', 12, 'Interpreter', 'none');
xlabel('Longitude'); ylabel('Latitude');
grid on; box on;

fprintf('✅ 绘图完成！请重点观察 "Dominant Ecological Regimes" 分区图。\n');
%% ================= 6. 数据导出 (ArcGIS Pro 专用接口) =================
% 目标：将主导生态因子分区结果编码并导出 CSV
% Output path: outputs/spatial_shap
% 编码：1=VPD, 2=Prec, 3=Temp, 4=Soil, 5=PET

fprintf('\nStatus: 正在初始化数据导出模块...\n');

% 1. 设置输出路径
saveDir = fullfile(projectRoot, "outputs", "spatial_shap");
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
    fprintf('  -> 检测到目录不存在，已创建: %s\n', saveDir);
else
    fprintf('  -> 输出目录已就绪: %s\n', saveDir);
end

% 2. 建立编码映射机制 (Robust Mapping)
% 不假设 selected_feat_names 的顺序，而是根据名称内容动态分配 Code
% max_k 是第5节计算出来的每个像元的主导因子索引 (对应 selected_feat_names)

n_pixels = length(max_k);

% 提取最大 SHAP 值 (Strength)
% shap_selected 在第5节已计算: abs(shap_map_raw(:, selected_feat_indices))
[Pixel_Strength, ~] = max(shap_selected, [], 2);

fprintf('  -> 正在进行栅格编码转换 (Total Pixels: %d)...\n', n_pixels);

% 遍历选出的几个核心因子，建立索引->编码的查找表
Map_Index_to_Code = zeros(n_selected, 1);
Map_Index_to_Group = cell(n_selected, 1);

for i = 1:n_selected
    fname = selected_feat_names{i};
    
    % 核心关键词匹配逻辑
    if contains(fname, 'vpd', 'IgnoreCase', true)
        code = 1; gname = 'Atmosphere';
    elseif contains(fname, {'prec', 'rain'}, 'IgnoreCase', true)
        code = 2; gname = 'Precipitation';
    elseif contains(fname, {'temp', 'tmp'}, 'IgnoreCase', true)
        code = 3; gname = 'Temperature';
    elseif contains(fname, 'soil', 'IgnoreCase', true)
        code = 4; gname = 'Soil_Moisture';
    elseif contains(fname, {'pet', 'rad'}, 'IgnoreCase', true)
        code = 5; gname = 'Energy';
    else
        code = 99; gname = 'Other'; % 兜底
    end
    
    Map_Index_to_Code(i) = code;
    Map_Index_to_Group{i} = gname;
    
    fprintf('     - Mapping: Index %d (%s) -> Code %d (%s)\n', ...
        i, fname, code, gname);
end

% 3. 应用映射到所有像元
Pixel_Codes      = Map_Index_to_Code(max_k);
Pixel_GroupNames = Map_Index_to_Group(max_k);
Pixel_FeatNames  = selected_feat_names(max_k)'; % 转置以匹配列向量

% 4. 构建 Table
T_export = table();
T_export.Longitude = Lon_Map(:);
T_export.Latitude  = Lat_Map(:);
T_export.Zone_Code = Pixel_Codes;       % 用于 ArcGIS 唯一值渲染 (1-5)
T_export.Zone_Group = Pixel_GroupNames; % 用于图例标签
T_export.Feature_Name = Pixel_FeatNames;% 具体特征名 (如 Ante_Temp_3m)
T_export.Dominance_Str = Pixel_Strength;% 主导强度 (SHAP绝对值)

% 5. 写入文件
%csvFileName = fullfile(saveDir, 'Huaihe_Ecological_Regimes.csv');
%writetable(T_export, csvFileName);

%fprintf('✅ 导出成功！文件已保存至:\n   %s\n', csvFileName);

%% ================= 7. 导出各单因子 SHAP 空间数据 (独立 CSV 版) =================
% 目标：将每个核心因子的空间分布数据分别导出为一个独立的 CSV 文件
% Output path: outputs/geo_factors

fprintf('\nStatus: 正在初始化批量导出模块...\n');

% 1. 设置输出路径 (自动创建文件夹)
% 注意：MATLAB 中路径字符串建议使用双引号
saveDir = fullfile(projectRoot, "outputs", "geo_factors");

if ~exist(saveDir, 'dir')
    mkdir(saveDir);
    fprintf('  -> 检测到目录不存在，已自动创建:\n     %s\n', saveDir);
else
    fprintf('  -> 输出目录已就绪:\n     %s\n', saveDir);
end

% 2. 循环导出每个因子
% n_selected 和 selected_feat_indices 来自脚本第3节的筛选结果
for i = 1:n_selected
    idx = selected_feat_indices(i);    % 获取该因子在原始 shap_map_raw 中的列索引
    raw_name = selected_feat_names{i}; % 获取原始特征名 (如 'vpd', 'Ante_prec_5')
    
    % 规范化文件名 (去除特殊字符，防止文件名报错)
    valid_name = matlab.lang.makeValidName(raw_name); 
    
    % 3. 构建单因子表格 (只包含：经度、纬度、该因子的SHAP值)
    T_Single = table();
    T_Single.Longitude = Lon_Map(:);
    T_Single.Latitude  = Lat_Map(:);
    
    % 提取该特征所有像元的 SHAP 值
    shap_values = shap_map_raw(:, idx);
    T_Single.(valid_name) = shap_values;
    
    % 4. 写入独立 CSV 文件
    csvFileName = fullfile(saveDir, sprintf('%s.csv', valid_name));
    writetable(T_Single, csvFileName);
    
    fprintf('  -> [%d/%d] 已导出文件: %s.csv\n', i, n_selected, valid_name);
end

fprintf('✅ 所有因子已分别导出完毕！\n');
fprintf('------------------------------------------------------\n');
fprintf('2. 对每个 CSV 右键 -> "Display XY Data" (X=Longitude, Y=Latitude)。\n');
fprintf('3. 使用 "IDW" 或 "Kriging" 工具对生成的点图层进行插值，Z值字段即为文件名对应的变量。\n');
fprintf('------------------------------------------------------\n');