%% ========================================================================
%  Script: Huaihe_Resilience_Lightweight.m
%  Purpose: 计算淮河流域生态恢复力 (16GB 内存优化版)
%  Strategy: 1. 强力降采样 (1/10)  2. 单精度计算 (Single Precision)
% =========================================================================

%% 1. 环境初始化
clc; clear; close all;
rng(42);

scriptDir = fileparts(mfilename('fullpath'));
codeDir = fileparts(scriptDir);
projectRoot = fileparts(codeDir);
addpath(genpath(fullfile(projectRoot, 'code')));

%% 2. 配置参数
baseFactorDir = fullfile(projectRoot, "data");
baseTrainDir  = fullfile(projectRoot, "models");
shpPath       = fullfile(projectRoot, "data", "huai_river_basin_boundary.shp");
geoFile       = fullfile(baseFactorDir, "Pixel_Data_GPP.mat");

% --- 【核心参数】内存安全阀 ---
% 设置为 10，意味着每 10 个像素只取 1 个计算。
% 配合 single 类型，内存占用约为原始的 1/20，16G 内存绝对跑得飞快。
SAMPLE_STEP = 10;  
% ---------------------------

%% 3. 数据加载与轻量化处理
fprintf('正在加载数据: %s ...\n', geoFile);
if ~isfile(geoFile), error('找不到文件'); end
load(geoFile);

% --- 智能变量识别与降采样 ---
fprintf('正在执行降采样 (步长=%d) 与类型转换...\n', SAMPLE_STEP);

if exist('Data_GPP', 'var')
    % 原始维度: [Pixels, Time]
    [nTotalPixels, ~] = size(Data_GPP);
    
    % 1. 生成采样索引
    sample_idx = 1:SAMPLE_STEP:nTotalPixels;
    
    % 2. 切片并转单精度 (转置为 Time x Pixel 以便后续计算)
    % 注意：先切片(Sample)再转类型，最大程度省内存
    GPP_Lite = single(Data_GPP(sample_idx, :))'; 
    
    % 3. 立即清除原始大变量
    clear Data_GPP;
    
elseif exist('GPP_Data_Full', 'var')
    % 原始维度: [Time, Pixels]
    [~, nTotalPixels] = size(GPP_Data_Full);
    sample_idx = 1:SAMPLE_STEP:nTotalPixels;
    
    GPP_Lite = single(GPP_Data_Full(:, sample_idx));
    clear GPP_Data_Full;
    
elseif exist('GPP_Data', 'var')
    % 原始维度: [Time, Pixels]
    [~, nTotalPixels] = size(GPP_Data);
    sample_idx = 1:SAMPLE_STEP:nTotalPixels;
    
    GPP_Lite = single(GPP_Data(:, sample_idx));
    clear GPP_Data;
else
    error('未找到 GPP 变量');
end

% --- 坐标同步降采样 ---
if exist('Anchor_Lat', 'var')
    Lat_Lite = Anchor_Lat(sample_idx);
    Lon_Lite = Anchor_Lon(sample_idx);
elseif exist('Lat_List', 'var')
    Lat_Lite = Lat_List(sample_idx);
    Lon_Lite = Lon_List(sample_idx);
else
    error('未找到坐标变量');
end

[nTime, nPixels] = size(GPP_Lite);
fprintf('内存优化完成。当前计算量: %d 个像元 (原始: %d)\n', nPixels, nTotalPixels);

%% 4. 计算距平 (Anomaly)
fprintf('正在计算距平 (Z-Score)...\n');
GPP_Anomaly = zeros(nTime, nPixels, 'single'); 

Month_Idx = repmat(1:12, 1, ceil(nTime/12));
Month_Idx = Month_Idx(1:nTime); 

for m = 1:12
    mask = (Month_Idx == m);
    % single 精度计算
    mu = mean(GPP_Lite(mask, :), 1, 'omitnan');
    sigma = std(GPP_Lite(mask, :), 0, 1, 'omitnan');
    sigma(sigma == 0) = NaN; 
    
    GPP_Anomaly(mask, :) = (GPP_Lite(mask, :) - mu) ./ sigma;
end

clear GPP_Lite; % 再次释放内存，只留 Anomaly

%% 5. 并行计算恢复时间 (TtR)
fprintf('正在计算恢复力指标 (Parfor)... \n');

Drought_Thresh  = -1.5; 
Recovery_Thresh = -0.5;

Mean_Recovery_Time = nan(nPixels, 1, 'single');
Event_Count = zeros(nPixels, 1, 'single');

% 自动管理并行池
if isempty(gcp('nocreate'))
    try
        % 限制 worker 数量为 4，防止每个 worker 占用太多内存
        parpool(4); 
    catch
        fprintf('并行池启动失败，将单核运行...\n');
    end
end

parfor i = 1:nPixels
    ts = GPP_Anomaly(:, i);
    
    if all(isnan(ts)), continue; end
    
    events_durations = [];
    in_drought = false;
    start_idx = 0;
    
    for t = 1:nTime
        val = ts(t);
        if isnan(val), continue; end
        
        if ~in_drought
            if val < Drought_Thresh
                in_drought = true;
                start_idx = t;
            end
        else
            % 如果正在干旱，检查是否恢复
            if val > Recovery_Thresh
                dur = t - start_idx;
                if dur > 0
                    events_durations = [events_durations; dur];
                end
                in_drought = false;
            end
        end
    end
    
    if ~isempty(events_durations)
        Mean_Recovery_Time(i) = mean(events_durations);
        Event_Count(i) = length(events_durations);
    end
end
fprintf('计算完成。\n');

%% 6. 可视化
figure('Color', 'w', 'Position', [100, 100, 800, 500]);

if exist(shpPath, 'file')
    try mapshow(shaperead(shpPath), 'FaceColor','none','EdgeColor',[0.5 0.5 0.5]); hold on; catch, end
end

% 绘制散点图
scatter(Lon_Lite, Lat_Lite, 15, Mean_Recovery_Time, 'filled', 's');
c = colorbar;
c.Label.String = 'Recovery Time (Months)';
c.Label.FontSize = 12;
colormap(flipud(jet));
caxis([0, 6]); % 锁定颜色范围便于对比

title(['Ecosystem Resilience (Step=', num2str(SAMPLE_STEP), ')']);
axis tight; box on; grid on;

% 保存轻量版结果
outputDir = fullfile(projectRoot, "outputs");
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
saveFile = fullfile(outputDir, 'Resilience_Lite_Result.mat');
save(saveFile, 'Mean_Recovery_Time', 'Event_Count', 'Lat_Lite', 'Lon_Lite', 'SAMPLE_STEP');
fprintf('结果已保存至: %s\n', saveFile);