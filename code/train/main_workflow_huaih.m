%% ================= 主程序：淮河流域 GPP 预测 (v5 Hybrid | PSO + 深度训练) =================
% 目标：结合 PSO 结构寻优 + 稳健的训练策略，彻底解决欠拟合问题
% 输入：Ready_for_Train_3D_Huaihe_v5.mat (12通道)
% 输出：Final_Model_Huaihe_v5.mat
% 更新时间：2025-12-05
clc; clear; close all;

rng(42); % 固定随机种子，确保结果可复现

scriptDir = fileparts(mfilename('fullpath'));
codeDir = fileparts(scriptDir);
projectRoot = fileparts(codeDir);
addpath(genpath(fullfile(projectRoot, 'code')));

%% ================= 1. 加载 v5 数据 =================
fprintf('Step 1: 加载 v5 数据集 (12通道)...\n');
% 请确保路径正确，如果不正确请修改此处
data_path = fullfile(projectRoot, "data", "Ready_for_Train_3D_Huaihe_v5.mat");

if exist(data_path, 'file')
    load(data_path, 'x_train', 'y_train', 'x_test', 'y_test');
    [nFeat, nTime, nTrain] = size(x_train);
    fprintf('特征维度: %d (确认是否为12)\n', nFeat);
    fprintf('训练样本数: %d, 时间步长: %d\n', nTrain, nTime);
else
    error('❌ 找不到文件: %s', data_path);
end

%% ================= 2. 归一化 (关键) =================
fprintf('Step 2: 归一化处理...\n');

% --- 处理 X (输入) ---
% 1. 展平训练集以计算统计量
X_Train_2D = reshape(x_train, nFeat, []); 
% 2. 计算归一化参数 (ps_in)
[X_Train_Norm_2D, ps_in] = mapminmax(X_Train_2D, 0, 1);
% 3. 还原训练集形状并转为 single
x_train = reshape(single(X_Train_Norm_2D), nFeat, nTime, nTrain);
clear X_Train_2D X_Train_Norm_2D; % 及时释放内存

% 4. 应用参数到测试集
x_test_2D = reshape(x_test, nFeat, []);
x_test_2D = mapminmax('apply', x_test_2D, ps_in);
x_test = reshape(single(x_test_2D), nFeat, nTime, size(x_test, 3));
clear x_test_2D;

% --- 处理 Y (输出) ---
% 1. 展平训练集
Y_Train_2D = reshape(y_train, 1, []);
% 2. 计算归一化参数 (ps_out)
[Y_Train_Norm_2D, ps_out] = mapminmax(Y_Train_2D, 0, 1);
% 3. 还原
y_train = reshape(single(Y_Train_Norm_2D), 1, nTime, nTrain);
clear Y_Train_2D Y_Train_Norm_2D;

% 4. 应用到测试集
y_test_2D = reshape(y_test, 1, []);
y_test_2D = mapminmax('apply', y_test_2D, ps_out);
y_test = reshape(single(y_test_2D), 1, nTime, size(y_test, 3));
clear y_test_2D;

fprintf('归一化完成。ps_in 和 ps_out 已保存用于后续反归一化。\n');

%% ================= Step 3: PSO 结构寻优 (调用 run_pso_optimization) =================
fprintf('Step 3: 启动 PSO 寻找最优 Hidden/Heads...\n');

% 检查当前目录下是否有 PSO 相关函数
if exist('run_pso_optimization', 'file') ~= 2
    error('❌ 未找到 run_pso_optimization.m！请确保它在当前文件夹。');
end

% 运行 PSO
% 这里会调用外部函数进行寻优
[opt_hidden, opt_heads] = run_pso_optimization(x_train, y_train, x_test, y_test);

fprintf('\n================ PSO 结果 ================\n');
fprintf('最优隐藏单元 (Hidden Units): %d\n', opt_hidden);
fprintf('最优注意力头 (Attention Heads): %d\n', opt_heads);
fprintf('==========================================\n');

%% ================= Step 4: 构建最终网络 (使用最优参) =================
fprintf('Step 4: 构建 LSTM-Transformer 混合模型...\n');

maxPosition = nTime;
att_dim_final = opt_heads * 16; % 强制维度对齐 (Heads * 16)

layers = [
    sequenceInputLayer(nFeat, 'Name', 'input')
    
    % LSTM 层 (使用 PSO 结果)
    lstmLayer(opt_hidden, 'OutputMode', 'sequence', 'Name', 'lstm')
    dropoutLayer(0.1, 'Name', 'drop1') % 防止过拟合
    
    % 位置编码与残差连接
    positionEmbeddingLayer(opt_hidden, maxPosition, 'Name', 'pos-emb')
    additionLayer(2, 'Name', 'add') 
    
    % 投影层 (Projection)
    fullyConnectedLayer(att_dim_final, 'Name', 'proj_att') 
    
    % Self-Attention (使用 PSO 结果)
    selfAttentionLayer(opt_heads, att_dim_final, 'AttentionMask', 'causal', 'Name', 'att1')
    layerNormalizationLayer('Name', 'norm1')
    
    % 前馈网络 (Feed Forward)
    fullyConnectedLayer(64, 'Name', 'ff1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'fc_out')
    regressionLayer('Name', 'output')
];

lgraph = layerGraph(layers);
% 连接残差边：LSTM 输出 -> Add 层
lgraph = connectLayers(lgraph, 'drop1', 'add/in2');

%% ================= Step 5: 深度全量训练 (核心保障) =================
fprintf('Step 5: 配置训练参数...\n');

% 定义 Datastore 读取函数
readFcn_Train = @(idx) getBatch(x_train, y_train, idx);
ds_Train = transform(arrayDatastore(1:nTrain), readFcn_Train);

readFcn_Test = @(idx) getBatch(x_test, y_test, idx);
ds_Test = transform(arrayDatastore(1:size(x_test,3)), readFcn_Test);

% 计算每个 Epoch 有多少次迭代 (用于设置 ValidationFrequency)
miniBatchSize = 512;
valFreq = floor(nTrain / miniBatchSize); 

options = trainingOptions('adam', ...                % 优化器选择Adam
    'MaxEpochs', 16, ...                           % 最大训练轮数
    'MiniBatchSize', 256, ...                        % 批次数大小
    'ValidationData', ds_Test, ...                     % 验证集
    'ValidationFrequency', valFreq, ...                 % [新增] 每个 Epoch 验证一次
    'InitialLearnRate', 0.001, ...                   % 初始学习率
    'LearnRateDropFactor', 0.1, ...                  % 学习率下降因子 0.1
    'LearnRateDropPeriod', 800, ...                  % 经过xx次训练后 学习率为 0.01 * 0.1
    'L2Regularization', 0.001,...                    % 正则化系数
    'GradientThreshold', 10, ...                     % 梯度裁剪阈值
    'ExecutionEnvironment', "auto", ...              % 执行环境选择，优先使用GPU，如果不可用则使用CPU
    'Verbose', true, ...                                % 关闭优化过程
    'Plots', 'training-progress');                   % 画出曲线


fprintf('开始全量深度训练 (MaxEpochs=16, Batch=256)...\n');
[net, info] = trainNetwork(ds_Train, lgraph, options);

%% ================= Step 6: 保存模型与参数 =================
modelDir = fullfile(projectRoot, "models");
if ~exist(modelDir, 'dir'), mkdir(modelDir); end
save_file_name = fullfile(modelDir, 'Final_Model_Huaihe_v5.mat');
fprintf('训练完成，正在保存至: %s\n', save_file_name);

% 保存所有关键变量：网络、训练信息、PSO超参、归一化参数
save(save_file_name, 'net', 'info', 'opt_hidden', 'opt_heads', 'ps_in', 'ps_out');
fprintf('✅ 全部流程结束。请检查文件是否生成。\n');

%% ================= 辅助函数 =================
function data = getBatch(X, Y, idx)
    if iscell(idx), idx=[idx{:}]; end
    X_batch = X(:,:,idx);
    Y_batch = Y(:,:,idx);
    N = length(idx);
    data = cell(N,2);
    for i=1:N
        data{i,1} = X_batch(:,:,i);
        data{i,2} = Y_batch(:,:,i);
    end
end