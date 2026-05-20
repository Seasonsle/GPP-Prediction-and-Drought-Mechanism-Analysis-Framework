function [best_hidden, best_heads] = run_pso_optimization(x_train, y_train, x_test, y_test)
% RUN_PSO_OPTIMIZATION 自动抽取子集并进行超参数寻优
% 输入：全量训练集和测试集 (3D 矩阵)
% 输出：最优 LSTM 隐藏层数, 最优 Attention 头数

    fprintf('==================================================\n');
    fprintf('Step 1: 启动 PSO 超参数自动寻优模块\n');
    fprintf('==================================================\n');

    %% 1. 数据采样 (只取 5% 用于快速搜索)
    ratio = 0.05; 
    nTrain = size(x_train, 3);
    nTest  = size(x_test, 3);
    
    % 随机抽样索引
    train_idx = randperm(nTrain, round(nTrain * ratio));
    test_idx  = randperm(nTest, round(nTest * 0.1)); % 测试集也可以少取点
    
    fprintf('正在构建寻优子数据集 (Train: %d, Test: %d)...\n', length(train_idx), length(test_idx));
    
    % 转换为 Cell 格式 (trainNetwork 需要)
    X_Sub_Train = local_to_cell(x_train, train_idx);
    Y_Sub_Train = local_to_cell(y_train, train_idx);
    
    X_Sub_Test  = local_to_cell(x_test, test_idx);
    % Y_Sub_Test 需要保留数值矩阵格式用于计算 RMSE
    Y_Sub_Test_Ref = squeeze(y_test(:,:,test_idx)); 

    %% 2. PSO 参数配置
    pop_size = 5;       % 种群数 (由于自动运行，设小一点节省时间)
    max_iter = 5;       % 迭代次数
    lb = [64, 2];       % [Hidden下限, Heads下限]
    ub = [256, 8];      % [Hidden上限, Heads上限]
    dim = 2;
    
    % 定义目标函数句柄 (闭包，包含数据)
    fobj = @(theta) fitness_function(theta, X_Sub_Train, Y_Sub_Train, X_Sub_Test, Y_Sub_Test_Ref);

    %% 3. 运行 PSO
    % 注意：需确保 PSO.m 和 ys.m (如果PSO依赖它) 在路径中
    % 这里假设你已经有 PSO 函数
    fprintf('PSO 寻优开始 (Pop=%d, Iter=%d)...\n', pop_size, max_iter);
    [best_rmse, gBest, cg_curve] = PSO(pop_size, max_iter, lb, ub, dim, fobj, 4);
    
    best_hidden = round(gBest(1));
    best_heads  = round(gBest(2));
    
    fprintf('\n---------------------------------------\n');
    fprintf('PSO 寻优完成！\n');
    fprintf('最优 RMSE: %.4f\n', best_rmse);
    fprintf('推荐参数 -> LSTM Hidden: %d, Attention Heads: %d\n', best_hidden, best_heads);
    fprintf('---------------------------------------\n\n');
    
    % 可选：绘制收敛曲线
    figure('Name', 'PSO Optimization Process');
    plot(cg_curve, 'r-o', 'LineWidth', 1.5);
    title('Hyperparameter Optimization Convergence');
    xlabel('Iteration'); ylabel('Validation RMSE');
    drawnow;
end

%% --- 内部辅助函数：转换数据为 Cell ---
function data_cell = local_to_cell(Data_3D, idx)
    data_cell = cell(length(idx), 1);
    for i = 1:length(idx)
        data_cell{i} = Data_3D(:,:,idx(i));
    end
end

%% --- 内部辅助函数：适应度函数 (训练并评估) ---
function rmse = fitness_function(params, X_Train, Y_Train, X_Test, Y_True)
    % 1. 解析参数
    numHidden = round(params(1));
    numHeads  = round(params(2));
    
    % 2. 动态构建网络
    [numFeat, nTime] = size(X_Train{1});
    
    % 保证维度匹配的策略：
    % 设定 Attention 的通道数为 Head 的整数倍 (e.g., Head * 16)
    att_dim = numHeads * 16; 
    
    layers = [
        sequenceInputLayer(numFeat, 'Name', 'in')
        
        % 动态 LSTM
        lstmLayer(numHidden, 'OutputMode', 'sequence', 'Name', 'lstm')
        dropoutLayer(0.1, 'Name', 'drop')
        
        % 位置编码 (需匹配 LSTM 输出)
        positionEmbeddingLayer(numHidden, nTime, 'Name', 'pos')
        additionLayer(2, 'Name', 'add')
        
        % 投影层 (关键：将维度调整为 Attention 所需的 att_dim)
        fullyConnectedLayer(att_dim, 'Name', 'proj')
        
        % 动态 Attention
        selfAttentionLayer(numHeads, att_dim, 'AttentionMask', 'causal', 'Name', 'att')
        layerNormalizationLayer('Name', 'ln')
        
        fullyConnectedLayer(1, 'Name', 'fc')
        regressionLayer('Name', 'out')
    ];
    
    lgraph = layerGraph(layers);
    lgraph = connectLayers(lgraph, 'drop', 'add/in2');
    
    % 3. 极速训练配置 (只跑 3-5 轮)
    options = trainingOptions('adam', ...
        'MaxEpochs', 3, ...             % 只要看趋势，不需要收敛
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 0.01, ...
        'Shuffle', 'every-epoch', ...
        'ExecutionEnvironment', 'auto', ...
        'Verbose', false);              % 静默模式

    try
        net = trainNetwork(X_Train, Y_Train, lgraph, options);
        
        % 预测与评估
        Y_Pred_Cell = predict(net, X_Test, 'MiniBatchSize', 256);
        
        % Cell 转 Matrix
        Y_Pred = zeros(size(Y_True), 'single');
        for k=1:length(Y_Pred_Cell), Y_Pred(:,k) = Y_Pred_Cell{k}; end
        
        % 计算 RMSE
        err = Y_Pred - Y_True;
        rmse = sqrt(mean(err(:).^2));
        
        % 简单的惩罚：如果结果是 NaN，给个大数
        if isnan(rmse), rmse = 1000; end
        
        fprintf('  > Try: Hidden=%-3d Heads=%-2d | RMSE=%.4f\n', numHidden, numHeads, rmse);
    catch
        rmse = 1000; % 报错惩罚
        fprintf('  > Try: Hidden=%-3d Heads=%-2d | Failed\n', numHidden, numHeads);
    end
end