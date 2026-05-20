function shapValues = shapley_lstm_3d(net, x_cell_data, ref_matrix, featureNames)
% SHAPLEY_LSTM_3D 计算 LSTM 时间序列模型的特征重要性 (适配淮河数据)
% 输入:
%   net: 训练好的网络
%   x_cell_data: {N x 1} 的 Cell 数组，每个 Cell 是 [Features x Time]
%   ref_matrix: [Features x Time] 的基准矩阵 (Reference)
% 输出:
%   shapValues: [N x Features] 矩阵，表示每个样本中每个特征的平均贡献

    numSamples = length(x_cell_data);
    numFeatures = size(ref_matrix, 1);
    
    shapValues = zeros(numSamples, numFeatures);
    
    fprintf('   > SHAP 计算模式: Marginal Contribution (Leave-One-Out Approximation)\n');
    
    % 1. 计算 Base Prediction
    try
        Full_Preds_Cell = predict(net, x_cell_data, 'MiniBatchSize', 256, 'ExecutionEnvironment','gpu');
    catch
        fprintf('⚠️ 显存不足，切换至 CPU...\n');
        Full_Preds_Cell = predict(net, x_cell_data, 'MiniBatchSize', 256, 'ExecutionEnvironment','cpu');
    end
    
    % 2. 逐特征循环
    for j = 1:numFeatures
        fprintf('     [%d/%d] 分析特征: %s ...\n', j, numFeatures, featureNames{j});
        
        % 构建干扰数据：将第 j 个特征替换为 Reference
        X_Perturbed = x_cell_data; 
        for i = 1:numSamples
            X_Perturbed{i}(j, :) = ref_matrix(j, :);
        end
        
        try
            Perturbed_Preds = predict(net, X_Perturbed, 'MiniBatchSize', 256, 'ExecutionEnvironment','gpu');
        catch
            Perturbed_Preds = predict(net, X_Perturbed, 'MiniBatchSize', 256, 'ExecutionEnvironment','cpu');
        end
        
        % 计算影响 (Marginal Contribution)
        for i = 1:numSamples
            if iscell(Full_Preds_Cell)
                y_full = Full_Preds_Cell{i};
                y_pert = Perturbed_Preds{i};
            else
                y_full = Full_Preds_Cell(i,:); 
                y_pert = Perturbed_Preds(i,:);
            end
            % 记录平均影响 (方向性)
            shapValues(i, j) = mean(y_full - y_pert); 
        end
    end
    fprintf('   > SHAP 计算完成。\n');
end