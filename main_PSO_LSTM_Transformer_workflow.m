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

%% ================= Main Workflow: PSO Optimization & Training =================
clc; clear; close all;

rng(42); % Set random seed for reproducibility

% 1. Load Dataset
fprintf('Loading sample dataset...\n');
% [Security] Use a relative path to load the sample dataset
dataPath = fullfile(pwd, 'data', 'sample_Train_3D.mat');
if ~isfile(dataPath)
    error('[!] File not found: %s. Please ensure your current directory is the project root.', dataPath);
end
load(dataPath, 'x_train', 'y_train', 'x_test', 'y_test');

%% ================= 2. Data Normalization (Critical Step) =================
fprintf('Performing data normalization...\n');

% Note: mapminmax requires 2D matrices [Features x Samples].
% We flatten the 3D data into 2D to calculate min/max parameters, then reshape it back.

% 1. Flatten X (Training Set)
[nFeat, nTime, nTrain] = size(x_train);
X_Train_2D = reshape(x_train, nFeat, []); 

% 2. Calculate normalization parameters (ONLY use training set to prevent data leakage)
[X_Train_Norm_2D, ps_in] = mapminmax(X_Train_2D, 0, 1);

% 3. Reshape X (Training Set) and convert back to single to save memory
x_train = reshape(single(X_Train_Norm_2D), nFeat, nTime, nTrain);
clear X_Train_2D X_Train_Norm_2D; % Free memory immediately

% Apply to Y (Test Set)
y_test_2D = reshape(y_test, 1, []);
y_test_2D = mapminmax('apply', y_test_2D, ps_out);
y_test = reshape(single(y_test_2D), 1, nTime, size(y_test, 3));
clear y_test_2D;

fprintf('Normalization complete. Memory cleared.\n');

%% ================= Step 1: Automatic PSO Invocation =================
% Call the custom PSO optimization function
fprintf('Step 1: Initiating Particle Swarm Optimization (PSO)...\n');
[opt_hidden, opt_heads] = run_pso_optimization(x_train, y_train, x_test, y_test);

%% ================= Step 2: Define Final Model with Optimal Parameters =================
fprintf('Step 2: Commencing full-scale training (Optimal Params: Hidden=%d, Heads=%d)...\n', opt_hidden, opt_heads);

[nFeat, nTime, nTrain] = size(x_train);
maxPosition = nTime;

% Dimension logic must remain consistent with the PSO configuration
att_dim_final = opt_heads * 16; 

layers = [
    sequenceInputLayer(nFeat, 'Name', 'input')
    
    % LSTM layer using the optimal hidden units found by PSO
    lstmLayer(opt_hidden, 'OutputMode', 'sequence', 'Name', 'lstm')
    dropoutLayer(0.1, 'Name', 'drop1')
    
    % Dimension matching for attention mechanism
    positionEmbeddingLayer(opt_hidden, maxPosition, 'Name', 'pos-emb')
    additionLayer(2, 'Name', 'add') 
    
    % Projection layer (corresponds to PSO logic)
    fullyConnectedLayer(att_dim_final, 'Name', 'proj_att') 
    
    fullyConnectedLayer(64, 'Name', 'ff1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'fc_out')
    regressionLayer('Name', 'output')
];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'drop1', 'add/in2');

%% ================= Step 3: Full-Scale Training Configuration =================
% Construct Datastore (Utilizing memory-efficient batch loading)
readFcn_Train = @(idx) getBatch(x_train, y_train, idx);
ds_Train = transform(arrayDatastore(1:nTrain), readFcn_Train);

% Validation Set Datastore
readFcn_Test = @(idx) getBatch(x_test, y_test, idx);
ds_Test = transform(arrayDatastore(1:size(x_test,3)), readFcn_Test);

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...              % Increased epochs for formal training
    'MiniBatchSize', 1024, ...
    'InitialLearnRate', 0.005, ...
    'ValidationData', ds_Test, ...    % Incorporate validation set
    'ValidationPatience', 10, ...     % Early stopping mechanism
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'gpu');

% Execute Training
fprintf('Starting neural network training...\n');
[net, info] = trainNetwork(ds_Train, lgraph, options);

% Save the final model and parameters
savePath = fullfile(pwd, 'models', 'Final_Model_With_PSO.mat');
save(savePath, 'net', 'info', 'opt_hidden', 'opt_heads', 'ps_in', 'ps_out');
fprintf('Training complete. Model saved to: %s\n', savePath);
