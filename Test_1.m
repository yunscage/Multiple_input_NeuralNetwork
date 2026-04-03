% 训练一个多输入神经网络
Dim1 = 101;
Dim2 = 24;
% 选择性训练

% 生成示例数据（修正：转置为 样本数×特征数）
N = 1000;
InData1 = rand(N, Dim1)';      
InData2 = rand(N, Dim2)';    
OutValue = rand(N, 1)';         

Xtrain={InData1,InData2};
Ytrain=OutValue;
% 创建独立的数据存储（注意：arrayDatastore 默认每行是一个样本）



NetPath1 = [
    featureInputLayer(Dim1, 'Name', 'Net1InLyr')
    fullyConnectedLayer(19)
    tanhLayer
    fullyConnectedLayer(6)
    tanhLayer('Name', 'Path1OutLayer')
];

NetPath2 = [
    featureInputLayer(Dim2, 'Name', 'Net2InLyr')
    fullyConnectedLayer(9)
    tanhLayer
    fullyConnectedLayer(3)
    tanhLayer('Name', 'Path2OutLayer')
];

commonPath = [
    concatenationLayer(1, 2, 'Name', 'concat')   % 沿特征维拼接
    fullyConnectedLayer(33)
    tanhLayer
    fullyConnectedLayer(6)
    tanhLayer
    fullyConnectedLayer(1)
];

% 构建层图
combNetwork = layerGraph();

combNetwork = addLayers(combNetwork, NetPath1);
combNetwork = addLayers(combNetwork, NetPath2);
combNetwork = addLayers(combNetwork, commonPath);

% 连接分支输出到拼接层
combNetwork = connectLayers(combNetwork, 'Path1OutLayer', 'concat/in1');
combNetwork = connectLayers(combNetwork, 'Path2OutLayer', 'concat/in2');

% 可视化或分析
% analyzeNetwork(dlnet(combNetwork));
plot(combNetwork);

% 训练选项（保持不变）
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'MiniBatchSize',512, ...
    'MaxEpochs', 50, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 1e-4, ...
    "Verbose",false,...
    'GradientThreshold', 1, ...
    'GradientThresholdMethod', 'global-l2norm', ...
    'ExecutionEnvironment', 'auto', ...
    'DispatchInBackground', false);

[Xtrain,Ytrain]=DataFormat(Xtrain,Ytrain);
Net=trainCustomNetwork(Xtrain,Ytrain,combNetwork,options);% trainnet trainNetwork

function [trainX,trainY]=DataFormat(trainX,trainY)
 if iscell(trainX)
    for i = 1:length(trainX)
        % 转为 dlarray 并指定维度标签
        trainX{i} = dlarray(trainX{i}, 'CB');
        % 如果需要 GPU
        trainX{i} = gpuArray(trainX{i});
    end
 else
     trainX = gpuArray(trainX);
 end
 if iscell(trainY)
    for i = 1:length(trainY)
        trainY{i} = gpuArray(trainY{i});
    end
 else
     trainY = gpuArray(trainY);
 end
end