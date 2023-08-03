clc
clear all
close all
clc
warning off
%%
load fisheriris
species=grp2idx(species)
meas=mapminmax(meas)
temp=randperm(size(meas,1))
P_train=meas(temp(1:120),:)'
I_train=species(temp(1:120),:)'
P_test=meas(temp(121:end),:)'
I_test=species(temp(121:end),:)'
M=size(P_train,2)
N=size(P_test,2)
[P_train,ps_input]=mapminmax(P_train,0,1)
P_test=mapminmax("apply",P_test,ps_input)
T_test=categorical(I_test)'
T_train=categorical(I_train)'
P_train =double(reshape(P_train,4,1,1,M))
P_test=double(reshape(P_test,4,1,1,N))
%
layers = [
    imageInputLayer([4 1 1]) % 4X1X1 refers to number of features per sample
    convolution2dLayer(3,16,'Padding','same')
    reluLayer
    fullyConnectedLayer(384) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(384) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(3) % 3 refers to number of neurons in next output layer (number of output classes)
    softmaxLayer
    classificationLayer];
%%
options=trainingOptions('sgdm', ...
    'MaxEpochs',800, ...       % training for 800 times
    'InitialLearnRate',1e-3, ...
    'L2Regularization',1e-04,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',400, ...
    'Shuffle','every-epoch', ...
    'ValidationPatience', Inf,...
    'Plots','training-progress', ...
    'Verbose',false)
%%
%%
net=trainNetwork(P_train,T_train,layers,options)
T_sim1=predict(net,P_train)
T_sim2=predict(net,P_test)
%%
T_sim1=vec2ind(T_sim1')
T_sim2=vec2ind(T_sim2')
%%
error1=sum((T_sim1==I_train))/M*100
error2=sum((T_sim2==I_test))/N*100
%%
figure(1)
plot(1:N,I_test,'r-*')
hold on
plot(1:N,T_sim2,'b:o')
grid on
legend('real class','predicted class')
xlabel('order')
ylabel('classes')
string = {'CNN netwrok';
          ['estimation accuracy= ' num2str(error2) '%']};
title(string)
figure
cm=confusionchart(I_test,T_sim2)
cm.Title='confusion matrix for test data'
cm.ColumnSummary='column-normalized'
cm.RowSummary='row-normalized'
