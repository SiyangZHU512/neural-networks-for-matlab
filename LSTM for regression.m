clc
clear all
W=xlsread('D:\桌面\hhh.xlsx',1,'A1:I84')
X=W(:,2:8)
Y=W(:,1)
%%
temp=randperm(size(X,1))
P_train=X(temp(1:50),:)'
I_train=Y(temp(1:50),:)'
P_test=X(temp(51:end),:)'
I_test=Y(temp(51:end),:)'
M=size(P_train,2)
N=size(P_test,2)
%%
[P_train,ps_input]=mapminmax(P_train,0,1)
P_test=mapminmax('apply',P_test,ps_input)
[t_train,ps_output]=mapminmax(I_train,0,1)
t_test=mapminmax('apply',I_test,ps_output)
%%
P_train=double(reshape(P_train,7,1,1,M))
P_test=double(reshape(P_test,7,1,1,N))
t_train=t_train'
t_test=t_test'
%%
for i=1:M
    p_train{i,1}=P_train(:,:,1,i)
end
for i=1:N
    p_test{i,1}=P_test(:,:,1,i)
end
%%
layers=[
    sequenceInputLayer(7)
    lstmLayer(4,'OutputMode','last')
    reluLayer

    fullyConnectedLayer(1)
    regressionLayer];
%%
options=trainingOptions('adam', ...
    'MiniBatchSize',30, ...    %training sample size for each time
    'MaxEpochs',2400, ...       % training for 800 times
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',800, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false)
%%
net=trainNetwork(p_train,t_train,layers,options)
T_sim1=predict(net,p_train)
T_sim2=predict(net,p_test)
%%
T_sim1=mapminmax('reverse',T_sim1,ps_output)'
T_sim2=mapminmax('reverse',T_sim2,ps_output)'
%%
MSE1=sqrt(sum(I_train-T_sim1)^2./size(I_train,2))
MSE2=sqrt(sum(I_test-T_sim2)^2./size(I_test,2))
%%
analyzeNetwork(layers)
%%
figure(1)
plot(1:size(I_test,2),I_test,'b:*',1:size(I_test,2),T_sim2,'r-o')
legend('original data','predict value')
title('LSTM:process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(MSE2)]};
title(string)
figure(2)
plot(1:size(I_train,2),I_train,'b:*',1:size(I_train,2),T_sim1,'r-o')
legend('original data','fitted value')
title('LSTM:process of fit')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(MSE1)]};
title(string)
