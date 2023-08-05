warning off
close all
clear 
clc
%%
A=xlsread('D:\桌面\logit.xlsx',2,'A1:A404')
%%
num_sam=length(A)
kim=10   
zim=1
figure(1)
plot(1:num_sam,A)
%%
for i=1:num_sam-kim-zim+1
   A1(i,:)=[reshape(A(i:i+kim-1),1,kim),A(i+kim+zim-1)]
end
%%
temp=randperm(size(A1,1))
P_train=A1(temp(1:300),1:10)'
I_train=A1(temp(1:300),11)'
P_test=A1(temp(301:end),1:10)'
I_test=A1(temp(301:end),11)'
M=size(P_train,2)
N=size(P_test,2)
%%
[P_train,ps_input]=mapminmax(P_train,0,1)
P_test=mapminmax('apply',P_test,ps_input)
[t_train,ps_output]=mapminmax(I_train,0,1)
t_test=mapminmax("apply",I_test,ps_output)
%%
P_train=double(reshape(P_train,10,1,1,M))
P_test=double(reshape(P_test,10,1,1,N))
t_train=double(t_train)'
t_test=double(t_test)'
%%
%%
for i=1:M
    p_train{i,1}=P_train(:,:,1,i)
end
for i=1:N
    p_test{i,1}=P_test(:,:,1,i)
end
%%
%%
layers=[
    sequenceInputLayer(10)
    lstmLayer(4,'OutputMode','last')
    reluLayer

    fullyConnectedLayer(1)
    regressionLayer];
%%
options=trainingOptions('adam', ...
    'MiniBatchSize',30, ...    %training sample size for each time
    'MaxEpochs',1200, ...       % training for 800 times
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
MSE1=sqrt(mean((I_train-T_sim1).^2))
MSE2=sqrt(mean((I_test-T_sim2).^2))
%%
figure(1)
plot(1:N,I_test,'b-',1:N,T_sim2,'r-')
legend('original data','predict value')
title('LSTM :process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(MSE2)]};
title(string)
figure(2)
plot(1:size(I_train,2),I_train,'b-',1:size(I_train,2),T_sim1,'r-')
legend('original data','fitted value')
title('LSTM:process of fit')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(MSE1)]};
title(string)
