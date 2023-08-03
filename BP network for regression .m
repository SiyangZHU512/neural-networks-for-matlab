clc
clear all
%% find optimal layer
W=xlsread('D:\桌面\hhh.xlsx',1,'A1:I84')
X=W(:,2:8)
Y=W(:,1)
MSE1=[]
MSE2=[]

for j=1:200
%% data preparation
temp=randperm(size(X,1))
P_train=X(temp(1:50),:)'
I_train=Y(temp(1:50),:)'
P_test=X(temp(51:end),:)'
I_test=Y(temp(51:end),:)'
M=size(P_train,2)
N=size(P_test,2)
%% standardized data
[P_train,ps_input]=mapminmax(P_train,0,1)
P_test=mapminmax('apply',P_test,ps_input)
[t_train,ps_output]=mapminmax(I_train,0,1)
t_test=mapminmax('apply',I_test,ps_output)
%%BP network setup
for i=1:30
Net=newff(P_train,t_train,i)
% set the parameter for BP network
Net.trainParam.eoichs=2000
Net.trainParam.goal=1e-6
Net.trainParam.lr=0.01
%% train the network
Net=train(Net,P_train,t_train);
%% prediction
T_sim1=sim(Net,P_train)
T_sim2=sim(Net,P_test)
T_sim1=mapminmax('reverse',T_sim1,ps_output)
T_sim2=mapminmax('reverse',T_sim2,ps_output)
%% model assessment
MSE1(j,i)=sqrt(sum(I_train-T_sim1)^2./size(I_train,2))
MSE2(j,i)=sqrt(sum(I_test-T_sim2)^2./size(I_test,2))
end
end
Error1=mean(MSE1)
Error2=mean(MSE2)
plot(1:30,Error1,'b-',1:30,Error2,'r-')
%% find the layer with min Error1 and Error2
%% model fit and prediction
clc
clear all
W=xlsread('D:\桌面\hhh.xlsx',1,'A1:I84')
X=W(:,2:8)
Y=W(:,1)
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
Net=newff(P_train,t_train,13)
Net.trainParam.eoichs=2000
Net.trainParam.goal=1e-6
Net.trainParam.lr=0.01
Net=train(Net,P_train,t_train);
T_sim1=sim(Net,P_train)
T_sim2=sim(Net,P_test)
T_sim1=mapminmax('reverse',T_sim1,ps_output)
T_sim2=mapminmax('reverse',T_sim2,ps_output)
MSE1=sqrt(sum(I_train-T_sim1)^2./size(I_train,2))
MSE2=sqrt(sum(I_test-T_sim2)^2./size(I_test,2))
%% prediction result presentation by graph
figure(1)
plot(1:size(I_test,2),I_test,'b:*',1:size(I_test,2),T_sim2,'r-o')
legend('original data','predict value')
title('BP network:process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(MSE2)]};
title(string)
figure(2)
plot(1:size(I_train,2),I_train,'b:*',1:size(I_train,2),T_sim1,'r-o')
legend('original data','fitted value')
title('BP network:process of fit')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(MSE1)]};
title(string)

%%using BP network for multiple output case
clear all
W=xlsread('D:\桌面\hhh.xlsx',1,'A1:K574')
X=W(:,3:9)
Y=W(:,1:2)
temp=randperm(size(X,1))
P_train=X(temp(1:50),:)'
I_train=Y(temp(1:50),:)'
P_test=X(temp(51:end),:)'
I_test=Y(temp(51:end),:)'
N=size(P_test,2)
[P_train,ps_input]=mapminmax(P_train,0,1)
P_test=mapminmax('apply',P_test,ps_input)
[t_train,ps_output]=mapminmax(I_train,0,1)
Net=newff(P_train,t_train,13)
Net.trainParam.eoichs=2000
Net.trainParam.goal=1e-7
Net.trainParam.lr=0.01
Net=train(Net,P_train,t_train);
T_sim1=sim(Net,P_train)
T_sim2=sim(Net,P_test)
T_sim1=mapminmax('reverse',T_sim1,ps_output)
T_sim2=mapminmax('reverse',T_sim2,ps_output)
%%
I_test1=I_test(1,:)
I_test2=I_test(2,:)
I_train1=I_train(1,:)
I_train2=I_train(2,:)
Y11=T_sim1(1,:)
Y21=T_sim1(2,:)
Y1=T_sim2(1,:)
Y2=T_sim2(2,:)
%%
MSE11=sqrt(sum(I_train1-Y11)^2./size(Y11,2))
MSE12=sqrt(sum(I_train2-Y21)^2./size(Y21,2))
MSE21=sqrt(sum(I_test1-Y1)^2./size(Y1,2))
MSE22=sqrt(sum(I_test2-Y2)^2./size(Y2,2))
%%
figure(1)
subplot 121
plot(1:size(I_test1,2),I_test1,'b:*',1:size(I_test2,2),Y1,'r-o')
legend('original data','predict value')
title('BP network:process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(MSE21)]};
subtitle(string)
subplot 122
plot(1:size(I_train1,2),I_train1,'b:*',1:size(I_train1,2),Y11,'r-o')
legend('original data','fitted value')
title('BP network:process of fitting')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(MSE11)]};
subtitle(string)
figure(2)
subplot 121
plot(1:size(I_test2,2),I_test2,'b:*',1:size(I_test2,2),Y2,'r-o')
legend('original data','predict value')
title('BP network:process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(MSE22)]};
subtitle(string)
subplot 122
plot(1:size(I_train2,2),I_train2,'b:*',1:size(I_train2,2),Y21,'r-o')
legend('original data','fitted value')
title('BP network:process of fitting')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(MSE12)]};
subtitle(string)
