warning off
close all
clear 
clc
%% data preparation
A=xlsread('D:\桌面\logit.xlsx',2,'A1:A404')
%% data transformation
num_sam=length(A)
kim=10    
zim=1
figure(1)
plot(1:num_sam,A)
%% 1-10 as input 11 as output then 2-11 as input 12 as output
for i=1:num_sam-kim-zim+1
   A1(i,:)=[reshape(A(i:i+kim-1),1,kim),A(i+kim+zim-1)]
end
%% training set and test set setup
temp=1:1:393
P_train=A1(temp(1:300),1:10)'
I_train=A1(temp(1:300),11)'
P_test=A1(temp(301:end),1:10)'
I_test=A1(temp(301:end),11)'
N=size(P_test,2)
%% standardized data
[P_train,ps_input]=mapminmax(P_train,0,1)
P_test=mapminmax('apply',P_test,ps_input)
[I_train,ps_output]=mapminmax(I_train,0,1)
I_test=mapminmax("apply",I_test,ps_output)
Net=newff(P_train,I_train,9)
%%network parameter setup
Net.trainParam.eoichs=2000
Net.trainParam.goal=1e-6
Net.trainParam.lr=0.01
%% network training
Net=train(Net,P_train,I_train);
%% prediction
t_sim1=sim(Net,P_train)
t_sim2=sim(Net,P_test)
T_sim1=mapminmax('reverse',t_sim1,ps_output)
T_sim2=mapminmax('reverse',t_sim2,ps_output)
I_train=mapminmax('reverse',I_train,ps_output)
I_test=mapminmax('reverse',I_test,ps_output)
MSE1=sqrt(sum((I_train-T_sim1).^2)/N)
MSE2=sqrt(sum((I_test-T_sim2).^2)/N)
%%result presentation by graph
figure(1)
plot(1:N,I_test,'b-',1:N,T_sim2,'r-')
legend('original data','predict value')
title('BP network:process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(MSE2)]};
title(string)
figure(2)
plot(1:size(I_train,2),I_train,'b-',1:size(I_train,2),T_sim1,'r-')
legend('original data','fitted value')
title('random forest:process of fit')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(MSE1)]};
title(string)