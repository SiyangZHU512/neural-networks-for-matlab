warning off
close all
clear 
clc
%%
A=xlsread('D:\桌面\logit.xlsx',2,'A1:A404')
%%
num_sam=length(A)
kim=3   
zim=2
figure(1)
plot(1:num_sam,A)
%% 1-10 as input 11 as output then 2-11 as input 12 as output
%%
for i=1:num_sam-kim-zim+1
   A1(i,:)=[reshape(A(i:i+kim-1),1,kim),A(i+kim+zim-1)]
end
%%
temp=1:1:399
P_train=A1(temp(1:300),1:3)'
I_train=A1(temp(1:300),4)'
P_test=A1(temp(301:end),1:3)'
I_test=A1(temp(301:end),4)'
N=size(P_test,2)
%%
Net=newgrnn(P_train,I_train)
%%
T_sim1=sim(Net,P_train)
T_sim2=sim(Net,P_test)
%%
MSE1=sqrt(sum((I_train-T_sim1).^2)/N)
MSE2=sqrt(sum((I_test-T_sim2).^2)/N)
%%
figure(1)
plot(1:N,I_test,'b-',1:N,T_sim2,'r-')
legend('original data','predict value')
title('GRNN network:process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(MSE2)]};
title(string)
figure(2)
plot(1:size(I_train,2),I_train,'b-',1:size(I_train,2),T_sim1,'r-')
legend('original data','fitted value')
title('GRNN:process of fit')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(MSE1)]};
title(string)