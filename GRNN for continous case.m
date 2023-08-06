%%
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
Net=newgrnn(P_train,I_train)
T_sim1=sim(Net,P_train)
T_sim2=sim(Net,P_test)
%%
Error11=sqrt(mean((I_train-T_sim1).^2))
Error21=sqrt(mean((I_test-T_sim2).^2))
figure(1)
plot(1:size(I_test,2),I_test,'b:*',1:size(I_test,2),T_sim2,'r-o')
legend('original data','predict value')
title('GRNN network:process of prediction')
xlabel('order')
ylabel('value')
string = {'prediction VS real value';
          ['estimation value error= ' num2str(Error21)]};
title(string)
figure(2)
plot(1:size(I_train,2),I_train,'b:*',1:size(I_train,2),T_sim1,'r-o')
legend('original data','fitted value')
title('GRNN network:process of fit')
xlabel('order')
ylabel('value')
string = {'fitted value VS real value';
          ['estimation value error= ' num2str(Error11)]};
title(string)




