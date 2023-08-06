clc
clear all
close all
A=xlsread('D:\桌面\logit.xlsx',1,'A1:C4001')
Y=A(:,1)
X=A(:,2:3)
temp=randperm(size(X,1))
P_train=X(temp(1:257),:)'
I_train=Y(temp(1:257),:)'
P_test=X(temp(258:end),:)'
I_test=Y(temp(258:end),:)'
%%
net_grnn=newgrnn(P_train,I_train)
t_sim_grnn=sim(net_grnn,P_test)
T_sim_grnn=round(t_sim_grnn)
error2=sum(abs(T_sim_grnn-I_test))/length(I_test)
%%
figure(1)
plot(1:length(I_test),I_test,'r-*')
hold on
plot(1:length(T_sim_grnn),T_sim_grnn,'b:o')
grid on
legend('real class','predict class')
xlabel('order')
ylabel('class')
string = {'prediction result of GRNN network';
          ['estimation value error= ' num2str(error2) '%']};
title(string)
figure(2)
cm=confusionchart(I_test,T_sim_grnn)
cm.Title='confusion matrix for test data'
cm.ColumnSummary='column-normalized'
cm.RowSummary='row-normalized'

