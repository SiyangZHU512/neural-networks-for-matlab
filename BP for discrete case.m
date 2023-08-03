clc
clear all
close all
clc
warning off
%% data preparation
load fisheriris
species=grp2idx(species)
%% set the training set and test set
meas=mapminmax(meas)
temp=randperm(size(meas,1))
P_train=meas(temp(1:120),:)'
I_train=species(temp(1:120),:)'
P_test=meas(temp(121:end),:)'
I_test=species(temp(121:end),:)'
%% standardized data
[P_train,ps_input]=mapminmax(P_train,0,1)
[P_test,ps_input]=mapminmax(P_test,0,1)
Net=newff(P_train,I_train,8)
%% network parameter setup
Net.trainParam.eoichs=2000
Net.trainParam.goal=1e-6
Net.trainParam.lr=0.01
Net=train(Net,P_train,I_train);
%% prediction
T_sim1=sim(Net,P_train)
T_sim2=sim(Net,P_test)
T_sim2=round(T_sim2)
error=sum(abs(T_sim2-I_test))/30
%%result presentation by graph
figure(1)
plot(1:30,I_test,'r-*')
hold on
plot(1:30,T_sim2,'b:o')
grid on
legend('real class','predicted class')
xlabel('order')
ylabel('classes')
string = {'BP netwrok';
          ['estimation value error= ' num2str(error) '%']};
title(string)
figure
cm=confusionchart(I_test,T_sim2)
cm.Title='confusion matrix for test data'
cm.ColumnSummary='column-normalized'
cm.RowSummary='row-normalized'