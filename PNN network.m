clc
clear all
close all
%%
load fisheriris
P_train=[]
I_train=[]
P_test=[]
I_test=[]
species=grp2idx(species)
for i=1:3
    temp_input=meas((i-1)*50+1:i*50,:)
    temp_output=species((i-1)*50+1:i*50,:)
    n=randperm(50)
    P_train=[P_train temp_input(n(1:40),:)']
    I_train=[I_train temp_output(n(1:40),:)']
    P_test=[P_test temp_input(n(41:50),:)']
    I_test=[I_test temp_output(n(41:50),:)']
end
%%
Ic_train=ind2vec(I_train)
net_pnn=newpnn(P_train,Ic_train)
Ic_test=ind2vec(I_test)
t_sim_pnn=sim(net_pnn,P_test)
I_sim_pnn=vec2ind(t_sim_pnn)
error1=sum(abs( I_sim_pnn-I_test))/length(I_test)
%%

figure(1)
plot(1:length(I_test),I_test,'r-*')
hold on
plot(1:length(I_sim_pnn), I_sim_pnn,'b:o')
grid on
legend('real class','predict class')
xlabel('order')
ylabel('class')
string = {'prediction result of PNN network';
          ['estimation value error= ' num2str(error1) '%']};
title(string)

figure(2)
cm=confusionchart(I_test,I_sim_pnn)
cm.Title='confusion matrix for test data'
cm.ColumnSummary='column-normalized'
cm.RowSummary='row-normalized'

