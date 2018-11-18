trd=load('trD.mat');
tstd=load('tstD.mat');
%bowCs=load('bowCs.mat');
[trIds, trLbs] = ml_load('C:\Users\Saif\Desktop\ms\sem 1\ML\HW5\hw5data\bigbangtheory_v2\train.mat','imIds', 'lbs');
%% 
%For question 3.4.2 and 3.4.3 by changing parameters 
model=svmtrain(trLbs,trd.trD','-s 0 -t 2 -g 20 -c 250');
%% 
[pred,acc,prob]=svmpredict(ones(1600,1),tstD',model);
%% 
csvwrite('pred.csv',pred);
%% 
%For question 3.4.4 and 3.4.5
kern=km_Utils2.getKernel(trd.trD');
gamma=mean(kern,'all');
%% 
%parameters tuning 
[train_chi,test_chi]=km_Utils2.cmpExpX2Kernel(trd.trD', tstD.tstD', 20);
%% 
train_chi=[(1:size(train_chi,1))' train_chi];
%% 
model_chi=svmtrain(trLbs,train_chi,'-t 4 -c 200 -v 5');
%% 
test_chi=[(1:size(test_chi,1))' test_chi];
%% 
p_chi=svmpredict(ones(1600,1),test_chi,model_chi);
%% 
csvwrite('p_chi.csv',p_chi);
