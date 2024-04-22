%%
clear;clc;
load('..\..\data\kiknet-fake\unbalanced\history\hist-0-10000.mat', 'history');
eq = history;
num = size(eq, 1);
clear history
%% rspectra
Ts = 0.05:0.05:2; % 0.02:0.02:10;
t = (0.02:0.02:0.02 * size(eq, 2))';
rsptrm = zeros(num,length(Ts),3); 
temp = zeros(length(Ts),3); 
for i = 1:num
    p = -1 * eq(i,:)';
    disp(i)
    for j =1:length(Ts)
        w = 2*pi./Ts(j);
        k = 1*w^2;
        [u,du,ddu] = newmark_int(t,p,0,0,1,k,0.05); %Newmark-beta
        temp(j,:) = [max(abs(ddu'+eq(i,:)),[],2), max(abs(du))', max(abs(u))'];
    end
    rsptrm(i,:,:) = temp;
end
clear t temp i p j w k u du ddu
save('.\data\kiknet-fake\unbalanced\rspectra\0.05-2.00\rspec.mat','rsptrm')
