clc;
clear all;

% EDA for Breast Cancer Data Set (UCI Repository)

%Retrieve the breast cancer data
D = csvread('wdbc.data'); % D = [X|y] belongs to 569x31 --> instances = 569
% features = 30
D_size = size(D);
disp(D_size);

X_temp1 = D(:,1); % features
X_temp2 = D(:,3:32); %features
X = [X_temp1 X_temp2]; %features

y = D(:,2); % labels

%X_benign = D(1:59,2:14); % Features --> benign
%X_malign = D(60:131,2:14); % Features --> malignant


%y_benign = D(1:59,1); % Label --> benign
%y_malign = D(60:131,1); % Label --> malignant



