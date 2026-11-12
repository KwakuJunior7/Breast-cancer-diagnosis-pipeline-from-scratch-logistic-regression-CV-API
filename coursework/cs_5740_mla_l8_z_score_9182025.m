clc;
clear all;

X = [1 2 3; ...
     2 1 2; ...
     3 1 2];

n = size(X,1);
N = size(X,2);

x_mean = mean(X); %1xp
x_std  = std(X); % 1xp

X_mean = ones(n,1)*x_mean;
X_c = X - X_mean; % (nxp) - (nx1)*(1xp) = nxp

X_std = 1./(ones(n,1)*x_std);
Z = X_c.*X_std;

R_ideal = corr(X);

R_z = Z'*Z./(N-1);
