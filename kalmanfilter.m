% q
q = readmatrix('joint_imp_q.csv');
q = q(1,:)

%% qdot
qdot = readmatrix('joint_imp_qd.csv')

M = qdot(1:4000,6)
plot(M)


%%
qddot = M(2:end) - M(1:end-1)
plot(qddot)

%% Calculate the Mean
mean_qdot = mean(M)

%% Covariance Matrix
covariance_qdot = cov(M)

%% There is no significatn correlation
[R , P] = corrcoef(M)

%% Plot the PlotMatrix
plotmatrix(M)

%% Anova
% There is an indiction that the mean are not the same, however we did not
% expect that neither. So we need to measure the 
[p, tbl, stats] = anova1(M)

%%
torque = readmatrix('torque.csv')
plotmatrix(torque)

%%
mean_torque = mean(torque)

%% covariance torque

[R , P] = corrcoef(torque)

%%
torque = readmatrix('joint_imp_torque.csv')
torque = torque(1:136,1:7)
plotmatrix(torque)
