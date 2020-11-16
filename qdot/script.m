clc; clear;
q = readmatrix('joint_imp_q.csv');
q = q(:,1);
qdot = readmatrix('joint_imp_qd.csv');
qdot = qdot(:,1);

figure(1);
plot(q)
figure(2);
plot(qdot)

q_est = zeros(length(q),1);
q_est(1) = q(1);

for i = 2:length(q)
   q_est(i) = q_est(i-1) + 0.001 * qdot(i);
end

figure(3)
plot(q)
hold on;
plot(q_est)

torque = readmatrix('file.csv')
torque = torque(:,1)

figure(4)
plot(torque)