clf;
qdot_mes = readmatrix('arbitrary/q_des_1/1/qdot.csv')';
qdot_lp = readmatrix('lp_arb.csv');
qdot_kal = readmatrix('kalman_arb.csv');
q_mes = readmatrix('arbitrary/q_des_1/1/q.csv')';

q_calc_mes = zeros(size(q_mes));
q_calc_mes(:,1) = q_mes(:,1);

q_calc_kal = q_calc_mes;
q_calc_lp = q_calc_mes;

T = 0.001015

for i = 2:length(q_mes(1,:))
    
    q_calc_mes(:,i) = q_calc_mes(:,i-1) + T * qdot_mes(:,i);
    q_calc_kal(:,i) = q_calc_kal(:,i-1) + T * qdot_kal(:,i);
    q_calc_lp(:,i) = q_calc_lp(:,i-1) + T * qdot_lp(:,i);
    
end

figure(1)
plot(q_mes(4,:))
hold on;
plot(q_calc_mes(4,:))
hold on;
plot(q_calc_kal(4,:))
hold on;
plot(q_calc_lp(4,:))
legend('qmes','q_{mesdot}', 'q_{kal}', 'q_{lp}')


figure(2)
plot((q_mes(4,:)-q_calc_mes(4,:)).^2)
hold on;
plot((q_mes(4,:)-q_calc_kal(4,:)).^2)
hold on;
plot((q_mes(4,:)-q_calc_lp(4,:)).^2)

obj1 = sum((q_mes(3,:)-q_calc_mes(3,:)).^2)
obj2 = sum((q_mes(3,:)-q_calc_kal(3,:)).^2)
obj3 = sum((q_mes(3,:)-q_calc_lp(3,:)).^2)

qddot_mes = zeros(size(qdot_mes));
qddot_kal = zeros(size(qdot_kal));
qddot_lp = zeros(size(qdot_lp));

for i = 2:length(q_mes(1,:))
    
    qddot_kal(:,i) = (qdot_kal(:,i) - qdot_kal(:,i-1))/T;
    qddot_mes(:,i) = (qdot_mes(:,i) - qdot_mes(:,i-1))/T;
    qddot_lp(:,i) = (qdot_lp(:,i) - qdot_lp(:,i-1))/T;
    
end

figure(3)
plot(qddot_mes(4,:))
hold on;
plot(qddot_kal(4,:))
hold on;
plot(qddot_lp(4,:))