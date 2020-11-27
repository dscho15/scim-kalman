% readmatrix
k = 1
for i = 1.1 : 0.1 : 1.4
    
    k = k + 1
    
    figure(k)
    
    q      = readmatrix( "q_test"      + string(i) + ".csv");
    qdot_d = readmatrix( "qdot_d_test" + string(i) + ".csv");
    qddot  = readmatrix( "qddot_test"  + string(i) + ".csv");
    
    title(string(i))
    
    subplot(3,1,1)
    plot(q)
    
    subplot(3,1,2)
    plot(qdot_d)
    
    subplot(3,1,3)
    plot(qddot)
    
    
    
end
    
% plot(q)
% plot(qdot_d)