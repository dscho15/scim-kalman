% readmatrix
k = 1
for i = 0.4 : 0.1 : 1.4
    
    k = k + 1
    
    figure(k)
    
    q      = readmatrix( "q_frq"      + string(double(i)) + ".csv");
    qdot_d = readmatrix( "qdot_frq"   + string(double(i)) + ".csv");
    qddot  = readmatrix( "qddot_frq"  + string(double(i)) + ".csv");
    
    title(string(i))
    
    % q
    subplot(3,1,1)
    plot(q)
    
    % qdot_d
    subplot(3,1,2)
    plot(qdot_d)
    
    % qddot
    subplot(3,1,3)
    plot(qddot)
end
    
% plot(q)
% plot(qdot_d)