
clc
clear

% load full workspace of full-horizon MPC script
load fullhorizon_completeWS.mat


% reduce data to a minimum just for direct comparison with
% infinitesimal-MPC
u_sol_vec_fullMPC = u_sol_vec;
x_sol_vec_fullMPC = x_sim; 
Q_full = Q_weight;
R_full = R;


iter_conv_full = k;

minSolveTime_full = min(tEnd, [], 'all') * 1000;
maxSolveTime_full = max(tEnd, [], 'all') * 1000;
meanSolveTime_full = mean(tEnd, 'all') * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime_full);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_full);
fprintf('Mean solve time: %f ms\n', meanSolveTime_full);


save('fullHor_MPC_comparison.mat',"meanSolveTime_full","maxSolveTime_full","minSolveTime_full","R_full","Q_full","x_sol_vec_fullMPC","u_sol_vec_fullMPC","iter_conv_full")





