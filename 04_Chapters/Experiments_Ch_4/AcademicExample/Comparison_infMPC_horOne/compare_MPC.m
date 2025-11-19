% ------------------------------------------------------------------------
%
% Short Description:    Within this script the horizon-one NMPC and the
%                       continuous-time infinetismal NMPC are compared in 
%                       a simple academic example. As a result an 
%                       aggregated table is provided to easily compare
%                       both approaches.
%
% Input:
%     1) Run reachability_NStep_horizonOne.m to get the backward reachable
%        set for horizon one. (stored in VDP_horizon5s.mat)
%     2) Run Synthesis.m to pre-compute the ingredients for the
%        continuous-time approach. (stored in terminalIngredients.mat)
%     3) Run this script.
%
% Note: Make sure cost both synthesis scripts have the same constraints and
%       in this script the cost function is the same. Also the sampling
%       rate in simulation must be in line with the horizon-one
%       pre-computation.
%
% ------------------------------------------------------------------------

close all
clear
clc

import casadi.*

%% Simulation parameter
problem.simTime = 100;
problem.simStepSize = 0.1;


%% load pre-computed data
load("VDP_horizon5s.mat")

gamma = 1;
problem.gamma = gamma;

%% define cost function
problem.R = 2.5;
problem.Q = diag([1 1]);

%% define problem
% variables
x  = SX.sym('x',2,1);
u  = SX.sym('u',1,1);
p  = SX.sym('p',4,1);

nx = length(x);
nu = length(u);

% define reachability storage function
problem.OneStep.V0_low = -inf;
problem.OneStep.V0_up  = 0;


% load from external matlab function
problem.OneStep.V = Function('f',...
    {p,x},...
    {V1(x(1),x(2))});

problem.RK45.V0_low = -inf;
problem.RK45.V0_up  = 0;

% define integration step size MPC
h         = 0.1;
problem.T = 5;           % prediction horizon
problem.N = problem.T/h; % intervals

% path cost
problem.Langrange_term = u'*problem.R*u + x'*problem.Q*x;


%% dynamics
f = [x(2);
    (1-x(1)^2)*x(2)-x(1)];

gx = [0;1];

problem.xdot =  f + gx*u;             

%% path constraints; not used in full horizon formulation, only simple constraints
problem.h0     = 3*x(2)^2 + x(1)^2 -1;
problem.h0_low = -inf;
problem.h0_up  = 0;


%% Initial conditions

x0 = [[-0.35;0.1],[-0.35;0.2],[0.2;0.2],[0.2;-0.1],[0.2;0]];

% initial conditions
problem.x0_low = x0;
problem.x0_up  = x0;

problem.u0_low = -inf;
problem.u0_up  = -inf;


% final conditions; actually no used
problem.xf_low = ones(2,1)*(-inf);
problem.xf_up  = -problem.xf_low;

problem.uf_low  =-inf;
problem.uf_up  = inf;


% simple bounds
problem.x_low =  [ -inf -inf]';
problem.x_up  = [  inf  inf ]';

maxu = 1;
problem.u_low = -maxu;
problem.u_up  =maxu;

problem.K = [];
problem.Nx = length(x);
problem.Nu = length(u);

problem.x = x;
problem.u = u;

%% solve problems with different MPC schemes. Simulations might take some time since we use matlab for simulation

tic
disp('-----------------------------------')
disp('One-step problem')

[iter_all_OneStep,x_sol_all_OneStep ,u_sol_all_OneStep, tEnd_all_OneStep] = solve1StepOpt_MPC(problem,alpha0);

tendSimOneS = toc;
disp(['Simulation time one-step : ' num2str(tendSimOneS) ' s'])
disp('One-step problem done!')

tic
disp('-----------------------------------')
disp('Inf.MPC problem')
[iter_all_infMPC, x_sol_all_infMPC, u_sol_all_infMPC, tEnd_all_infMPC,suffCon_all_infMPC] = inf_MPC_simulation(x0);

tendSimOneS = toc;
disp(['Simulation time Inf.MPC : ' num2str(tendSimOneS) ' s'])
disp('One-step problem done!')

t_short = linspace(0, problem.simTime, (problem.simTime/problem.simStepSize)-1);


%% plotting
close all
clc
load('terminalIngredients.mat','W_fun')


figure(2)
pcontour2(V0,0,[-1 1 -1 1],'b-')
hold on
g = @(x1,x2) 3*x2.^2 + x1.^2 - 1;
pcontour2(g,0,[-1 1 -1 1],'k-')

pcontour2(W_fun,0,[-1 1 -1 1],'g--')
% fcontour(@(x,y) full(W_fun(x,y)), [-1 1], 'g--', 'LevelList', 0)
grid off
legend('V(0,x) = 0','h(x) = 0','g(x) = 0')

% for j = 1:size(x0,2)
%     plot(x_sol_all_OneStep{j}(1,1:iter_all_OneStep{j}),x_sol_all_OneStep{j}(2,1:iter_all_OneStep{j}),'b')
%     plot(x_sol_all_infMPC{j}(1,1:iter_all_infMPC{j}),x_sol_all_infMPC{j}(2,1:iter_all_infMPC{j}),'g')
% end
legend('V(0,x) = 0','g(x) = 0','h(x) = 0')
xlabel('x_1')
ylabel('x_2')

 cleanfigure();
 matlab2tikz('comp_subLvlSets.tex','width','\figW','height','\figH');


figure(4)
grid on
x = linspace(0,problem.simTime,(problem.simTime/problem.simStepSize)-1);
for j = 1:size(x0,2)
    semilogy(t_short(1:iter_all_OneStep{j}-1), tEnd_all_OneStep(1,(1:iter_all_OneStep{j}-1)))
    hold on
    semilogy(t_short(1:iter_all_infMPC{j}-1), tEnd_all_infMPC(1,(1:iter_all_infMPC{j}-1)))
end
xlabel('Simulation Time [s]')
ylabel('Runtime [s]')
legend('Horizon-one','Inf.MPC','Location','northoutside','Orientation','horizontal')
ylim([10^(-6) 10^(0)])


% Evaluation
conv_onestep = cell2mat(iter_all_OneStep);
conv_inf = cell2mat(iter_all_infMPC);


Q = problem.Q;
R = problem.R;

stage = @(x,u) x'*Q*x +u'*R*u;


t_inf_mean = zeros(size(x0,2),1);
t_one_mean = zeros(size(x0,2),1);
t_inf_max = zeros(size(x0,2),1);
t_one_max = zeros(size(x0,2),1);

t_inf_min = zeros(size(x0,2),1);
t_one_min= zeros(size(x0,2),1);

% inf_stage_cost = zeros(size(x0,2),conv_inf(k)-1);
% stage_onestep = zeros(size(x0,2),conv_onestep(k)-1);

for k = 1:size(x0,2)
    t_inf_mean(k)   = mean(tEnd_all_infMPC(k,1:conv_inf(k)-1));
    t_one_mean(k,:) = mean(tEnd_all_OneStep(k,1:conv_onestep(k)-1));

    t_inf_max(k)   = max(tEnd_all_infMPC(k,1:conv_inf(k)-1));
    t_one_max(k,:) = max(tEnd_all_OneStep(k,1:conv_onestep(k)-1));

    t_inf_min(k)   = min(tEnd_all_infMPC(k,1:conv_inf(k)-1));
    t_one_min(k,:) = min(tEnd_all_OneStep(k,1:conv_onestep(k)-1));

    for kk = 1:1:conv_inf(k)-1
        inf_stage_cost(k,kk) = stage(x_sol_all_infMPC{k}(:,kk),u_sol_all_infMPC{k}(:,kk) );
    end

    for kk = 1:1:conv_onestep(k)-1
        stage_onestep(k,kk) = stage(x_sol_all_OneStep{k}(:,kk),u_sol_all_OneStep{k}(:,kk) );
    end


end


%% Evaluate aggregated table
avg_conv_Time       = [mean(conv_inf)*h; mean(conv_onestep)*h];
avg_Int_stage_cost  = [mean(sum(inf_stage_cost,2)); mean(sum(stage_onestep,2))];
comp_Time_mean      = [mean(t_inf_mean); mean(t_one_mean)]*1000;
comp_Time_worst     = [max(t_inf_mean); max(t_one_mean)]*1000;
% Row names
rowNames = {'Inf.MPC','Horizon-one'};

% Create the table
T = table(avg_conv_Time, comp_Time_mean, comp_Time_worst, avg_Int_stage_cost, 'RowNames', rowNames);

% Display the table
disp(T);
