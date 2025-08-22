close all
clear
clc

import casadi.*


%% Simulation parameter
problem.simTime = 50;
problem.simStepSize = 0.1;


%% load pre-computed data
load("VDP_horizon5s.mat")


gamma = 1;
problem.gamma = gamma;
%% define cost function
problem.R = 1;
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
                             {p(1)*V1(x(1),x(2))});  


problem.RK45.V0_low = -inf;
problem.RK45.V0_up  = 0;

% define integration step size MPC
h = 0.1;
problem.T = 5;          % prediction horizon
problem.N = problem.T/h; % intervals

% path cost
problem.Langrange_term = u'*problem.R*u + x'*problem.Q*x;

% terminal region
P = [6.4314    0.4580
    0.4580    5.8227];

% terminal set was computed via constrained ROA;
problem.P = P;
problem.Phi = Function('f',...      % Name
            {x, u,},...             % Input variables
            {x'*P*x});              


%% dynamics
f = [x(2);
        (1-x(1)^2)*x(2)-x(1)];

gx = [0;1];

problem.xdot =  f +gx*u;              % MRP kinematics

%% path constraints; not used in full horizon formulation, only simple constraints
problem.h0     = 3*x(2)^2 + x(1)^2 -1;
problem.h0_low = -inf;
problem.h0_up  = 0;


%% Initial conditions

x0 = [0.3;0.1];
% check if we lie inisde the backward reachable set
V0(x0(1),x0(2)) % V(0,x) \leq 0

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

% real-time iteration scheme using IPOPT with iter max = 1
tic 
disp('-----------------------------------')
disp('RTI problem')

problem.RTI_maxSteps = 1;
[t_vec_Full_RTI,xsim_RTI,u_sol_vec_RTI] = solveOCP_RK45_MPC_RTI(problem);

tendSimRTI = toc;
disp(['Simulation time RTI: ' num2str(tendSimRTI) ' s'])
disp('RTI problem done!')


% % real-time iteration scheme using IPOPT with iter max = 3
% tic 
% disp('-----------------------------------')
% disp('RTI problem')
% 
% problem.RTI_maxSteps = 3;
% [t_vec_Full_RTI2,xsim_RTI2,u_sol_vec_RTI2] = solveOCP_RK45_MPC_RTI(problem);
% 
% tendSimRTI2 = toc;
% disp(['Simulation time RTI: ' num2str(tendSimRTI2) ' s'])
% disp('RTI problem done!')

%full horizon problem RK45 using IPOPT
tic
disp('-----------------------------------')
disp('Full hor. problem')

[t_vec_Full,xsim_Full,u_sol_vec_Full] = solveOCP_RK45_MPC(problem);

tendSimFull = toc;
disp(['Simulation time Full hor.: ' num2str(tendSimFull) ' s'])
disp('Full hor. problem done!')

% one-step RK45 using IPOPT with alpha to guarantee sufficient conditions
tic
disp('-----------------------------------')
disp('One-step problem')

[t_vec_OneStep, xsim_onestep,u_sol_vec_onestep,suffCon] = solve1StepOpt_MPC(problem,alpha0);

tendSimOneS = toc;
disp(['Simulation time one-step : ' num2str(tendSimOneS) ' s'])
disp('One-step problem done!')

% alpha set to 1 
tic
disp('-----------------------------------')
disp('One-step problem bad alpha')

alpha = 1;
[t_vec_OneStep_alpha1, xsim_onestep_alpha1,u_sol_vec_onestep_alpha1,suffCon_badalpha] = solve1StepOpt_MPC(problem,alpha);

tendSimOneS2 = toc;
disp(['Simulation time one-step bad alpha: ' num2str(tendSimOneS2) ' s'])
disp('One-step bad alpha problem done!')




%% plotting
close all
clc

t = linspace(0,problem.simTime,(problem.simTime/problem.simStepSize));
figure(1)
subplot(211)
plot(t,xsim_onestep(1,:),'b')
subplot(212)
plot(t,xsim_onestep(2,:),'b')
xlabel('Simulation Time [s]')

figure(2)
% domain = [-1 1 -1 1];
%  Nx = 1000;
%  Ny = 1000;
% xg = linspace(domain(1),domain(2),Nx);
% yg = linspace(domain(3),domain(4),Ny);
% [xg,yg] = meshgrid(xg,yg);
% 
% % evaluate casadi function
% pgrid = full(V0(xg(:)',yg(:)'));
% 
% % reshape to grid
% pgrid = reshape(pgrid,size(xg));
% contour(xg,yg,pgrid,0,'k-')
pcontour2(V0,0,[-1 1 -1 1],'k-')
figure(2)
g = @(x1,x2) 3*x2.^2 + x1.^2 -1;
% fcontour(@(x,y) full(V0(x,y)), [-1 1], 'k-', 'LevelList', 0)
hold on
pcontour2(V1,0,[-1 1 -1 1],'k--')
pcontour2(g,0,[-1 1 -1 1],'k--')
% fcontour(@(x,y) full(V1(x,y)), [-1 1], 'k--', 'LevelList', 0)
% fcontour(@(x,y) full(g(x,y)), [-1 1], 'g-', 'LevelList', 0)
grid off
legend('V(0,x) = 0','V(1,x) = 0','g(x) = 0')

% figure(3)
% g = @(x1,x2) 3*x2^2 + x1^2 -1;
% fcontour(@(x,y) full(V0(x,y)), [-1 1], 'k-', 'LevelList', 0)
% hold on
% fcontour(@(x,y) full(V1(x,y)), [-1 1], 'k--', 'LevelList', 0)
% fcontour(@(x,y) full(g(x,y)), [-1 1], 'g-', 'LevelList', 0)
% % hold off
% grid off
plot(xsim_onestep(1,:),xsim_onestep(2,:),'b')
% plot(xsim_onestep_alpha1(1,:),xsim_onestep_alpha1(2,:),'b--')
plot(xsim_Full(1,:),xsim_Full(2,:),'r-')
plot(xsim_RTI(1,:),xsim_RTI(2,:),'r-+')
legend('V(0,x) = 0','V(1,x) = 0','g(x) = 0', ...
    'Horizon-one \alpha = 126', ...
    'NMPC',...
    'RTI')
xlabel('x_1')
ylabel('x_2')
cleanfigure()
matlab2tikz('TrajVdp_horizon1.tex','width','\figW','height','\figH')


figure(4)
grid on
x = linspace(0,problem.simTime,(problem.simTime/problem.simStepSize)-1);

y = t_vec_Full;
semilogy(x,y,'k')
hold on

y = t_vec_Full_RTI;
semilogy(x,y,'g')

y = t_vec_OneStep;
semilogy(x,y,'b')

% y = t_vec_OneStep_alpha1;
% semilogy(x,y,'b')

xlabel('Simulation Time [s]')
ylabel('Runtime [s]')
legend('Full','RTI ','Horizon-one','Location','northoutside','Orientation','horizontal')
ylim([10^(-4) 10^(0)])


cleanfigure()
matlab2tikz('RunTimeCompVDP.tex','width','\figW','height','\figH')


 disp(['Ratio min. time RTI to max. time Horizon One: ' num2str( min(t_vec_Full_RTI)/ max(t_vec_OneStep))])
 disp(['Ratio min. time RTI to mean time Horizon One: ' num2str( min(t_vec_Full_RTI)/ mean(t_vec_OneStep))])
 disp(['Ratio min. time RTI to min. time Horizon One: ' num2str( min(t_vec_Full_RTI)/ min(t_vec_OneStep))])