%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Script to execute Monte-carlo simulations for the 
%                      infinitesimal-horizon MPC. The problem is setup
%                      using CasADi. The pre-computed CLF and CBF
%                      as well as the weights Q and R are and spacecraft
%                      specific data (bounds, inertia tensor) are
%                      loaded from a terminalIngredients.mat file. 
%                      We make use of CasADi's qpsol interface. Different
%                      simulation modes are possible (single, MC and
%                      comparison). For single and comparsion, the user
%                      must select initial values such that B(x) <= 0, i.e,
%                      we must start in the CBF sublevel set (safe set).
%                      A .mex file is generated for the simulation (RK45) 
%                      to improve simulation time. 
%
%                       We want to solve the following QP:
%
%                       min_u  x^T Q x + u^T R u + V_x f(x,u)
%                       s.t.   W_x f(x,u) + gamma(W(x)) <= 0 
%                              u \in U
% 
%
%                       where Q, R are pre-defined, V and W are
%                       pre-computed terminal penalty and invariant set
%                       respectively. u is the decision variable and x is
%                       the current state.
%
%
%   Needed software: - CasADi 3.6 
%                    - A C/C++ compiler for .mex file generation (We use a
%                      MS 2022 C/C++ compiler) 
%
%
% ------------------------------------------------------------------------

import casadi.*

clear
clear textprogressbar  % to clear persistent variables
close all
clc

% functions to convert e.g. from MRP to Euler angles
addpath('..\helperFunc\')

% select a QP solver
qpSolver = 'qrqp'; % 'qpOASES','osqp','proxqp', 'qrqp'


SimulationType = 'MC'; % 'MC': Monte-carlo with uniform distribution
                               % 'Comparison': Several Pre-selected inital conditions
                               % 'Single': Single run

% load pre-computed data
load terminalIngredients.mat

%% Setup satellite parameter and dynamics
x  = SX.sym('x',2,1);
u  = SX.sym('u',1,1);
p  = SX.sym('p',length(x),1);

Nx = length(x);
Nu = length(u);

% ODE solve/simulation
dt0         = 0.1;
simTime     = 1000; % maxTime allowed for simulation
simStepSize = dt0;

% dynamics
mu =1;

% dynamics
xdot = [x(2);
      mu*(1-x(1)^2)*x(2) - x(1) - u];


% torque bounds (read in above from terminalIngredients.mat) 
u_low = umin;
u_up  = umax;

%% load weights, terminal penalty and invariant set 
% (weights are used below in cost function;
%  read in above from terminalIngredients.mat)

x_1 = x(1);
x_2 = x(2);


% pre-computed CBF and CLF and their derivatives
W =    W_fun(x_1, x_2);
V =    V_fun(x_1, x_2);

% derivative continous-time CBF constraint; gamma = 0.0001
Wfun_c = Function('f',{x,u},{jacobian(W,x)*xdot + 0.0001*W});   

% terminal set just to check if we are in the terminal set
Wfun = Function('f',{x},{W});   

% we have a zero sublevel set
W0_low = -inf;
W0_up  = 0;

% terminal penalty
Phi = Function('f',{x,u},{jacobian(V,x)*xdot});

%% fixed-step Runge-Kutta 4 integration for simulation
f = Function('f', {x, u}, {xdot});

rk = 4;
dt = dt0/rk;
x0 = SX.sym('x0',size(x));
uk = SX.sym('uk',size(u));

% pre-allocate
k1 = SX(length(x),rk);
k2 = SX(length(x),rk);
k3 = SX(length(x),rk);
k4 = SX(length(x),rk);
xk = [x0 SX(length(x),rk)];

% loop over subintervals
for j=1:rk
    k1(:,j) = f(xk(:,j), uk);
    k2(:,j) = f(xk(:,j) + dt*k1(:,j)/2, uk);
    k3(:,j) = f(xk(:,j) + dt*k2(:,j)/2, uk);
    k4(:,j) = f(xk(:,j) + dt*k3(:,j), uk);
    xk(:,j+1) = xk(:,j) + dt*(k1(:,j) + 2*k2(:,j) + 2*k3(:,j) + k4(:,j))/6;
end

xkp1 = Function('fk', {x0 uk}, {xk(:,end)}, {'x0' 'uk'}, {'xk'});

%% decision variables
uk = SX.sym('U', [length(u) 1]);

% vector of decision variables (just to distinguish it)
z = uk;

% box-constraints for controls
lbz =  u_low;
ubz =  u_up;

%% cost function; p is a parameter vector, here the current state
J =  p' * Q * p + uk' * R * uk + Phi(p,uk);

%% path constraints i.e. CBF derivative
g = Wfun_c(p,uk);   
g = reshape(g,1,size(g,1)*size(g,2));

lbg = W0_low;
ubg = W0_up;

%% setup QP solver

% problem struct
prob   = struct('f', J,...
                'x', z,...
                'g', g,...
                'p',p);

% setup QP solver; leave at default values but turn of display output
switch qpSolver

    case 'qpOASES'
        % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',0, ...
                         'printLevel','none',...
                         'print_problem',0,... 
                         'record_time',true,... % get computation times of qp
                         'verbose',0);   

        solver = qpsol('S', 'qpoases', prob,options);

    case 'osqp'
        % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',false, ...
                         'print_problem',false,... 
                         'record_time',true,... % get computation times of qp
                         'verbose',0);   
        
        solver = qpsol('S', 'osqp', prob,options);
    
    case 'proxqp'
        % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',false, ...
                         'print_problem',false,... 
                         'record_time',true,... % get computation times of qp
                         'verbose',0);   
        
        solver = qpsol('S', 'proxqp', prob,options);

    case 'hpipm' % did not work in the past
        % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',false, ...
                         'print_problem',false,... 
                         'record_time',true,... % get computation times of qp
                         'verbose',0);   
        
        solver = qpsol('S', 'hpipm', prob,options);
    
    case 'qrqp'

                % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',false, ...
                         'print_problem',false,... 
                         'record_time',true,... % get computation times of qp
                         'print_header', false,...
                         'print_out',false,...
                         'print_iter',false,...
                         'verbose',0);   
        
        solver = qpsol('S', 'qrqp', prob,options);


    otherwise % qpOASES as default

        % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',0, ...
                         'printLevel','none',...
                         'print_problem',0,... 
                         'record_time',true,... % get computation times of qp
                         'verbose',0);   
        
        solver = qpsol('S', 'qpoases', prob,options);

end



%% generate matlab mex for simulation (just for speed up in matlab)
fprintf('Pre-compile mex-files for simulation.\n')
f = Function('f',{x,u},{xkp1(x,u)});
C = CodeGenerator('sim_mex.c');
C.add(f);
opts = struct('mex', true);

f.generate('sim_mex.c',opts);

mex("sim_mex.c")

%% Simulation preparation

% currently only setup for rest-to-rest profile; can be changed also to
% consider initial rates
if strcmp(SimulationType,'MC')
    
    % get 100 initial attitude; rates set to zero because rest-to-rest
    a4 = -1;  b4 = 1;
    x0_low = (b4-a4)*rand(2,1000)+a4;
    
    % we only consider initial states that lie in the terminal set
    idx = full(Wfun(x0_low)) <= 0;

    % reduce to feasible initial conditions
    x0_low = x0_low(:,idx);

    numRuns  = 100;

elseif strcmp(SimulationType,'Comparison')
     % single axis rest-to-rest maneuver
     numRuns  = 3;
   
     theta0 = 0;
     psi0   = 0;
     
     % must lie in CBF sublevel set! (not checked here)
     phi0   = [110;90;75]*pi/180;
      
     % we assume rest to rest, thus rate zero
     x0_low = zeros(6,numRuns);
     for j = 1:length(phi0)
        x0_low(4:6,j)  = Euler1232MRP([phi0(j),theta0,psi0]);
     end

     % .mat file name
     matFileName = 'infiniteMPC_comparison_3_runs.mat';

else
    % single run
    numRuns  = 1;      

     % just a single fixed initial attitude
     phi0   = 110*pi/180;
     theta0 = 0;
     psi0   = 0;
    
     x0_low(4:6,1)  = Euler1232MRP([phi0,theta0,psi0]);

     % .mat file name
     matFileName = 'infiniteMPC_comparison.mat';

end


startSim = tic;

% Pre-allocate storage for multiple runs
x_sol_all   = cell(numRuns, 1);
u_sol_all   = cell(numRuns, 1);
tEnd_all    = nan(numRuns, simTime/simStepSize - 1);
suffCon_all = nan(numRuns, simTime/simStepSize - 1);
iter_all    = cell(numRuns, 1);
Barrier_all = nan(numRuns, simTime/simStepSize);


%% Simulations
for j = 1:numRuns
    fprintf('Simulation Run: %d/%d\n', j, numRuns);
    
    % get initial conditions for the j-th run
    x0 = x0_low(:,j); 
    
    % pre-allocated arrays for j-th run
    x_sol_vec_infMPC = zeros(Nx, simTime/simStepSize);
    u_sol_vec_infMPC = zeros(Nu, simTime/simStepSize - 1);
    suffCon          = nan(simTime/simStepSize - 1,1);
    Barrier          = nan(simTime/simStepSize,1);


    % set initial conditions  
    x_sim                  = x0;
    x_sol_vec_infMPC(:,1)  = x0;

    % first initial guess for decision variables
    z0 = zeros(Nu,1); % could be more sophisticated, but totally fine

    textprogressbar('Simulation for horizon-one MPC:');
    
    Barrier(1) =     full(Wfun( x_sim(:,1)));

    for k = 2:simTime/simStepSize

        % Solve continous-time infinetismal-horizon MPC
        [sol] = solver('x0', z0, 'p', x0', 'lbx', lbz, 'ubx', ubz, 'lbg', lbg, 'ubg', ubg);
        
        % Get wall time in seconds
        tEnd_all(j, k-1) = solver.stats.t_wall_total;
        
        % Extract solution
        u_sol = full(sol.x);

        % store for later plotting
        u_sol_vec_infMPC(:,k-1) = u_sol;
        
        % Simulate
        x_sim(:,k)      = sim_mex(x_sim(:,k-1), u_sol);
        
        % store solution for later plotting
        x_sol_vec_infMPC(:,k)  = x_sim(:,k);
        

        % Store sufficient condition i.e. cost of QP
        suffCon(k-1) = full(sol.f);
       
        Barrier(k) =     full(Wfun( x_sim(:,k)));

        % check convergence 
        if norm(x_sol_vec_infMPC(:,k),inf)*180/pi < 1e-4 && ... 
           norm(x_sol_vec_infMPC(:,k),inf) < 1e-4           
           break
        end

        % state for next step
        x0 = x_sim(:,k);

        % use current solution as initial guess for next iteration
        z0 = u_sol;         
        
        % update progress bar
        textprogressbar(k/(simTime/simStepSize)*100);
    end
    
    % close current progress bar
    textprogressbar('Progress bar  - termination')

    % Store results for j-th run
    x_sol_all{j}      = x_sol_vec_infMPC;
    iter_all{j}       = k;
    u_sol_all{j}      = u_sol_vec_infMPC;
    suffCon_all(j, :) = suffCon;
    Barrier_all(j,:)  =  Barrier;
end

% total time for Monte-carlo
simTimeMeas = toc(startSim);
fprintf('\nTotal Simulation time: %f seconds\n', simTimeMeas);

%% Compute statistics on computation time in miliseconds
minSolveTime_infMPC  = min(tEnd_all, [], 'all') * 1000;
maxSolveTime_infMPC  = max(tEnd_all, [], 'all') * 1000;
% Convert matrix to cell array, one cell per row
tEnd_all_cell = mat2cell(tEnd_all, ones(1, size(tEnd_all, 1)), size(tEnd_all, 2));

% Remove NaNs and make each output a column vector
tEnd_all_cleaned = cellfun(@(row) row(~isnan(row))', tEnd_all_cell, 'UniformOutput', false);

% Concatenate all cleaned column vectors and compute the mean in ms
meanSolveTime_infMPC = mean(cell2mat(tEnd_all_cleaned)) * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime_infMPC);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_infMPC);
fprintf('Mean solve time: %f ms\n', meanSolveTime_infMPC);

% Compute statistics on sufficient condition
minSuffCond = min(suffCon_all, [], 'all'); % just for plotting
maxSuffCond = max(suffCon_all, [], 'all');

% this value should be <= 0 
fprintf('Maximum value of sufficient condition: %f\n', maxSuffCond);

%% Plotting
t      = linspace(0, simTime, simTime/simStepSize);
colors = lines(numRuns); % Get a colormap for different runs

% Plot Rates in Degree/second
figure('Name', 'Rates');
for i = 1:2
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t(1:iter_all{j}), x_sol_all{j}(i,1:iter_all{j}) * 180/pi, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\omega_%c [°/s]', 'x' + (i-1)));
    grid on;

end


% Plot Control Torques in miliNetwonmeter
t_short = linspace(0, simTime, (simTime/simStepSize)-1);

% set up with torques in mili-Newtonmeter
ulow = u_low';
uup  = u_up';

figure('Name', 'Control Torques')
for i = 1
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        % stored control torques also in mili-Newtonmeter
        plot(t(1:iter_all{j}-1), u_sol_all{j}(i,(1:iter_all{j}-1)), 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\tau_%c [mNm]', 'x' + (i-1)));
    grid on;

    % plot gray shadded area and dashed gray line
    xLimits = [0, t(max(iter_all{j}))]; 
    yDashed = ulow(i); 
    miny =  ulow(i)+0.5* ulow(i);
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max(iter_all{j}))]; 
    yDashed =  uup(i); 
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5]) 
    maxy = uup(i)+0.5*uup(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max(iter_all{j})) miny maxy])

end

% Plot Sufficient Conditions evaluted along trajectories
figure('Name', 'Sufficient Conditions');
plot([0 t(max(iter_all{j}))], [0 0.0], 'k--');
axis([0 t(max(iter_all{j})) minSuffCond 0.1])
hold on;
for j = 1:numRuns
    plot(t(1:iter_all{j}-1), suffCon_all(j,(1:iter_all{j}-1)), 'Color',colors(j, :));
end
xlabel('t [s]');
ylabel('\nabla_x F \cdot f(x,u)');



% Plot barrier evaluted along trajectories; should be B(x(t)) <= 0
figure('Name', 'Barrier');
plot([0 t(max(iter_all{j}))], [0 0.0], 'k--');
hold on;
for j = 1:numRuns
       plot(t(1:iter_all{j}), Barrier_all(j,(1:iter_all{j})), 'Color',colors(j, :));
end
xlabel('t [s]');
ylabel('W(x)');


% Plot Solve Time
figure('Name', 'Solve Time');

for j = 1:numRuns
    semilogy(t_short(1:iter_all{j}-1), tEnd_all(j,(1:iter_all{j}-1)), 'Color', colors(j, :));
    hold on;
end
xlabel('Simulation time [s]');
ylabel('Computation time [s]');
axis([0 t(max(iter_all{j})) 1e-6 1])
grid on;

% %% store data for comparison
% % to distinguihs it from other approaches
% Q_infinite    = Q;
% R_infinite    = R;
% iter_conv_inf = iter_all;
% tEnd_all_inf  = tEnd_all;
% Barrier_inf   = Barrier_all;
% suffCon_inf   = suffCon_all;
% 
% x_sol_vec_infMPC = x_sol_all;
% u_sol_vec_infMPC = u_sol_all ;
% 
% Omega_bounds  = [omegaMax1;omegaMax2;omegaMax3];
% 
% cd ../
% % store in main folder for comparison
% save(matFileName,...
%     'Q_infinite','R_infinite',...
%     'x_sol_vec_infMPC','u_sol_vec_infMPC',...
%     'u_low','u_up','Omega_bounds',...
%     "simTime","simStepSize","iter_conv_inf","maxSolveTime_infMPC",'tEnd_all_inf','meanSolveTime_infMPC',...
%     'Barrier_inf','suffCon_inf')
% 
% cd ./inf_MPC/
% 
% % remove helper functions from path
% rmpath('..\helperFunc\')
