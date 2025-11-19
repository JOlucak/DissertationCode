
clear
close all
clear textprogressbar
clc

import casadi.*

% state, control and parameter
x  = SX.sym('x',3,1);
u  = SX.sym('u',3,1);
p  = SX.sym('p',length(x),1);

Nx = length(x);
Nu = length(u);


Q = eye(3)*100;
R = eye(3)*0.001;        

% ODE solve/simulation
dt0         = 0.1;
simTime     = 1000;
simStepSize = dt0;

% satellite dynamics (rates and MRP kinematics)
I = diag([31046;77217;78754]);

% cross-product matrix
cpm = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];


xdot =  -I\cpm(x(1:3))*I*x(1:3) + I\u;

% bounds for rates (needed later for plotting)
omegaMax1 = 2*pi/180;
omegaMax2 = 1*pi/180;
omegaMax3 = 1*pi/180;

% torque bounds 
u_low = [-1 -1 -1]'*1;
u_up  = [ 1  1  1]'*1;

% pre-computed terminal set
% load terminalset.mat
x_1 = x(1);
x_2 = x(2);
x_3 = x(3);

V = -1 + 756804*x_1^4 + 118815*x_1^2*x_2^2 + 1.07768e+07*x_2^4 + 644429 ...
  *x_1^2*x_3^2 + 1.61543e+07*x_2^2*x_3^2 + 1.07768e+07*x_3^4;

P = 537357*x_1^2  + 1.9603e+06*x_2^2  + 2.01983e+06*x_3^2;

% continous time terminal set constraint
Vfun_c = Function('f',{x,u},{jacobian(V,x)*xdot});   

% terminal set just to check if we are in the terminal set
Vfun = Function('f',{x},{V});   

V0_low = -inf;
V0_up  = 0;


% load MPC_penalty.mat

% terminal penalty
Phi = Function('f',...
    {x,u},...
    {jacobian(P,x)*xdot});

Pfun = Function('f',{x},{P});   
% fixed-step Runge-Kutta 4 integration  for simlation i.e. real plant
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

% vector of decision varianles
z = uk;

% just for optimizer; actual state constraints are encoded in terminal set
lbz =  u_low;
ubz =  u_up;

%% path constraints
% cost function
J =  p'*Q*p + uk'*R*uk + Phi(p,uk) ;

% add path constraints i.e. terminal constraint
g = Vfun_c(p,uk);   
g = reshape(g,1,size(g,1)*size(g,2));

lbg = -inf;
ubg= 0;

% setup solver
prob   = struct('f', J,...
    'x', z,...
    'g', g,...
    'p',p);

%% setup solver
options = struct('print_time',0, ...
                 'print_problem',0,...
                 'record_time',true,...
                 'verbose',0);       % get computation times of qp

solver = qpsol('S', 'osqp', prob,options);


%% generate matlab mex for simulation (just for speed up in matlab)
fprintf('Pre-compile mex-files for simulation and sufficient conditions.\n')
f = Function('f',{x,u},{xkp1(x,u)});
C = CodeGenerator('sim_mex.c');
C.add(f);
opts = struct('mex', true);

f.generate('sim_mex.c',opts);

mex("sim_mex.c")


%% Monte-Carlo Simulation

numRuns  = 1;       % Define number of Monte-Carlo runs
startSim = tic;

% Pre-allocate storage for multiple runs
x_sol_all = cell(numRuns, 1);
u_sol_all = cell(numRuns, 1);
tEnd_all = nan(numRuns, simTime/simStepSize - 1);
suffCon_all = nan(numRuns, simTime/simStepSize - 1);

% Pre-allocation (could be also put outside the for-loop)
x0_low_vec = zeros(3,1,numRuns); 
    
% we could add rates not equal to zero could be used, for the paper we
% only consider rest-to-rest profile i.e. rates equal zero

a1 = -2*pi/180;  b1 = 2*pi/180;
x0_low_vec(1,:) = (b1-a1)*rand()+a1;

a2 = -1*pi/180;  b2 = 1*pi/180;
x0_low_vec(2,:) = (b2-a2)*rand()+a2;
% 
% a3 = -1*pi/180;  b3 = 1*pi/180;
% x0_low_vec(3,:) = (b3-a3)*rand()+a3;
    

%% Run
for j = 1:numRuns
    fprintf('Simulation Run: %d/%d\n', j, numRuns);

    x0_low = x0_low_vec(:,1,j);
    x0     = x0_low;
    % check if we lie in the terminal set. If not, skip
    if full(Vfun(x0_low)) > 0
        disp('Initial state outside of terminal set! ')
        
        % pre-allocated arrays
        x_sol_vec = nan(3, simTime/simStepSize);
        u_sol_vec = nan(3, simTime/simStepSize - 1);
        suffCon  = nan(simTime/simStepSize - 1,1);
        constraintSatisfaction = nan(simTime/simStepSize - 1,1);

        % Store results for each run
        x_sol_all{j} = x_sol_vec;
        u_sol_all{j} = u_sol_vec;
        suffCon_all(j, :) = suffCon;
        continue
    end
    
    % pre-allocated arrays
    x_sol_vec = zeros(3, simTime/simStepSize);
    u_sol_vec = zeros(3, simTime/simStepSize - 1);
    suffCon  = nan(simTime/simStepSize - 1,1);
    constraintSatisfaction = nan(simTime/simStepSize - 1,1);
    J0 = zeros(1, simTime/simStepSize - 1);
    
    % set first step
    x_sol_vec(:,1)  = x0_low;
    x_sim           = x0_low;
    

    % first initial guess for decision variables
    z0 = zeros(Nu,1);

    textprogressbar('Simulation for horizon-one MPC:');

    for k = 2:simTime/simStepSize
        
        
        % Solve continous-time one-step MPC
        [sol] = solver('x0', z0, 'p', x0', 'lbx', lbz, 'ubx', ubz, 'lbg', lbg, 'ubg', ubg);
        
        % Get wall time in seconds
        tEnd_all(j, k-1) = solver.stats.t_wall_total;
        
        % Extract solution
        u_sol = full(sol.x);
        % store for later plotting
        u_sol_vec(:,k-1) = u_sol;
        
        % Simulate
        x_sim(:,k)      = sim_mex(x_sim(:,k-1), u_sol);
        x_sol_vec(:,k) = x_sim(:,k);
        
        % Store sufficient condition
        suffCon(k-1) = full(sol.f);
        
        constraintSatisfaction(k-1) = full(Vfun(x0));

        % state for next step
        x0 = x_sim(:,k);

        J0(k-1) = u_sol'*R*u_sol + x0'*Q*x0;

        % use current solution as initial guess for next iteration
        z0 = u_sol;         
        
        % update progress bar
        textprogressbar(k/(simTime/simStepSize)*100);
    end
    
    textprogressbar('Progress bar  - termination')

    % Store results for each run
    x_sol_all{j} = x_sol_vec;
    u_sol_all{j} = u_sol_vec;
    suffCon_all(j, :) = suffCon;
end

% total time for Monte-carlo
simTimeMeas = toc(startSim);
fprintf('\nTotal Simulation time: %f seconds\n', simTimeMeas);

%% Compute statistics on computation time in miliseconds
tEnd_all(isnan(tEnd_all)) = [];

minSolveTime = min(tEnd_all, [], 'all') * 1000;
maxSolveTime = max(tEnd_all, [], 'all') * 1000;
meanSolveTime = mean(tEnd_all, 'all') * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime);
fprintf('Maximum solve time: %f ms\n', maxSolveTime);
fprintf('Mean solve time: %f ms\n', meanSolveTime);

% Compute statistics on sufficient condition
minSuffCond = min(suffCon_all, [], 'all'); % just for plotting
maxSuffCond = max(suffCon_all, [], 'all');
fprintf('Maximum value of sufficient condition: %f\n', maxSuffCond);

%% Plotting
t = linspace(0, simTime, simTime/simStepSize);
colors = lines(numRuns); % Get a colormap for different runs

% re-scale rate constraints (real physical constraints)
x_low =  [-omegaMax1*180/pi -omegaMax2*180/pi -omegaMax3*180/pi]';
x_up  =  [ omegaMax1*180/pi  omegaMax2*180/pi  omegaMax3*180/pi]';

% Plot Rates
figure('Name', 'Rates');
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t(1:k), x_sol_all{j}(i,1:k) * 180/pi, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\omega_%c [°/s]', 'x' + (i-1)));
    % title(sprintf('Rate \\omega_%c', 'x' + (i-1)));
    grid on;

    % plot gray shadded area
    xLimits = [0, t(k)]; 
    yDashed = x_low(i); 
    plot([0 t(k)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    miny =  x_low(i)+0.5* x_low(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    % xLimits = [0, simTime]; 
    yDashed =  x_up(i);
    plot([0 t(k)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    maxy = x_up(i)+0.5*x_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(k) miny maxy])
end

% Plot Control Torques
t_short = linspace(0, simTime, (simTime/simStepSize)-1);
% set up with torques in miliNetwonmeter
u_low = [-1 -1 -1]'*1000;
u_up  = [ 1  1 1]'*1000;

figure('Name', 'Control Torques');
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t_short(1:k-1), u_sol_all{j}(i,1:k-1) * 1000, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\tau_%c [mNm]', 'x' + (i-1)));
    % title(sprintf('Control Torque \\tau_%c', 'x' + (i-1)));
    grid on;

    % plot gray shadded area
    xLimits = [0, t(k-1)]; 
    yDashed = u_low(1); 
    miny =  u_low(i)+0.5* u_low(i);
    plot([0 t(k-1)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(k-1)]; 
    yDashed =  u_up(i); 
    plot([0 t(k-1)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5]) 
    maxy = u_up(i)+0.5*u_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(k-1) miny maxy])

end

% Plot Sufficient Conditions evaluted along trajectories
figure('Name', 'Sufficient Conditions');
plot([0 t(k)], [0 0.0], 'k--');
axis([0 t(k) minSuffCond 0.1])
hold on;
for j = 1:numRuns
    plot(t_short(1:k-1), suffCon_all(j,1:k-1), 'Color',[0.7 0.7 0.7]);
end
xlabel('t [s]');
ylabel('\nabla_x F \cdot f(x,u)');
% custom legend
h = zeros(1, 1);
h(1) = plot(NaN,NaN,'k--');
% h(2) = plot(NaN,NaN,'b-');
legend(h, 'Zero line');

% Eval cost
figure('Name','Evaluate penalty along trajectory.')
plot(t,full(Pfun(x_sol_vec)), 'k');
ylabel('V(x)')
xlabel('t [s]');



% Plot Solve Time
figure('Name', 'Solve Time');
for j = 1:numRuns
    semilogy(t_short(1:k-1), tEnd_all(j,1:k-1), 'Color', colors(j, :));
    hold on;
end
xlabel('Simulation time [s]');
ylabel('Computation time [s]');
axis([0 t_short(k-1) 1e-6 1])
grid on;


