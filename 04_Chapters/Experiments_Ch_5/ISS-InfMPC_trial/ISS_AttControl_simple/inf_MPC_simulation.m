%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak and Torbjørn Cunis
%
%   Short Description: Script to syntheszise caompatible robust CBF and
%                      ISS-CLF for the Van-der-Pol Oscillator. 
%                      
%
%   Needed software: - CasADi 3.6 (for CaSoS)
%                    - CaSoS
%
% ------------------------------------------------------------------------

import casadi.*

clear
clear textprogressbar  % to clear persistent variables
close all
clc

rng(40)
% select a QP solver
qpSolver = 'osqp'; % 'qpOASES','osqp','proxqp', 'qrqp'


% load pre-computed data
load terminalIngredients.mat

%% Setup satellite parameter and dynamics
x  = SX.sym('x',nx,1);
u  = SX.sym('u',nu,1);
w  = SX.sym('w',nw,1);
p  = SX.sym('p',length(x),1);

Nx = length(x);
Nu = length(u);

% ODE solve/simulation
dt0         = 1/10;
simTime     = 2000; % maxTime allowed for simulation
simStepSize = dt0;

xdot_nom = xdot_nom(u(1),u(2),u(3),x(1),x(2),x(3),x(4),x(5),x(6));

% dynamics
xdot = x_dot(u(1),u(2),u(3),w(1),w(2),w(3),x(1),x(2),x(3),x(4),x(5),x(6));

% torque bounds (read in above from terminalIngredients.mat) 
u_low = umin;
u_up  = umax;

%% load weights, terminal penalty and invariant set 
% (weights are used below in cost function;
%  read in above from terminalIngredients.mat)

x_1 = x(1);
x_2 = x(2);
x_3 = x(3);
x_4 = x(4);
x_5 = x(5);
x_6 = x(6);

% pre-computed CBF and CLF and their derivatives
h =    h_fun(x_1, x_2,x_3,x_4,x_5,x_6);
V =    V_fun(x_1, x_2,x_3,x_4,x_5,x_6);


% derivative continous-time CBF constraint
hfun_c = Function('f',{x,u},{jacobian(h,x)*xdot_nom + gamma*h});   

% terminal set just to check if we are in the terminal set
hfun = Function('f',{x},{h});   


% we have a zero sublevel set
h0_low = -1e20;
h0_up  = 0;

% terminal penalty
Phi = Function('f',{x,u},{jacobian(V,x)*xdot_nom});


suffCon_fun = Function('f',{x,u,w},{jacobian(V,x)*xdot - x' * Q * x - u' * R * u - sigma_fun(norm(w,2)) + a_fun(norm(x,2))});


%% fixed-step Runge-Kutta 4 integration for simulation
f = Function('f', {x, u, w}, {xdot});

rk = 4;
dt = dt0/rk;
x0 = SX.sym('x0',size(x));
uk = SX.sym('uk',size(u));
wk = SX.sym('wk',size(w));


% pre-allocate
k1 = SX(length(x),rk);
k2 = SX(length(x),rk);
k3 = SX(length(x),rk);
k4 = SX(length(x),rk);
xk = [x0 SX(length(x),rk)];

% loop over subintervals
for j=1:rk
    k1(:,j) = f(xk(:,j), uk, wk);
    k2(:,j) = f(xk(:,j) + dt*k1(:,j)/2, uk, wk);
    k3(:,j) = f(xk(:,j) + dt*k2(:,j)/2, uk, wk);
    k4(:,j) = f(xk(:,j) + dt*k3(:,j), uk, wk);
    xk(:,j+1) = xk(:,j) + dt*(k1(:,j) + 2*k2(:,j) + 2*k3(:,j) + k4(:,j))/6;
end

xkp1 = Function('fk', {x0 uk  wk}, {xk(:,end)}, {'x0' 'uk' 'wk'}, {'xk'});

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
g = hfun_c(p,uk);   
g = reshape(g,1,size(g,1)*size(g,2));

lbg = h0_low;
ubg = h0_up;

% g = [];
% lbg = [];
% ubg = [];

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

        osqp.eps_abs = 1e-5;
        osqp.eps_rel = 1e-5;
        % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',false, ...
                         'print_problem',false,... 
                         'record_time',true,... % get computation times of qp
                         'osqp',osqp,...
                         'verbose',0);   
        
        solver = qpsol('S', 'osqp', prob,options);
    
    case 'proxqp'
        % suppress all display outputs; only qpoases banner will be plottet twice
        options = struct('print_time',false, ...
                         'print_problem',false,... 
                         'record_time',true,... % get computation times of qp
                         'verbose',0);   
        
        solver = qpsol('S', 'proxqp', prob,options);

    
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
f = Function('f',{x,u,w},{xkp1(x,u,w)});
C = CodeGenerator('sim_mex.c');
C.add(f);
opts = struct('mex', true);

f.generate('sim_mex.c',opts);

mex("sim_mex.c")

%% Simulation preparation
    
    % get 100 initial attitude; rates set to zero because rest-to-rest
    a4 = -0.55;  b4 = 0.55;
    x0_low(4:6,:) = (b4-a4)*rand(3,100)+a4;
    
    % we only consider initial states that lie in the terminal set
    idx = full(hfun(x0_low)) <= 0;

    % reduce to feasible initial conditions
    x0_low = x0_low(:,idx);

    numRuns  = 10;


startSim = tic;

% Pre-allocate storage for multiple runs
x_sol_all   = cell(numRuns, 1);
u_sol_all   = cell(numRuns, 1);
tEnd_all    = nan(numRuns, simTime/simStepSize - 1);
suffCon_all = nan(numRuns, simTime/simStepSize - 1);
iter_all    = cell(numRuns, 1);
Barrier_all = nan(numRuns, simTime/simStepSize);
w_sol_all   = cell(numRuns, 1);

%% Simulations
for j = 1:numRuns
    fprintf('Simulation Run: %d/%d\n', j, numRuns);
    
    % get initial conditions for the j-th run
    x0 = x0_low(:,j); 
    
    % pre-allocated arrays for j-th run
    x_sol_vec_infMPC = zeros(Nx, simTime/simStepSize);
    u_sol_vec_infMPC = zeros(Nu, simTime/simStepSize - 1);
    w_sim            = zeros(nw, simTime/simStepSize - 1); 
    suffCon          = nan(simTime/simStepSize - 1,1);
    Barrier          = nan(simTime/simStepSize,1);

    wmax = 0.1*umax(1);
    % set initial conditions  
    x_sim                  = x0;
    x_sol_vec_infMPC(:,1)  = x0;

    % first initial guess for decision variables
    z0 = zeros(Nu,1); % could be more sophisticated, but totally fine

    textprogressbar(['Simulation for ' char(8706) 'NMPC:']);
    
    Barrier(1) =     full(hfun( x_sim(:,1)));

    for k = 2:simTime/simStepSize
        try
        % Solve continous-time infinetismal-horizon MPC
        [sol] = solver('x0', z0, 'p', x0', 'lbx', lbz, 'ubx', ubz, 'lbg', lbg, 'ubg', ubg);
        
        % Get wall time in seconds
        tEnd_all(j, k-1) = solver.stats.t_wall_total;
        
        % Extract solution
        u_sol = full(sol.x);

        % store for later plotting
        u_sol_vec_infMPC(:,k-1) = u_sol;
        
        
        w_sim(:,k)  = (wmax-(-wmax))*rand(3,1)-wmax;
     
        % Simulate
        x_sim(:,k)      = sim_mex(x_sim(:,k-1), u_sol,w_sim(:,k));
        
        % store solution for later plotting
        x_sol_vec_infMPC(:,k)  = x_sim(:,k);


          % store Euler Angles in degree for plotting and convergence check
        [phi,theta,psi]  =  mrp2eul(x_sim(4:6,k));
       
        % store Eule-angles in degree
        x_sol_vec_infMPC(4:6,k) = [phi,theta,psi]'*180/pi;
        

        % Store sufficient condition i.e. cost of QP
        suffCon(k-1) = full(suffCon_fun(x_sim(:,k),u_sol,w_sim(:,k)));
       
        Barrier(k) =     full(hfun( x_sim(:,k)));

        % check convergence 
        % if norm(x_sol_vec_infMPC(:,k),inf)*180/pi < 0.001  && ... 
        %    norm(u_sol,inf) < 0.001 
        %    break
        % end

        % state for next step
        x0 = x_sim(:,k);

        % use current solution as initial guess for next iteration
        z0 = u_sol;         
        
        % update progress bar
        textprogressbar(k/(simTime/simStepSize)*100);
        catch
            fprintf('Problem at solving problem number %d failed in iteration' , [j,k])

        end

    end
    
    % close current progress bar
    textprogressbar('Progress bar  - termination')

    % Store results for j-th run
    x_sol_all{j}      = x_sol_vec_infMPC;
    iter_all{j}       = k;
    u_sol_all{j}      = u_sol_vec_infMPC;
    suffCon_all(j, :) = suffCon;
    Barrier_all(j,:)  = Barrier;
    w_sol_all{j}      = w_sim;
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

% simple bounds
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;

% re-scale rate constraints (real physical constraints) to deg/s
x_low =  [-omegaMax1*180/pi -omegaMax2*180/pi -omegaMax3*180/pi]';
x_up  =  [ omegaMax1*180/pi  omegaMax2*180/pi  omegaMax3*180/pi]';


% Plot Rates in Degree/second
figure('Name', 'Rates');
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t(1:iter_all{j}), x_sol_all{j}(i,1:iter_all{j})*180/pi, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\omega_%c [°/s]', 'x' + (i-1)));
    grid on;

        % plot gray shadded area and dashed gray lines
    xLimits = [0, t(max([iter_all{:}])) ]; 
    yDashed = x_low(i); 
    plot([0 t(max([iter_all{:}]))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    miny =  x_low(i)+0.5* x_low(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max([iter_all{:}]))]; 
    yDashed =  x_up(i);
    plot([0 t(max([iter_all{:}]))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    maxy = x_up(i)+0.5*x_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max([iter_all{:}])) miny maxy])

end

% Plot Attitude in Euler-Angles in degree
figure('Name', 'Attitude');
Euler_names = {'\phi','\theta','\psi'};
for i = 4:6
    subplot(3,1,i-3);
    hold on;
    for j = 1:numRuns
        plot(t(1:iter_all{j}), x_sol_all{j}(i,1:iter_all{j}), 'Color', colors(j, :));
    end
    % axis([0 t(max([iter_all{:}])) -180 180])
    xlabel('t [s]');
    ylabel([Euler_names{i-3} ' [deg]']);
    grid on;
end

% Plot Control Torques in miliNetwonmeter
t_short = linspace(0, simTime, (simTime/simStepSize)-1);

% set up with torques in mili-Newtonmeter
ulow = u_low';
uup  = u_up';

figure('Name', 'Control')
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        % stored control torques also in mili-Newtonmeter
        plot(t(1:iter_all{j}-1), u_sol_all{j}(i,(1:iter_all{j}-1)), 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel('u');
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


figure('Name', 'Disturbance')
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        % stored control torques also in mili-Newtonmeter
        plot(t(1:iter_all{j}-1), w_sol_all{j}(i,(1:iter_all{j}-1)), 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel('w');
    grid on;

    % % plot gray shadded area and dashed gray line
    % xLimits = [0, t(max(iter_all{j}))]; 
    % yDashed = -wmax; 
    % % miny =  -wmax+0.5*(-wmax);
    % plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    % fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    % 
    % xLimits = [0, t(max(iter_all{j}))]; 
    % yDashed =  wmax(i); 
    % plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5]) 
    % maxy = wmax+0.5*wmax;
    % fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    % % axis([0 t(max(iter_all{j})) miny maxy])
end


% Plot Sufficient Conditions evaluted along trajectories
figure('Name', 'Sufficient Conditions (CLF)');
plot([0 t(max(iter_all{j}))], [0 0.0], 'k--');
axis([0 t(max(iter_all{j})) minSuffCond 0.1])
hold on;
for j = 1:numRuns
    plot(t(1:iter_all{j}-1), suffCon_all(j,(1:iter_all{j}-1)), 'Color',colors(j, :));
end
xlabel('t [s]');
ylabel('\nabla_x F \cdot f(x,u)');


% Plot barrier evaluted along trajectories; should be B(x(t)) <= 0
figure('Name', 'Barrier (CBF)');
plot([0 t(max(iter_all{j}))], [0 0.0], 'k--');
hold on;
for j = 1:numRuns
       plot(t(1:iter_all{j}), Barrier_all(j,(1:iter_all{j})), 'Color',colors(j, :));
end
xlabel('t [s]');
ylabel('h(x)');


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
