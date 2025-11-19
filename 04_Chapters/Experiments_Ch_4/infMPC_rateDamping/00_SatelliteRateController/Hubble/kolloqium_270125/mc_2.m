
clear 
clear vars
clear classes
close all
clc

import casadi.*

t  = SX.sym('t');
x  = SX.sym('x',3,1);
u  = SX.sym('u',3,1);
p  = SX.sym('p',5,1);

w  = SX.sym('w',3,1);

dt0 = 0.1;
simTime     = 1500;
simStepSize = dt0;

alpha = 1;

Q_weight = eye(3);
R_weight = eye(3)*0.001;


V = -1 + 769773*x(1)^4 + 182157*x(1)^2*x(2)^2 + 1.07768e+07*x(2)^4 + 734354 ...
  *x(1)^2*x(3)^2 + 1.56626e+07*x(2)^2*x(3)^2 + 1.07768e+07*x(3)^4;


Vfun = Function('f',...       
            {x, t},...     
            {V});   % dt = T/N, k = 1 ( first discrete step) 

V0_low = -inf;
V0_up  = 0;


I = diag([31046;77217;78754]);

% skew-symmetric matrix
skew = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

xdot = -I\skew(x(1:3))*I*x(1:3) + I\u;

numRuns = 5;
% Initialize arrays to store results for each Monte Carlo run
all_x_sol_vec = cell(numRuns, 1);
all_u_sol_vec = cell(numRuns, 1);
all_x_sim = cell(numRuns, 1);
all_tEnd = cell(numRuns, 1);
all_suffCon = cell(numRuns, 1);

max_rate = 0.8*pi/180;
min_rate = -0.8*pi/180;

Nx = 3;
Nu = 3;

textprogressbar('MC Simulation for horizon-one MPC:');

for runIdx = 1:numRuns
    x0_low = (max_rate -min_rate ).*rand(3,1) + min_rate ;

    % x0_low = [omegaMax1 omegaMax2 omegaMax3]';

    if full(Vfun(x0_low,0)) > 0
        continue;  % Skip this run if the condition is not met
    end


u_low = [-1 -1 -1]'*1;
u_up  = [ 1  1 1]'*1;

Nx = length(x);
Nu = length(u);

Phi = Function('f',...        
            {x},...     
            {   11175*x(1)^2 + 99922.4*x(2)^2 + 103402*x(3)^2});   


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

%% for simlation i.e. real plant
f = Function('f', {x, u,w}, {xdot + w}); 

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
    k1(:,j) = f(xk(:,j), uk,wk);
    k2(:,j) = f(xk(:,j) + dt*k1(:,j)/2, uk,wk);
    k3(:,j) = f(xk(:,j) + dt*k2(:,j)/2, uk,wk);
    k4(:,j) = f(xk(:,j) + dt*k3(:,j), uk,wk);
    xk(:,j+1) = xk(:,j) + dt*(k1(:,j) + 2*k2(:,j) + 2*k3(:,j) + k4(:,j))/6;
end

sim = Function('fk', {x0 uk wk}, {xk(:,end)}, {'x0' 'uk' 'wk'}, {'xk'});


%% decision variables
X = SX.sym('X', [length(x) 1]);
U = SX.sym('U', [length(u) 1]);

z = [X(:) ;U(:)];

% state constraints are encoded in RS
lbz =  [-inf(Nx,1); u_low];
ubz =  [+inf(Nx,1); u_up];

%% path constraints
% dynamics
g = X - xkp1(p(3:end), U); %  multiple shooting
g = reshape(g,1,size(g,1)*size(g,2));

uk = U;

% cost function
J =  p(3:end)' * Q_weight * p(3:end) + uk' * R_weight * uk + Phi(X) * p(2);

% equality constraint
lbg_dyn = zeros(1,size(g,1)*(size(g,2)));
ubg_dyn = lbg_dyn;


% add path constraints
V = Vfun(X,dt);   % dt = T/N, k = 1 ( first discrete step) 
V = reshape(V,1,size(V,1)*size(V,2));

g = [g,V];

lbg_cust = V0_low;
ubg_cust = V0_up;


% combine path constraints (defect constraints and user defined in-/equality constraints
lbg = [lbg_dyn,lbg_cust];
ubg = [ubg_dyn,ubg_cust];


% setup solver
prob   = struct('f', J,...
                'x', z,...
                'g', g,...
                'p',p);


% options = struct('ipopt',struct('print_level',0), ...
%                  'print_time',false,...
%                  'record_time',true);


options = struct('print_status',0,...
                 'print_header',0,...
                 'print_time',0,...
                 'record_time',true,...
                 'verbose_init',0,...
                 'print_iteration',0,...
                 'qpsol','qpoases');

options.qpsol_options.printLevel = 'none';


disp('Setup solver ...')
% solver = casadi.nlpsol('solver', 'ipopt', prob,options);
solver = casadi.nlpsol('solver', 'sqpmethod', prob,options);
disp('Solver setup succesful!')


%% generate simulation mex

f = Function('f',{x,u,w},{sim(x,u,w)});
C = CodeGenerator('sim_mex.c');
C.add(f);
opts = struct('mex', true);

f.generate('sim_mex.c',opts);

% if ~exist('sim_mex.mexw64','file')
   mex("sim_mex.c")
% end

% sufficient condition
xk1 = SX.sym('xk1',size(x));
f = Function('f',{x0,xk1,u},{Phi(x0) - Phi(xk1) - x0'*Q_weight*x0 - u' * R_weight * u});
C = CodeGenerator('suffCon_mex.c');
C.add(f);
opts = struct('mex', true);

f.generate('suffCon_mex.c',opts);
if ~exist('suffCon_mex.mexw64','file')
    mex("suffCon_mex.c")
end

%% Simulation
    x0           = x0_low;
    x_sol_vec(:,1)  = x0_low;


    % initial guess
    z0 = zeros(Nx+Nu,1);

    simSteps = simTime/simStepSize; 
    
    u_sol_vec = zeros(3,simSteps-1);
    wk        = zeros(3,simSteps);
    tEnd = nan(simSteps-1,1);
    suffCon = nan(simSteps-1,1);

   x_sim(:,1) = x0_low;

    % profile -memory on;

startSim =tic;
% fprintf(1,'here''s my integer:  ');
    for k = 2:simSteps

       % solve one-step MPC
        [sol]   = solver('x0',  z0,...
                       'p',   [dt0, alpha,x0'],...  %[dt,alpha,x]
                       'lbx', lbz,...
                       'ubx', ubz,...
                       'lbg', lbg,...
                       'ubg', ubg);

      
        tEnd(k-1) = solver.stats.t_wall_total;
      

        z_opt = full(sol.x);
        x_sol = z_opt(1:Nx);
        u_sol = z_opt(Nx+1:end);

        % suffCon(k-1) = (full(Phi(x0))-full(Phi(xkp1(x_sim(:,k-1),u_sol)))) - x0'*Q_weight*x0 - u_sol' * R_weight * u_sol;
                
        x_sol_vec(:,k)   = x_sol;
        u_sol_vec(:,k-1) = u_sol;

        % additive disturbance: angular acceleration
        % lower = u_low(1)/I(1,1)*0.04;
        % upper = u_up(1)/I(1,1)*0.04;

        % wk(:,k) = lower + (upper-lower)*randn(3,1);

        x_sim(:,k) = sim_mex(x_sim(:,k-1),u_sol,wk(:,k)) ;
        
        suffCon(k-1) = suffCon_mex(x0,x_sim(:,k),u_sol);
        
        % if suffCon(k-1) < 0 
        %     disp('Sufficient condition violated')
        %     return
        % end

        x0 = x_sim(:,k);
        
         % initial guess based on old solution         
         z0 = [x_sol;u_sol];

    end
 
    % Store results for this Monte Carlo run
    all_x_sol_vec{runIdx} = x_sol_vec;
    all_u_sol_vec{runIdx} = u_sol_vec;
    all_x_sim{runIdx} = x_sim;
    all_tEnd{runIdx} = tEnd;
    all_suffCon{runIdx} = suffCon;

    textprogressbar(runIdx/numRuns*100);
end
%%
% max_rate = max_rate*180/pi;
% min_rate = min_rate*180/pi;

% simple bounds
% omegaMax1 = 6/60;
% omegaMax2 = 6/60;
% omegaMax3 = 6/60;

x_low =  [-2  -1 -1]';
x_up  =  [ 2   1  1]';


% Time vector
t = linspace(0, simTime, (simTime / simStepSize));

% States Plot
figure('Name', 'States')
for runIdx = 1:numRuns
    % Plot each run's state values for each component
    subplot(311)
    plot(t, all_x_sim{runIdx}(1,:) * 180 / pi)  % omega_x
    hold on

    subplot(312)
    plot(t, all_x_sim{runIdx}(2,:) * 180 / pi)  % omega_y
    hold on

    subplot(313)
    plot(t, all_x_sim{runIdx}(3,:) * 180 / pi)  % omega_z
    hold on
end


% Shading region for state bounds
subplot(311)
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = x_low(1); % y value where the dashed line is
miny =  x_low(1)+0.5* x_low(1);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  x_up(1); % y value where the dashed line is
maxy = x_up(1)+0.5*x_up(1);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])
xlabel('t [s]')
ylabel('\omega_x [^\circ/s]')
legend off
% Shading region for state bounds
subplot(312)
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = x_low(2); % y value where the dashed line is
miny =  x_low(2)+0.5* x_low(2);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  x_up(2); % y value where the dashed line is
maxy = x_up(2)+0.5*x_up(2);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])
xlabel('t [s]')
ylabel('\omega_y [^\circ/s]')
legend off
subplot(313)
hold on
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = x_low(3); % y value where the dashed line is
miny =  x_low(3)+0.5* x_low(3);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  x_up(3); % y value where the dashed line is
maxy = x_up(3)+0.5*x_up(3);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])
xlabel('t [s]')
ylabel('\omega_z [^\circ/s]')
legend off

% matlab2tikz('states.tex','width','\figW','height','\figH');

t = linspace(0,simTime,(simTime/simStepSize)-1);

% set up with torques in miliNetwonmeter

u_low = u_low*1000;
u_up = u_up*1000;

% Controls Plot
figure('Name', 'Control')
for runIdx = 1:numRuns
    % Plot each run's control values
    subplot(311)
    plot(t, all_u_sol_vec{runIdx}(1,:) * 1000)  % tau_x
    hold on

    subplot(312)
    plot(t, all_u_sol_vec{runIdx}(2,:) * 1000)  % tau_y
    hold on

    subplot(313)
    plot(t, all_u_sol_vec{runIdx}(3,:) * 1000)  % tau_z
    hold on
end
% Shading region for control bounds
subplot(311)
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = u_low(1); % y value where the dashed line is
miny =  u_low(1)+0.5* u_low(1);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  u_up(1); % y value where the dashed line is
maxy = u_up(1)+0.5*u_up(1);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])
xlabel('t [s]')
ylabel('\tau_y [mNm]')
legend off


subplot(312)
hold on
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = u_low(1); % y value where the dashed line is
miny =  u_low(2)+0.5* u_low(2);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  u_up(1); % y value where the dashed line is
maxy = u_up(2)+0.5*u_up(2);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])
xlabel('t [s]')
ylabel('\tau_y [mNm]')
legend off

subplot(313)
hold on
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = u_low(3); % y value where the dashed line is
miny =  u_low(3)+0.5* u_low(3);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  u_up(3); % y value where the dashed line is
maxy = u_up(3)+0.5*u_up(3);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])
xlabel('t [s]')
ylabel('\tau_z [mNm]')
legend off


% matlab2tikz('control.tex','width','\figW','height','\figH');

% Sufficient Conditions Plot
figure('Name', 'Sufficient Conditions')
for runIdx = 1:numRuns
    plot(t, all_suffCon{runIdx}, 'b')
    hold on
end
plot([0 t(end)], [0 0], 'k--')
xlabel('t [s]')
ylabel('\Delta P(\cdot) - W(\cdot)')
% axis([0 simTime -1e-4 5e-4])
legend('Sufficient Condition', 'Zero line')
legend off

% matlab2tikz('suffCon.tex','width','\figW','height','\figH');

% Solve Time Plot
figure('Name', 'Solve Time')
for runIdx = 1:numRuns
    semilogy(t, all_tEnd{runIdx}, 'b')
    hold on
end
xlabel('simulation time [s]')
ylabel('computation time [s]')
axis([0 simTime 1e-5 1])

% legend off

% matlab2tikz('comptTime.tex','width','\figW','height','\figH');

% Compute min, mean, max for computation time and sufficient condition
all_tEnd_allRuns = cell2mat(all_tEnd');
all_suffCon_allRuns = cell2mat(all_suffCon');

fprintf('Maximum solve time: %d ms\n', max(all_tEnd_allRuns(:)) * 1000)
fprintf('Mean solve time: %d ms\n', mean(all_tEnd_allRuns(:)) * 1000)
fprintf('Minimum solve time: %d ms\n', min(all_tEnd_allRuns(:)) * 1000)

fprintf('Minimum value of sufficient condition: %d \n', min(all_suffCon_allRuns(:)))
fprintf('Maximum value of sufficient condition: %d \n', max(all_suffCon_allRuns(:)))
fprintf('Mean value of sufficient condition: %d \n', mean(all_suffCon_allRuns(:)))
