clc
close all
clear

import casadi.*

solver = 'IPOPT';

%simulation
dt          = 0.25;
simTime     = 400;
simStepSize = dt;

x  = SX.sym('x',3,1);
u  = SX.sym('u',3,1);

Nx = length(x);
Nu = length(u);

% parameter is the current state
p  = SX.sym('p',length(x),1);

% weights for quadratic cost
Q_weight = eye(3)*100;
R_weight = eye(3)*0.01;

% path constraints and bounds
h0      = [];
h0_low  = [];
h0_up   = [];

% path constraints
H = Function('f',...       % Name
             {x,p},...     % Input variables
             {h0});   


% NMPC horizon
T  = 100;
N  = T/dt;

% Satellite Parameter
J = diag([8970;9230;3830]);

% cross product-symmetric matrix
cpm = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

xdot = -J\cpm(x(1:3))*J*x(1:3) + J\u;

% linearize system to compute terminal ingredients
x0 = zeros(3,1);
u0 = zeros(3,1);

A = full(casadi.DM(substitute(jacobian(xdot,x),[x;u],[x0;u0])));
B = full(casadi.DM(substitute(jacobian(xdot,u),[x;u],[x0;u0])));

[K,P] = lqr(A,B,Q_weight,R_weight);

% terminal cost
Phi            = Function('f',...       % Name
                           {x,p},...     % Input variables
                            {x'*P*x});   

% level set for terminal set
gamma          = 1; 

% initial conditions (for simulation)
x0_low = [1;1;1]*pi/180;

% simple bounds state
x_low = [-2 -2 -2]*pi/180;
x_up = -x_low;

% simple bounds control
u_low = [-1 -1 -1];
u_up  = [1 1 1];

%% fixed-step Runge-Kutta 4 integration; compare to casadi example package
f = Function('f', {x, u}, {xdot}); 

rk = 4; % subintervals
dt = T/N/rk;
x0 = SX.sym('x0',size(x));
uk = SX.sym('uk',size(u));

k1 = SX(length(x),rk); 
k2 = SX(length(x),rk); 
k3 = SX(length(x),rk); 
k4 = SX(length(x),rk);
xk = [x0 SX(length(x),rk)];

for j=1:rk
    k1(:,j) = f(xk(:,j), uk);
    k2(:,j) = f(xk(:,j) + dt*k1(:,j)/2, uk);
    k3(:,j) = f(xk(:,j) + dt*k2(:,j)/2, uk);
    k4(:,j) = f(xk(:,j) + dt*k3(:,j), uk);
    xk(:,j+1) = xk(:,j) + dt*(k1(:,j) + 2*k2(:,j) + 2*k3(:,j) + k4(:,j))/6;
end

xkp1 = Function('fk', {x0 uk}, {xk(:,end)}, {'x0' 'uk'}, {'xk'});

%% decision variables 
X = SX.sym('X', [length(x) N]);
U = SX.sym('U', [length(u) N]);

% decision variables
z = [X(:) ;U(:)];

% simple constraints
control_lb_grid = repmat(u_low,1,N);
control_ub_grid = repmat(u_up,1,N);

state_lb_grid = repmat(x_low,1,N);
state_ub_grid = repmat(x_up,1,N);

% simple bounds on decision variables
lbz = [state_lb_grid(:); control_lb_grid(:)];
ubz = [state_ub_grid(:); control_ub_grid(:)];

%% path constraints

% dynamics
g = X - xkp1([x0 X(:,1:N-1)], U);       %  multiple shooting
g = reshape(g,1,size(g,1)*size(g,2));

% equality constraint
lbg_dyn = zeros(1,size(g,2));
ubg_dyn = lbg_dyn;

% add path constraints and terminal constraint
gh = [gamma-X(:,end)'*P*X(:,end)]; % terminal set   
      % H(X(:,1:end),p)] ;          % path constraint

gH = reshape(gh,1,size(gh,1)*size(gh,2));
g = [g,gH];

lbg_cust = repmat(h0_low ,1,N);
ubg_cust = repmat(h0_up ,1,N);

% combine path constraints: defect constraints, terminal set and user defined
% in-/equality (path) constraints
lbg = [lbg_dyn, 0];%, lbg_cust ] ; 
ubg = [ubg_dyn, inf];%, ubg_cust ];


%% cost function
l = Function('l', {x, u}, { x'*Q_weight*x + u'*R_weight*u});

% J = terminal penalty + path cost
J = Phi(X(:,end),zeros(size(u))) + sum( l([x0 X(:,1:end-1)],U(:,1:end)) ); 

%% initial guess for first iteration ( all zeros)
z0 = zeros(N*(Nx+Nu),1);

%% setup solver
prob   = struct('f', J, 'x', z, 'g', g,'p',x0);

disp('Building solver ...')

switch solver
    case 'IPOPT'
        options = struct('ipopt',struct('print_level',0),'print_time',false);
        solver  = casadi.nlpsol('solver', 'ipopt', prob,options);
    case 'SQP'
        options = struct('print_status',0,...
                         'print_header',0,...
                         'print_time',0,...
                         'verbose_init',0,...
                         'print_iteration',0,...
                         'qpsol','qpoases');

       options.qpsol_options.printLevel = 'none';
       solver = casadi.nlpsol('solver', 'sqpmethod', prob,options);
end

disp('Solver building succesful!')

%% Simulation
nx = Nx;
nu = Nu;

simSteps = simTime/simStepSize; 

% initialize arrays
tEnd = nan(simSteps-1,1);
u_sol_vec =  nan(nu,simSteps-1);

% initial condition
x_sim(:,1) = x0_low;

disp('Start Simulation ...')

for k = 2:simSteps

    
    tic
    [sol]   = solver('x0',  z0,...       % initial guess
                   'p', x_sim(:,k-1),... % x0 i.e.current state
                   'lbx', lbz,...
                   'ubx', ubz,...
                   'lbg', lbg,...
                   'ubg', ubg);
    tEnd(k-1) = toc   ;     
    
    % get solution
    z_opt = full(sol.x);
    
    % reshape 
    x_sol = reshape(z_opt(1:Nx*N),Nx,N);
    
    u_sol = z_opt((nx*N)+1:end);
    u_sol = reshape(u_sol,3,N);

    u_sol_vec(:,k-1) = u_sol(:,1);

    % apply first entry of optimal solution to plant 
    x_sim(:,k) = full(xkp1(x_sim(:,k-1),u_sol(:,1)));
    
    % set current solution as initial guess for next iteration
    z0 = z_opt;

end 

disp('Simulation done Full-horizon...')


%% plotting
t = linspace(0,simTime,(simTime/simStepSize));

% states in degrees per second
x_low_deg =  x_low*180/pi;
x_up_deg  =  x_up*180/pi;

% states
figure('Name','States')
subplot(311)
plot(t,x_sim(1,:)*180/pi)
xlabel('t [s]')
ylabel('\omega_x [^\circ/s]')
hold on
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = x_low_deg(1); % y value where the dashed line is
miny =  x_low_deg(1)+0.5* x_low_deg(1);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  x_up_deg(1); % y value where the dashed line is
maxy = x_up_deg(1)+0.5*x_up_deg(1);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])

subplot(312)
plot(t,x_sim(2,:)*180/pi)
xlabel('t [s]')
ylabel('\omega_y [^\circ/s]')
hold on
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = x_low_deg(2); % y value where the dashed line is
miny =  x_low_deg(2)+0.5* x_low_deg(2);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  x_up_deg(2); % y value where the dashed line is
maxy = x_up_deg(2)+0.5*x_up_deg(2);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])

subplot(313)
plot(t,x_sim(3,:)*180/pi)
xlabel('t [s]')
ylabel('\omega_z [^\circ/s]')
hold on
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed = x_low_deg(3); % y value where the dashed line is
miny =  x_low_deg(3)+0.5* x_low_deg(3);
% Create an area plot with light gray background below the dashed line
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
% Define x limits and y limits for the shaded region
xLimits = [0, simTime]; % adjust according to your data
yDashed =  x_up_deg(3); % y value where the dashed line is
maxy = x_up_deg(3)+0.5*x_up_deg(3);
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
axis([0 simTime miny maxy])


% plot control ( zero-order hold)
t = linspace(0,simTime,(simTime/simStepSize)-1);

% set up with torques in miliNetwonmeter
u_low = [-1 -1 -1]'*1000;
u_up  = [ 1  1 1]'*1000;

% controls
figure('Name','Control')
subplot(311)
stairs(t,u_sol_vec(1,:)*1000)
xlabel('t [s]')
ylabel('\tau_x [mNm]')
hold on
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

subplot(312)
stairs(t,u_sol_vec(2,:)*1000)
xlabel('t [s]')
ylabel('\tau_y [mNm]')
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

subplot(313)
stairs(t,u_sol_vec(3,:)*1000)
xlabel('t [s]')
ylabel('\tau_z [mNm]')
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

% computation times
figure('Name','Solve Time')
semilogy(t,tEnd,'b')
xlabel('t [s]')
axis([0 simTime min(tEnd)-min(tEnd)*0.2 max(tEnd)+max(tEnd)*1000])

fprintf('Maximum solve time: %d ms\n',max(tEnd)*1000)
fprintf('Mean solve time: %d ms\n',mean(tEnd)*1000)
fprintf('Minimum solve time: %d ms\n',min(tEnd)*1000)


