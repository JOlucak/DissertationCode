%% ------------------------------------------------------------------------
%
%   NMPC for Van der Pol oscillator with shifted coordinates
%   (Equilibrium NOT at origin, e.g., x_e = [0.2;0], u_e = 0.2).
%
%   The MPC controller works in **shifted coordinates**:
%       x̃ = x - x_e
%       ũ = u - u_e
%
%   The plant simulation is done in real (unshifted) coordinates.
%   At each time step:
%     1) Measure x_k (real state)
%     2) Shift: x̃_k = x_k - x_e
%     3) Solve NMPC for x̃_k
%     4) Get optimal control increment ũ_0
%     5) Convert back: u_k = u_e + ũ_0 (applied to plant)
%     6) Simulate plant in real coordinates
%
% ------------------------------------------------------------------------

close all
clear
clear textprogressbar
clc

addpath('..\helperFunc\')  % Helper functions if needed

import casadi.*

% Choose solver for NMPC (IPOPT, fatrop, SQP, etc.)
solvermethod = 'IPOPT';

%% Simulation parameters
simTime = 100;       % total simulation time
simStepSize = 0.1;   % integration step
numRuns  = 1;        % number of Monte-Carlo runs

% Initial condition (real coordinates)
x0_real = [-0.63; 0.2];

% Prediction horizon and discretization
h_vec = 0.1;
T_vec = 10;          
N = T_vec/h_vec;     % number of shooting intervals

%% Define cost weights
Q = diag([1,1]);     % state cost
R = 10;               % input cost

%% Define equilibrium point (not at origin)
x_e = [1;0];       % equilibrium state
u_e = x_e(1);        % equilibrium input (steady-state control)

%% Define system dynamics (real coordinates)
x = SX.sym('x',2,1);
u = SX.sym('u',1,1);

xdot_real = [ x(2);
              (1 - x(1)^2)*x(2) - x(1) + u ];

%% Shift system dynamics to error coordinates
% In shifted coordinates: x̃ = x - x_e, ũ = u - u_e
x_shifted = SX.sym('x_shifted',2,1);
u_shifted = SX.sym('u_shifted',1,1);

xdot_shifted = substitute(xdot_real, [x;u], [x_shifted + x_e; u_shifted + u_e]);

%% Linearization around equilibrium (for terminal cost & set)
A_shift = full(casadi.DM(substitute(jacobian(xdot_shifted,x_shifted),...
             [x_shifted;u_shifted],[zeros(2,1);0])));
B_shift = full(casadi.DM(substitute(jacobian(xdot_shifted,u_shifted),...
             [x_shifted;u_shifted],[zeros(2,1);0])));

% Terminal cost from LQR design (stabilizing locally)
[K_shift,P_shift] = lqr(A_shift,B_shift,Q,R);

% Terminal penalty function (quadratic Lyapunov cost in shifted coords)
Phi = Function('Phi',{x_shifted,u_shifted},...
               {x_shifted'*P_shift*x_shifted});

%% Constraints (shifted coordinates)
% Original path constraint (example, from paper)
h0_real = 1 - x(1)^2 - 3*x(2)^2;

% Shift it since x = x̃ + x_e
h0_shifted = substitute(h0_real,x,x_shifted+x_e);

h0_lower = 0;
h0_upper = inf;

% State and input bounds (real coordinates)
x_lower_real = [-inf; -inf];
x_upper_real = [ inf;  inf];

u_max = 2;
u_lower_real = -u_max;
u_upper_real =  u_max;

% Shift bounds: 
%
% xmin <= x <= xmax substitute x = x̃ + x_e
% xmin <= x̃ + x_e <= xmax <--> xmin-x_e <= x̃  <= xmax- x_e
%
x_lower_shifted = x_lower_real - x_e;
x_upper_shifted = x_upper_real - x_e;
u_lower_shifted = u_lower_real - u_e;
u_upper_shifted = u_upper_real - u_e;

%% Integration (Runge-Kutta 4)
% MPC propagates based on shifted dynamical system
f_shifted = Function('f',{x_shifted,u_shifted},{xdot_shifted});
rk = 4; % substeps per interval
dt = T_vec/N/rk;

% x0 is later shifted in simulation, but our initial condition
x0 = SX.sym('x0',2,1);
uk = SX.sym('uk',1,1);

k1 = SX(2,rk);
k2 = SX(2,rk);
k3 = SX(2,rk);
k4 = SX(2,rk);
xk = [x0 SX(2,rk)];

for jj=1:rk
    k1(:,jj) = f_shifted(xk(:,jj),uk);
    k2(:,jj) = f_shifted(xk(:,jj)+dt*k1(:,jj)/2,uk);
    k3(:,jj) = f_shifted(xk(:,jj)+dt*k2(:,jj)/2,uk);
    k4(:,jj) = f_shifted(xk(:,jj)+dt*k3(:,jj),uk);
    xk(:,jj+1) = xk(:,jj) + dt*(k1(:,jj)+2*k2(:,jj)+2*k3(:,jj)+k4(:,jj))/6;
end
x_next = Function('x_next',{x0,uk},{xk(:,end)},{'x0','uk'},{'xk'});

%% Decision variables (multiple shooting)
% these are the shifted coordinates
X = SX.sym('X', [2,N]);
U = SX.sym('U', [1,N]);

z = [X(:);U(:)];

% Bounds
control_lb = repmat(u_lower_shifted,1,N);
control_ub = repmat(u_upper_shifted,1,N);
state_lb   = repmat(x_lower_shifted,1,N);
state_ub   = repmat(x_upper_shifted,1,N);

lbz = [state_lb(:);control_lb(:)];
ubz = [state_ub(:);control_ub(:)];

%% Dynamics constraints
g_dyn = X - x_next([x0 X(:,1:N-1)], U);
g_dyn = reshape(g_dyn,1,size(g_dyn,1)*size(g_dyn,2));
lbg_dyn = zeros(1,length(g_dyn));
ubg_dyn = lbg_dyn;

%% Path and terminal constraints
H_path = Function('H',{x_shifted},{h0_shifted});
h_vals = H_path(X); % for all nodes

% Terminal Lyapunov constraint (x_N inside level set gamma)
gamma = 1;
terminal_constraint = gamma - X(:,end)'*P_shift*X(:,end);

% Combine constraints
g_all = [g_dyn, h_vals, terminal_constraint];
lbg = [lbg_dyn, repmat(h0_lower,1,N), 0];
ubg = [ubg_dyn, repmat(h0_upper,1,N), inf];

%% Cost function (stage + terminal)
stage_cost = Function('L',{x_shifted,u_shifted},...
                      {x_shifted'*Q*x_shifted + u_shifted'*R*u_shifted});
J = Phi(X(:,end),0) + sum(stage_cost([x0 X(:,1:end-1)],U(:,1:end)));

%% Solver setup
prob = struct('f',J,'x',z,'g',g_all,'p',x0);

  options = struct('ipopt',struct('print_level',0), ...
                     'print_time',false,...
                     'record_time',true);

    solver = casadi.nlpsol('solver', 'ipopt', prob,options);



% fprintf('Pre-compile mex-files for simulation.\n')
% f = Function('f',{x,u},{xkp1(x,u)});
% C = CodeGenerator('sim_mex.c');
% C.add(f);
% opts = struct('mex', true);
% 
% f.generate('sim_mex.c',opts);
% 
% mex("sim_mex.c")

%% Simulation loop
x_sim_real(:,1) = x0_real;  % store real state trajectory
u_sim_real = nan(1,simTime/simStepSize-1);

z0 = zeros(N*(2+1),1); % initial guess
textprogressbar('Simulation for full-horizon MPC:');

for k = 2:simTime/simStepSize
    % Shift current real state for NMPC
    x_shift_k = x_sim_real(:,k-1) - x_e;

    % Solve NMPC in shifted coordinates
    sol = solver('x0',z0,'p',x_shift_k,...
                 'lbx',lbz,'ubx',ubz,'lbg',lbg,'ubg',ubg);
    z_opt = full(sol.x);

    % Extract solution
    X_opt = reshape(z_opt(1:2*N),2,N);
    U_opt = reshape(z_opt(2*N+1:end),1,N);

    % Apply first control (convert back to real coordinates)
    u_k = u_e + U_opt(1);
    u_sim_real(k-1) = u_k;

    % Simulate plant in real coordinates (using real dynamics!)
    x_sim_real(:,k) = sim_mex(x_sim_real(:,k-1), u_k);

    % Warm start (based on shifted solution)
    z0 = z_opt;

    % Stop early if near equilibrium
    if norm(x_sim_real(:,k)-x_e,inf) < 1e-3
        break
    end

    % update progress bar
    textprogressbar(k/(simTime/simStepSize)*100);
end

%% Plot results
t = linspace(0,simTime,length(x_sim_real));
for j = 1:numRuns
  xsol =  x_sim_real;
  usol = u_sim_real;

  xsol = xsol(:,1:k);
  usol = usol(:,1:k);


  figure(j)
  subplot(311)
  plot(t(1:k),xsol(1,:))
  hold
  plot([0 t(k)],[x_e(1) x_e(1)],'k--')
  xlabel('Time in seconds')
  ylabel('x_1')

  subplot(312)
  plot(t(1:k),xsol(2,:))
  hold on
  plot([0 t(k)],[x_e(2) x_e(2)],'k--')
  xlabel('Time in seconds')
  ylabel('x_2')

  subplot(313)
  plot(t(1:k),usol(1,:))
  hold on
  plot([0 t(k)],[u_e(1) u_e(1)],'k--')
  ylabel('u')
  xlabel('Time in seconds')

end

