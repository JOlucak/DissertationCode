function [tEnd,x_sim,u_sol_vec,suffCon] = solve1StepOpt_MPC(problem,alpha)  

import casadi.*

%% depack problem struct

%simulation
simTime = problem.simTime;
simStepSize = problem.simStepSize;

% MPC
x = problem.x;
u = problem.u;
p  = SX.sym('p',2+length(x),1); %[dt,alpha,x]

Q_weight = problem.Q;
R_weight = problem.R;

V      = problem.OneStep.V;
V0_low = problem.OneStep.V0_low;
V0_up  = problem.OneStep.V0_up;

N = problem.N;
T = problem.T;

xdot = problem.xdot;

x0_low = problem.x0_low;

u_low = problem.u_low;
u_up  = problem.u_up;

Nx = problem.Nx;
Nu = problem.Nu;

Phi = Function('f',...       
            {x, p},...     
            {V(p,x)});   % dt = T/N, k = 1 ( first discrete step) 


%% fixed-step Runge-Kutta 4 integration
f = Function('f', {x, u}, {xdot}); 

rk = 4;
dt = T/N/rk;
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
J =  p(3:end)' * Q_weight * p(3:end) + uk' * R_weight * uk + Phi(X,p) * p(2);

% equality constraint
lbg_dyn = zeros(1,size(g,1)*(size(g,2)));
ubg_dyn = lbg_dyn;


% add path constraints
V = V(T/N,X);   % dt = T/N, k = 1 ( first discrete step) 
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


% options = struct('ipopt',struct('print_level',0),'print_time',false);


options = struct('print_status',0,...
                     'print_header',0,...
                     'print_time',0,...
                     'record_time',true,...
                     'verbose_init',0,...
                     'print_out',false,...
                     'print_iteration',false);  % Maximum number of SQP iterations (RTI)


                   % 'hessian_approximation','exact',...
                     % 'convexify_strategy','eigen-reflect',...

    options.qpsol_options.print_info = false;
    options.qpsol_options.print_out = false;
    options.qpsol_options.print_in = false;
    options.qpsol_options.print_header = false;
    options.qpsol_options.print_iter = false;

 
    options.qpsol_options.error_on_fail = false;
    options.qpsol = 'qrqp';


disp('Setup solver ...')
% solver = casadi.nlpsol('solver', 'ipopt', prob,options);
solver = casadi.nlpsol('solver', 'sqpmethod', prob,options);
disp('Solver setup succesful!')

%% Simulation
    tmpx0           = x0_low;
    x_sol_vec(:,1)  = x0_low;

   
    % initial guess
    z0 = zeros(Nx+Nu,1);

    simSteps = simTime/simStepSize; 


   costFun = [];

   x_sim(:,1) = x0_low;
suffCon = [];
    % profile -memory on;
    % setpref('profiler','showJitLines',1);
   disp('Start Simulation for horizon-one MPC')
    for k = 2:simSteps

       % solve one-step MPC
        [sol]   = solver('x0',  z0,...
                       'p',   [T/N, alpha,tmpx0'],...  %[dt,alpha,x]
                       'lbx', lbz,...
                       'ubx', ubz,...
                       'lbg', lbg,...
                       'ubg', ubg);

          % runtime measurement using timeit; see matlab documentation
          g = @() solver('x0',  z0,...
                       'p',   [T/N, alpha,tmpx0'],...  %[dt,alpha,x]
                       'lbx', lbz,...
                       'ubx', ubz,...
                       'lbg', lbg,...
                       'ubg', ubg);
  
        tEnd(k-1) = timeit(g);
      

%         solver.stats
        % depack solution
%         cost(k) = full(Phi(tmpx0,)) full(sol.f);
        z_opt = full(sol.x);
        x_sol = z_opt(1:Nx);
        u_sol = z_opt(Nx+1:end);

        % suffCon(k-1) = alpha*(full(Phi(tmpx0,zeros(8,1)))-full(Phi(xkp1(x_sim(:,k-1),u_sol),zeros(8,1)))) - (tmpx0'*Q_weight*tmpx0 + u_sol' * R_weight * u_sol);
        % estimate
%         costFun = [costFun full(F(x_sim(:,k-1),u_sol))' * Q_weight * full(F(x_sim(:,k-1),u_sol)) + u_sol' * R_weight * u_sol];
        
        x_sol_vec(:,k)   = x_sol;
        u_sol_vec(:,k-1) = u_sol;


        x_sim(:,k) = full(xkp1(x_sim(:,k-1),u_sol));

        tmpx0 = x_sim(:,k);
        
         % initial guess based on old solution         
         z0 = [x_sol;u_sol];

    end
% profile viewer
disp('Simulation horizon-one MPC done!')

end