function [tEnd,x_sim,u_sol_vec] = solveOCP_RK45_MPC_RTI_lowDisc(problem)

import casadi.*

%% depack problem struct

%simulation
simTime = problem.simTime;
simStepSize = problem.simStepSize;

% MPC
gamma  = problem.gamma;
RTI_maxSteps = problem.RTI_maxSteps;

x = problem.x;
u = problem.u;
p  = SX.sym('p',length(x),1);

Q_weight = problem.Q;
R_weight = problem.R;

h0      = problem.h0;
h0_low  = problem.h0_low;
h0_up   = problem.h0_up;
K = problem.K;
N = problem.N/2;
T = problem.T;

Phi            = problem.Phi;
P              = problem.P;

xdot = problem.xdot;

x0_low = problem.x0_low;

x_low = problem.x_low;
x_up = problem.x_up;

u_low = problem.u_low;
u_up  = problem.u_up;

Nx = problem.Nx;
Nu = problem.Nu;


H = Function('f',...      
             {x,p},...     
             {h0});   


%% fixed-step Runge-Kutta 4 integration

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


z = [X(:) ;U(:)];

control_lb_grid = repmat(u_low,1,N);
control_ub_grid = repmat(u_up,1,N);

state_lb_grid = repmat(x_low,1,N);
state_ub_grid = repmat(x_up,1,N);


lbz = [state_lb_grid(:); control_lb_grid(:)];
ubz = [state_ub_grid(:); control_ub_grid(:)];



%% path constraints
% dynamics
g = X - xkp1([x0 X(:,1:N-1)], U); %  multiple shooting
g = reshape(g,1,size(g,1)*size(g,2));

% equality constraint for dynamics
lbg_dyn = zeros(1,size(g,2));
ubg_dyn = lbg_dyn;

% add path constraints
gh = [gamma-X(:,end)'*P*X(:,end)];%,...    
%       H(X(:,1:end),p)] ; % terminal set
gH = reshape(gh,1,size(gh,1)*size(gh,2));
g = [g,gH];

lbg_cust = repmat(h0_low ,1,N);
ubg_cust = repmat(h0_up ,1,N);


% combine path constraints (defect constraints and user defined
% in-/equality constraints
lbg = [lbg_dyn,0];%,lbg_cust ] ;
ubg = [ubg_dyn,inf];%,ubg_cust ];


%% cost function
Q = Function('f', {x, u}, { x'*Q_weight*x + u'*R_weight*u});
r = 100;

J = r*Phi(X(:,end),zeros(size(u))) + sum(Q([x0 X(:,1:end-1)],U(:,1:end)));


%% initial guess

z0 = zeros(N*(Nx+Nu),1);

%% setup solver
prob   = struct('f', J, 'x', z, 'g', g,'p',x0);

disp('Building solver ...')

options = struct('ipopt',struct('print_level',1),'print_time',false);
options.ipopt.max_iter = RTI_maxSteps;
solver = casadi.nlpsol('solver', 'ipopt', prob,options);

disp('Solver building succesful!')


%% Simulation
nx = problem.Nx;
nu = problem.Nu;

x_sim(:,1) = x0_low;

lamx0 = zeros(size(z0));
lamg0 = zeros(size(lbg));


simSteps = simTime/simStepSize; 

disp('Start Simultion ...')
for k = 2:simSteps


tic;
[sol]   = solver('x0',  z0,...
               'p', x_sim(:,k-1),...
               'lbx', lbz,...
               'ubx', ubz,...
               'lbg', lbg,...
               'ubg', ubg,...
                'lam_x0', lamx0	,...
                'lam_g0', lamg0 );
tEnd(k-1) = toc   ;     


z_opt = full(sol.x);

% x_sol = reshape(z_opt(1:Nx*N),Nx,N);

u_sol = z_opt((nx*N)+1:end);
u_sol = reshape(u_sol,3,N);


%         x_sol_vec(:,k)   = x_sol;
        u_sol_vec(:,k-1) = u_sol(:,1);



        % apply first entry of optimal solution
x_sim(:,k) = full(xkp1(x_sim(:,k-1),u_sol(:,1)));

% stage = full((Q(x_sol(:,end),u_sol(:,end-1))));
% uN = -K*x_sol(:,end);
% xNp1 = xkp1(x_sol(:,end),uN);
% cost = r*Phi(xNp1,zeros(size(u))) + sum(Q([x_sol(:,1:end)],[u_sol(:,2:end) uN])); %x_sol(:,end-1)'*P*x_sol(:,end-1) - full(xkp1( x_sol(:,end-1),u_sol(:,end-1)))'*P*full(xkp1(x_sol(:,end),u_sol(:,end-1))) - stage
%%
% figure(1)
% title(['Sim time: ' num2str(k*rk*dt) ])
% plot(x_sol(5,end),x_sol(2,end),'*','MarkerIndices',1)
% pvar x1 x2
% hold on
% pcontour([x1;x2]'*P([5 2],[5 2])*[x1;x2],1)
% drawnow
% measure state and set for new optimization
% lbg(1:Nx) = x_sim(:,k);
% ubg(1:Nx) = x_sim(:,k);

% initial guess for next iteration
z0    = z_opt;
lamx0 = sol.lam_x; % dual
lamg0 = sol.lam_g; % dual


end 
toc


disp('Simulation done RTI...')
end