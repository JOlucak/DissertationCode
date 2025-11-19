close all
clear
clear textprogressbar
clc


import casadi.*

solvermethod = 'SQP';

%% Simulation parameter
simTime = 1500;
simStepSize = 0.1;

%% define cost function
Q = eye(3)*100;
R = eye(3)*0.001;        

%% define problem
% variables
x  = SX.sym('x',3,1);
u  = SX.sym('u',3,1);

nx = length(x);
nu = length(u);

% define integration step size MPC
h = 1;
T = 400;          % prediction horizon
N = T/h;          % intervals


% dynamics
% skew-symmetric matrix
% cross-product matrix
cpm = @(x) [   0   -x(3)  x(2);
              x(3)   0   -x(1);
             -x(2)  x(1)    0 ];

% satellite parameter
I = diag([31046;77217;78754]);

% dynamics

xdot =  -I\cpm(x(1:3))*I*x(1:3) + I\u;

% path cost
Langrange_term = u'*R*u + x'*Q*x;

% compute terminal set and terminal penalty
A = full(casadi.DM(substitute(jacobian(xdot,x),[x;u],[zeros(3,1);zeros(3,1)])));
B = full(casadi.DM(substitute(jacobian(xdot,u),[x;u],[zeros(3,1);zeros(3,1)])));

[K,P] = lqr(A,B,Q,R);

% kappa = real(-max(eig(A-B*K)))
% substitute(xdot,u,-K*x)

% level set; found with SOS program
gamma = 1;

% terminal set was computed via constrained ROA;
Phi = Function('f',...      % Name
            {x, u,},...             % Input variables
            {x'*P*x});              

%% path constraints
h0     = [];
h0_low = [];
h0_up  = []; 


%% Initial conditions

% initial attitude
% phi0   = 120*pi/180;
% theta0 = 0*pi/180;
% psi0   = 0;
% 
% sigma_0 = Euler1232MRP([phi0,theta0,psi0]);
% 
% EulerAngle = 4*atan(sqrt(sigma_0'*sigma_0))*180/pi;

% disp(['Euler angle: ' num2str(EulerAngle) ' in deg'])

% initial rates
omega_0 = [0.0254;-0.0015; 0];

% initial conditions
x0_low = omega_0;
x0_up  = omega_0;

u0_low =  [-inf -inf -inf]';
u0_up  = -[-inf -inf -inf]';

% final conditions; actually not used because we have a terminal set
xf_low = ones(3,1)*(-inf);  
xf_up  = -xf_low; 

uf_low  = [-inf -inf -inf]';
uf_up  = -[-inf -inf -inf]';

% simple bounds
x_low    =  [-2*pi/180 -1*pi/180 -1*pi/180]';
x_up     = -x_low';

maxu = 1.2; % Nm
u_low = [-maxu -maxu -maxu]';
u_up  = [ maxu maxu  maxu]';


Nx = length(x);
Nu = length(u);


%% solve problems with different MPC schemes. Simulations might take some time since we use matlab for simulation

%full horizon problem RK45 using IPOPT
tic
disp('-----------------------------------')
disp('Full hor. problem')


import casadi.*
p  = SX.sym('p',length(x),1);

Q_weight = Q;
R_weight = R;

% path constraint function
H = Function('f',...       % Name
             {x,p},...     % Input variables
             {h0});   


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

% vector of decision variables
z = [X(:) ;U(:)];

% simple constraints
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

% add path constraints and terminal constraint
gh = gamma-X(:,end)'*P*X(:,end);  %  [ H(X(:,1:end),p), gamma-X(:,end)'*P*X(:,end)] ; 
gH = reshape(gh,1,size(gh,1)*size(gh,2));

% put all path constraints together
g = [g,gH];

lbg_cust = repmat(h0_low ,1,N);
ubg_cust = repmat(h0_up ,1,N);

% combine path constraints (defect constraints and user defined
% in-/equality constraints
lbg = [lbg_dyn,0 ] ; % [lbg_dyn,lbg_cust,0 ] ;
ubg = [ubg_dyn,inf ]; % [ubg_dyn,ubg_cust,inf ];

%% cost function
Q = Function('f', {x, u}, { x'*Q_weight*x + u'*R_weight*u});


J = Phi(X(:,end),zeros(size(u))) + sum(Q([x0 X(:,1:end-1)],U(:,1:end))); 

%% initial guess for first iteration
z0 = zeros(N*(Nx+Nu),1);

%% setup solver
prob   = struct('f', J, 'x', z, 'g', g','p',x0);


fprintf('Setup nlp. solver using %s.\n',solvermethod)

% if strcmp(solvermethod,'SQP')
% 
% 
options = struct('print_status',0,...
                 'print_header',0,...
                 'print_time',0,...
                 'record_time',true,...
                 'verbose_init',0,...
                 'print_iteration',0,...
                 'qpsol','qpoases');

options.qpsol_options.printLevel = 'none';
options.max_iter = 1;
solver = casadi.nlpsol('solver', 'sqpmethod', prob,options);
% else
% 
%     options = struct('ipopt',struct('print_level',0), ...
%                        'print_time',false,...
%                         'record_time',true);
%     solver = casadi.nlpsol('solver', 'ipopt', prob,options);
% end
fprintf('Solver is setup using %s.\n',solvermethod)


% solver = casadi.nlpsol('solver', 'alpaqa', prob);


%% Simulation
simSteps = simTime/simStepSize; 

x_sim(:,1) = x0_low;
x_sol_vec(1:3,1) = x0_low;

textprogressbar('Simulation for full-horizon MPC:');

lamx0 = zeros(size(z0));
lamg0 = zeros(size(lbg));

for k = 2:simSteps

        [sol]   = solver('x0',  z0,'p', x_sim(:,k-1),'lbx', lbz,'ubx', ubz,'lbg', lbg,'ubg', ubg, 'lam_x0', lamx0,'lam_g0', lamg0 );
        
        tEnd(k-1) = solver.stats.t_wall_total ;     
        
        % get solution
        z_opt = full(sol.x);
        
        x_sol = reshape(z_opt(1:Nx*N),Nx,N);
        
        u_sol = z_opt((nx*N)+1:end);
        u_sol = reshape(u_sol,3,N);
        
        u_sol_vec(:,k-1) = u_sol(:,1);
        
        % apply first entry of optimal solution
        x_sim(:,k) = sim_mex(x_sim(:,k-1), u_sol(:,1));

        x_sol_vec(1:3,k)  = x_sim(1:3,k);

        % check convergence: rates below threshold and torque below threshold
        if norm(x_sim(1:3,k),inf)*180/pi < 1e-3  && norm(u_sol(:,1),inf) < 0.001
            break
        end

z0 = z_opt;
        lamx0 = sol.lam_x; % dual
        lamg0 = sol.lam_g; % dual
        

% update progress bar
textprogressbar(k/(simTime/simStepSize)*100);
end 

textprogressbar('Progress bar  - termination')



%% plotting
numRuns  = 1;

t = linspace(0, simTime, simTime/simStepSize);
colors = lines(numRuns); % Get a colormap for different runs

% re-scale rate constraints (real physical constraints)
x_low =  [-0.5 -0.2 -0.2]';
x_up  =  -x_low;

% Plot Rates
figure('Name', 'Rates');
for i = 1:3
    subplot(3,1,i);
    hold on;
        plot(t(1:k), x_sim(i,1:k) * 180/pi, 'Color', colors);
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
u_low = [-1 -1 -1]'*1200;
u_up  = [ 1  1 1]'*1200;

figure('Name', 'Control Torques');
for i = 1:3
    subplot(3,1,i);
    hold on;
        plot(t_short(1:k-1), u_sol_vec(i,1:k-1) * 1000, 'Color', colors);
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

save('fullhorizon_completeWS.mat')

