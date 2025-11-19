
clear
close all
clc

import casadi.*

t  = SX.sym('t');
x  = SX.sym('x',3,1);
u  = SX.sym('u',3,1);
p  = SX.sym('p',5,1);

w  = SX.sym('w',3,1);

dt0 = 0.1;
simTime     = 2000;
simStepSize = dt0;

alpha = 1;

Q_weight = eye(3);
R_weight = eye(3)*0.001;


V =   -0.9 + 1.07768e+11*x(1)^4 - 0.00793778*x(1)^3*x(2) - 36261.4*x(1)^2*x(2)^2 ... 
  - 0.0442671*x(1)*x(2)^3 + 1.07768e+11*x(2)^4 + 0.0834083*x(1)^3*x(3) ... 
  + 0.0116344*x(1)^2*x(2)*x(3) + 0.0324123*x(1)*x(2)^2*x(3) + 0.210193*x(2)^3 ...
  *x(3) - 37064.7*x(1)^2*x(3)^2 - 0.213545*x(1)*x(2)*x(3)^2 + 1808.43*x(2)^2 ...
  *x(3)^2 + 0.0790018*x(1)*x(3)^3 + 0.141489*x(2)*x(3)^3 + 1.07768e+11*x(3)^4;

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


% initial conditions
omegaMax1 = 4/60*pi/180;
omegaMax2 = 4/60*pi/180;
omegaMax3 = 4/60*pi/180;


x0_low = [omegaMax1  omegaMax2  omegaMax3]';

Vfun(x0_low,0)

u_low = [-1 -1 -1]'*0.8;
u_up  = [ 1  1 1]'*0.8;

Nx = length(x);
Nu = length(u);

Phi = Function('f',...        
            {x},...     
            {     212751*x(1)^2 + 0.000245755*x(1)*x(2) + 161160*x(2)^2 - 8.662e-05*x(1)*x(3) ...
  - 0.000176199*x(2)*x(3) + 156568*x(3)^2});   


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

if ~exist('sim_mex.mexw64','file')
   mex("sim_mex.c")
end

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

textprogressbar('Simulation for horizon-one MPC:');
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


        textprogressbar(k/simSteps*100);

    end
 
    simTimeMeas = toc(startSim);
    fprintf(1,'\n')
    fprintf('Simulation time: %f seconds\n',simTimeMeas)

%%
t = linspace(0,simTime,(simTime/simStepSize));

% simple bounds
omegaMax1 = 6/60;
omegaMax2 = 6/60;
omegaMax3 = 6/60;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';

% states
figure('Name','States')
subplot(311)
plot(t,x_sim(1,:)*180/pi)
xlabel('t [s]')
ylabel('\omega_x [^\circ/s]')
hold on
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


subplot(312)
plot(t,x_sim(2,:)*180/pi)
xlabel('t [s]')
ylabel('\omega_y [^\circ/s]')

hold on
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


subplot(313)
plot(t,x_sim(3,:)*180/pi)
xlabel('t [s]')
ylabel('\omega_z [^\circ/s]')
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


t = linspace(0,simTime,(simTime/simStepSize)-1);

% set up with torques in miliNetwonmeter
u_low = [-1 -1 -1]'*1000;
u_up  = [ 1  1 1]'*1000;

% controls
figure('Name','Control')
subplot(311)
plot(t,u_sol_vec(1,:)*1000)
xlabel('t [s]')
ylabel('\tau_y [mNm]')
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
plot(t,u_sol_vec(2,:)*1000)
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
plot(t,u_sol_vec(3,:)*1000)
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



figure('Name','Sufficient Conditions')
plot(t,suffCon,'b')
hold on
plot([0 t(end)],[0 0],'k--')
xlabel('t [s]')
axis([0 t(end), min(suffCon)+0.1*min(suffCon) max(suffCon)+0.1*max(suffCon)])
ylabel('\alpha \Delta V(\cdot) - W(\cdot)')
legend('Sufficient Condition','Zero line')


% time
figure('Name','Solve Time')
semilogy(t,tEnd,'b')
xlabel('simulation time [s]')
ylabel('computation time [s]')
axis([0 simTime 1e-5 1])

fprintf('Maximum solve time: %d ms\n',max(tEnd)*1000)
fprintf('Mean solve time: %d ms\n',mean(tEnd)*1000)
fprintf('Minimum solve time: %d ms\n',min(tEnd)*1000)

fprintf('Minimum value of sufficient condition: %d \n',min(suffCon))

