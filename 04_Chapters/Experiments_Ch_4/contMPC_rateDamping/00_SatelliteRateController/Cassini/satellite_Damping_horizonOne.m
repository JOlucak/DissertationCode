
clear
close all
clc

import casadi.*

t  = SX.sym('t');
x  = SX.sym('x',3,1);
u  = SX.sym('u',3,1);
p  = SX.sym('p',5,1);

dt = 0.1;
simTime     = 150;
simStepSize = dt;

alpha = 1;

Q_weight = eye(3)*10;
R_weight = eye(3)*0.001;



V =     -0.9 + 1.07768e+11*x(1)^4 + 0.124245*x(1)^3*x(2) + 38088.7*x(1)^2*x(2)^2 ...
  + 0.0479643*x(1)*x(2)^3 + 1.07768e+11*x(2)^4 + 0.0278467*x(1)^3*x(3)  ...
  + 0.0687741*x(1)^2*x(2)*x(3) - 0.0908202*x(1)*x(2)^2*x(3) - 0.0744837*x(2)^3 ...
  *x(3) + 19253.1*x(1)^2*x(3)^2 + 0.0124688*x(1)*x(2)*x(3)^2 + 19253.2*x(2)^2 ...
  *x(3)^2 + 0.00579223*x(1)*x(3)^3 - 0.0181549*x(2)*x(3)^3 + 6.73551e+09*x(3)^4;

Vfun = Function('f',...       
            {x, t},...     
            {V});   % dt = T/N, k = 1 ( first discrete step) 

V0_low = -inf;
V0_up  = 0;

J = diag([8970;9230;3830]);

% skew-symmetric matrix
skew = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

xdot = -J\skew(x(1:3))*J*x(1:3) + J\u;

% simple bounds; from slew rate [deg/min] to [rad/s]
omegaMax1 = -0.08*pi/180;
omegaMax2 = 0.04*pi/180;
omegaMax3 = 0*pi/180;


x0_low = [ omegaMax1  omegaMax2  omegaMax3]';

Vfun(x0_low,0)

u_low = [-1 -1 -1]'*0.2;
u_up  = [ 1  1 1]'*0.2;

Nx = length(x);
Nu = length(u);

Phi = Function('f',...       
            {x},...     
            {227159*x(1)^2 + 2.55617e-05*x(1)*x(2) + 233571*x(2)^2 + 6.66998e-06*x(1)*x(3) ...
  - 0.000135343*x(2)*x(3) + 53174.3*x(3)^2});   % dt = T/N, k = 1 ( first discrete step) 


%% fixed-step Runge-Kutta 4 integration
f = Function('f', {x, u}, {xdot}); 

rk = 4;
dt = 0.1/rk;
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


% options = struct('ipopt',struct('print_level',0),'print_time',false);


options = struct('print_status',0,...
                 'print_header',0,...
                 'print_time',0,...
                 'verbose_init',0,...
                 'print_iteration',0,...
                 'qpsol','qpoases');

options.qpsol_options.printLevel = 'none';


disp('Setup solver ...')
% solver = casadi.nlpsol('solver', 'ipopt', prob);
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

    % profile -memory on;
    % setpref('profiler','showJitLines',1);
   disp('Start Simulation for horizon-one MPC')
    for k = 2:simSteps

       % solve one-step MPC
        [sol]   = solver('x0',  z0,...
                       'p',   [dt, alpha,tmpx0'],...  %[dt,alpha,x]
                       'lbx', lbz,...
                       'ubx', ubz,...
                       'lbg', lbg,...
                       'ubg', ubg);

          % runtime measurement using timeit; see matlab documentation
          g = @() solver('x0',  z0,...
                       'p',   [dt, alpha,tmpx0'],...  %[dt,alpha,x]
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

        suffCon(k-1) = alpha*(full(Phi(tmpx0))-full(Phi(xkp1(x_sim(:,k-1),u_sol)))) - (tmpx0'*Q_weight*tmpx0 + u_sol' * R_weight * u_sol);
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

%%
t = linspace(0,simTime,(simTime/simStepSize));

% simple bounds
omegaMax1 = 1;
omegaMax2 = 1;
omegaMax3 = 2;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';

% states
figure('Name','States')
subplot(311)
plot(t,x_sol_vec(1,:)*180/pi)
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
plot(t,x_sol_vec(2,:)*180/pi)
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
plot(t,x_sol_vec(3,:)*180/pi)
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
u_low = [-1 -1 -1]'*200;
u_up  = [ 1  1 1]'*200;

% controls
figure('Name','Control')
subplot(311)
plot(t,u_sol_vec(1,:)*1000)
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
axis([0 t(end), -0.5 0.5])
ylabel('\alpha \Delta V(\cdot) - W(\cdot)')
legend('Sufficient Condition','Zero line')


% time
figure('Name','Solve Time')
semilogy(t,tEnd,'b')
xlabel('t [s]')
axis([0 simTime 1e-4 1])

fprintf('Maximum solve time: %d ms\n',max(tEnd)*1000)
fprintf('Mean solve time: %d ms\n',mean(tEnd)*1000)
fprintf('Minimum solve time: %d ms\n',min(tEnd)*1000)


