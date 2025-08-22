clear
close all
clc

import casos.*

% States (shifted indeterminates)
x = PS('x',2,1);
u = PS('u',1,1);

% Control limits (not directly enforced here)
umin = -1; umax = 1;

%% System Dynamics
% Equilibrium point (trim)
xeq = [0.2;0];
ueq = xeq(1);  % steady-state input

% Original Van der Pol dynamics (symbolic)
mu = 1;
f = [ x(2);
      mu*(1 - x(1)^2)*x(2) - x(1) + u ];

% Shift the dynamics to work in x̃ = x - xeq
f_shifted = subs(f,[x;u],[x+xeq;u+ueq]);

% Linearize shifted dynamics at origin (x=0,u=0)
A0_shifted = full(PD(subs(nabla(f_shifted,x),[x;u],[zeros(2,1);0])));
B0_shifted = full(PD(subs(nabla(f_shifted,u),[x;u],[zeros(2,1);0])));

% LQR design for initial guess
Q = diag([1,1]); 
R = 2.5;
[K0,P0] = lqr(A0_shifted,B0_shifted,Q,R);

% Initial Lyapunov guess: quadratic form
Wval = x'*P0*x;

%% State constraints (shifted)
% Original safe set: x1^2 + 3*x2^2 <= 1 (ellipse)
g = x(1)^2 + 3*x(2)^2 - 1;

% This is already in shifted coords (x represents x̃)

%% Constant feedback law for SOS test (shifted)
% Control is: u = ueq + (-K0*x)
K = -K0*x;

%% Setup SOS Problem for CLF (with fixed feedback)
V = PS.sym('v',monomials(x,2));      % Lyapunov candidate (quadratic)
s1 = PS.sym('s1',monomials(x,2));    % SOS multiplier (degree-2)

b = 1;   % Level set (gamma)

opts = struct('sossol','mosek','verbose',1,'max_iter',100);

% Objective: minimize ||g - V||^2 (fit V to constraint g)
cost = dot(g - V, g - V);

% NOTE: f_cl must use full control law u = ueq + (-K0*x)
f_cl = subs(f,[x;u],[x+xeq;ueq + (-K0*x)]);

sos = struct('x',[V;s1], 'f',cost,'p',[]);
sos.g = [
    s1;
            s2
        s3;
    V - 1e-6*(x'*x);                           % V > 0
    s1*(V - b) - nabla(V,x)*f_cl               % decay inside V<=b
            s2*(V-b) + K - umin_shift;
        s3*(V-b) + umax_shift - K;
];

opts.Kx = struct('lin',length(sos.x));
opts.Kc = struct('sos',length(sos.g));

S = nlsossol('S','sequential',sos,opts);

% Initial guess for sequential solver
x0 = PD([ g; x'*x ]);

% Solve SOS
sol = S('x0',x0);

Vsol_re = sol.x(1) - full(PD(b)); % Final Lyapunov function

%% Plot ROA level set vs state constraint
import casos.toolboxes.sosopt.*
figure(1); clf
pcontour(Vsol_re,0,[-1 1 -1 1],'g'); hold on
pcontour(g,0,[-1 1 -1 1],'k--');
legend('ROA estimate','Safe set');

%% Simulate closed-loop system (real dynamics)
dt = 0.01; T = 50;
x0_real = [-0.4;0.1]; % initial condition (real)

Nsteps = round(T/dt);
x_sim = nan(2,Nsteps);
x_sim_shift = nan(2,Nsteps);
u_sim = nan(1,Nsteps-1);
V_sim = nan(1,Nsteps);
t = linspace(0,T,Nsteps);

dyn = to_function(f);          % real (unshifted) dynamics
Vfun = to_function(sol.x(1));  % Lyapunov function (shifted coords)

x_sim(:,1) = x0_real;
x_sim_shift(:,1) = x0_real- xeq;
for k = 2:Nsteps
    x_shift = x_sim(:,k-1) - xeq;
    
    V_sim(k-1) = full(Vfun(x_shift(1),x_shift(2)));

    % Control: u = ueq - K0*x̃
    ureal = ueq - K0*x_shift;
    u_sim(k-1) = ureal;

    % Integrate using Euler (could replace with ode45/RK4 for accuracy)
    dx = full(dyn(ureal,x_sim(1,k-1),x_sim(2,k-1)));
    x_sim(:,k) = x_sim(:,k-1) + dt*dx;
    x_sim_shift(:,k) = x_sim(:,k)-xeq;
end

%% Plot results
figure(1);
subplot(4,1,1)
plot(t,x_sim(1,:),'LineWidth',1.5); hold on
yline(xeq(1),'--r'); ylabel('x_1 (real)');
title('Van der Pol with SOS Lyapunov + LQR Control');
subplot(4,1,2)
plot(t,x_sim(2,:),'LineWidth',1.5); hold on
yline(xeq(2),'--r'); ylabel('x_2 (real)');
subplot(4,1,3)
plot(t(1:end-1),u_sim,'LineWidth',1.5); hold on
yline(ueq,'--r'); ylabel('u (real)');
subplot(4,1,4)
plot(t(1:end),V_sim,'LineWidth',1.5);
ylabel('V(x̃)');
xlabel('Time [s]');

% plot trajectory in shifted coordinates
import casos.toolboxes.sosopt.*
figure(2)
pcontour(Vsol_re,0,[-1 1 -1 1],'g'); hold on
plot(x_sim_shift(1,:),x_sim_shift(2,:),'k')
plot(x_sim_shift(1,1),x_sim_shift(2,1),'b*')
plot(x_sim_shift(1,end),x_sim_shift(2,end),'r*')