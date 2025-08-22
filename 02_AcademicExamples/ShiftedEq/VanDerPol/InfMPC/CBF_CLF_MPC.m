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

%  in shifted coords (x represents x̃)
g_shift = subs(g,x,x+xeq);

% shift control bounds
umax_shift = umax-ueq;
umin_shift = umin-ueq;

%% Setup SOS Problem for CLF (with fixed feedback)

h  = PS.sym('h',monomials(x,2));     % CBF
V  = PS.sym('v',monomials(x,2));      % Lyapunov candidate (quadratic)
s1 = PS.sym('s1',monomials(x,2));    % SOS multiplier (degree-2)
s2 = PS.sym('s2',monomials(x,0));    % SOS multiplier (degree-2)
s3 = PS.sym('s3',monomials(x,0));    % SOS multiplier (degree-2)
s4 = PS.sym('s4',monomials(x,2));    % SOS multiplier (degree-2)
s5 = PS.sym('s5',monomials(x,2));    % SOS multiplier (degree-2)

K  = PS.sym('k',monomials(x,1));
b = 1;   % Level set (gamma)

opts = struct('sossol','mosek','verbose',1,'max_iter',100);

% Objective: minimize ||g - V||^2 (fit V to constraint g)
% NOTE: f_cl must use full control law u = ueq + (-K0*x)
f_cl = subs(f,[x;u],[x+xeq;ueq + K]);

tau = ( nabla(V,x)*f_cl    + K'*R*K + x'*Q*x); 

cost = dot(g_shift-h,g_shift-h);



sos = struct('x',[h;V;s1;K;s2;s3;s4;s5], 'f',cost,'p',[]);

% K-function parameter for CBF, i.e., gamma(W(x)) = gamma*W(x)
gamma = 0.0001; % heuristically found

sos.g = [
        s1;
        s2
        s3;
        s4;
        s5;
        V - 1e-6*(x'*x);                           % V > 0
        s1*(h - b) - nabla(h,x)*f_cl - gamma*(h-b)              % decay inside V<=b
        s2*(h - b) + K - umin_shift;
        s3*(h - b) + umax_shift - K;
        s4*(h - b) - g_shift;
        s5*(h - b) -  tau ;                                   % CLF dissipatio
];

opts.Kx = struct('lin',length(sos.x));
opts.Kc = struct('sos',length(sos.g));

S = nlsossol('S','sequential',sos,opts);

% Initial guess for sequential solver
x0 = PD([g; g; x'*x ;-K0*x;0;0;x'*x;x'*x]);

% Solve SOS
sol = S('x0',x0);

% shifted coordinates
hsol_re = sol.x(1) - full(b);
Vsol_re = sol.x(2); % Final Lyapunov function

% control law in shifted coordinates
Ksol    = sol.x(4);


%% Plot ROA level set vs state constraint
import casos.toolboxes.sosopt.*
figure(1); clf
% plot (domain) is in real coordinates, thus we have to transform into
% shifted coordinates i.e  x̃ = x - xeq
pcontour(subs(hsol_re,x,x-xeq),0,[-1 1 -1 1],'g'); hold on
pcontour(g,0,[-1 1 -1 1],'k--');
legend('ROA estimate','Safe set');

%% Simulate closed-loop system (real dynamics)
dt = 0.01; 
T = 100;
x0_real = [-0.2;0.1]; % initial condition (real)

Nsteps = round(T/dt);
x_sim = nan(2,Nsteps);
x_sim_shift = nan(2,Nsteps);
u_sim = nan(1,Nsteps-1);
V_sim = nan(1,Nsteps);
t = linspace(0,T,Nsteps);

dyn = to_function(f);          % real (unshifted) dynamics

hfun = to_function(hsol_re);  % Lyapunov function (shifted coords)
Vfun = to_function(sol.x(2));  % Lyapunov function (shifted coords)
W_fun= to_function(hsol_re);
V_fun= to_function(sol.x(2));
Kfun = to_function(Ksol);
x_sim(:,1) = x0_real;
x_sim_shift(:,1) = x0_real- xeq;

% check if we are in the sublevel set
if full(hfun(x_sim_shift(1,1),x_sim_shift(2,1))-full(PD(b))) > 0
    disp('Real state does not lie in shifted sublevel set!')
    return
end

for k = 2:Nsteps
    x_shift = x_sim(:,k-1) - xeq;
    
    V_sim(k-1) = full(Vfun(x_shift(1),x_shift(2)));

    % Control: u = ueq + K(x̃)
    ureal = ueq + full(Kfun(x_shift(1),x_shift(2)));
    u_sim(k-1) = ureal;

    % Integrate using Euler (could replace with ode45/RK4 for accuracy)
    dx = full(dyn(ureal,x_sim(1,k-1),x_sim(2,k-1)));
    x_sim(:,k) = x_sim(:,k-1) + dt*dx;
    x_sim_shift(:,k) = x_sim(:,k)-xeq;
end

% Plot results
figure(2);
subplot(4,1,1)
plot(t,x_sim(1,:),'LineWidth',1.5); hold on
yline(xeq(1),'--r'); ylabel('x_1 (real)');
title('Van der Pol with SOS Lyapunov + Polynomial Control Law');
subplot(4,1,2)
plot(t,x_sim(2,:),'LineWidth',1.5); hold on
yline(xeq(2),'--r'); ylabel('x_2 (real)');
subplot(4,1,3)
plot(t(1:end-1),u_sim,'LineWidth',1.5); hold on
yline(ueq,'--r'); ylabel('u (real)');
yline(umax,'--k');
yline(umin,'--k');
axis([0 t(end) umin+0.1*umin umax+0.1*umax])
subplot(4,1,4)
plot(t(1:end),V_sim,'LineWidth',1.5);
ylabel('V(x̃)');
xlabel('Time [s]');

% plot real trajectory 
figure(1)
hold on
plot(x_sim(1,:),x_sim(2,:),'k')
plot(x_sim(1,1),x_sim(2,1),'b*')
plot(x_sim(1,end),x_sim(2,end),'r*')

% plot trajectory in shifted coordinates
import casos.toolboxes.sosopt.*
figure(3)
pcontour(hsol_re,0,[-1 1 -1 1],'g'); hold on
plot(x_sim_shift(1,:),x_sim_shift(2,:),'k')
plot(x_sim_shift(1,1),x_sim_shift(2,1),'b*')
plot(x_sim_shift(1,end),x_sim_shift(2,end),'r*')


%store data in .mat for QP simulation
f_shifted = to_function(f_shifted);
save('terminalIngredients.mat','f_shifted','gamma','W_fun','V_fun','xeq','ueq','umax','umin','Q','R')