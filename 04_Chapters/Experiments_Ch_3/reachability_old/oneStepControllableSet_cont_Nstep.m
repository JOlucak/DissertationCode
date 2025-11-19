% Inner-approximation reachable set Van-der-Pol Oscillator

clear
close all
clc

import casos.toolboxes.sosopt.*

% system states
x = casos.PS('x',2,1);
u = casos.PS('u',1,1);
t  = casos.PS('t');
t0 = casos.PS.sym('t0');
t1 = casos.PS.sym('t1');

% system dynamics
f = [x(2);
        (1-x(1)^2)*x(2)-x(1)];

gx = [0;1];

% terminal region
P = [6.4314    0.4580
    0.4580    5.8227];

% actual terminal set; not needed
l = x'*P*x-1;            
lT = l;
% for simplicity assume this is the safe set!
g = 3*x(2)^2 + x(1)^2 -1;  % g(x) \leq 0


% to ensure feasibility we make use of a scaled safe set, i.e., we make the
% starting condition slightly smaller; a inner-approx. of the inner-approx.
% l = 3*x(2)^2 + x(1)^2 - 0.95;

Vval = x'*P*x;
% control constraint
umin = -1;
umax = 1;

% Time horizon is double of dt
T  = 0.2;

% time polynomial
hT = (t)*(T-t);

% Vval = l;
% 
% b  = casos.PS.sym('b');
b = 0;
V  = casos.PS.sym('v',monomials([x;t],0:4));
l0  = casos.PS.sym('l0',monomials([x;t],0:4));
K  = casos.PS.sym('k',monomials(x,0:2));
s1 = casos.PS.sym('s1',monomials([x;t],0:4));
s2 = casos.PS.sym('s2',monomials([x;t],0:2));
s3 = casos.PS.sym('s3',monomials([x;t],0:2));
s4 = casos.PS.sym('s4',monomials([x;t],0:4));
s5 = casos.PS.sym('s5',monomials([x;t],0:2));
s6 = casos.PS.sym('s6',monomials([x;t],0:4));
s7 = casos.PS.sym('s7',monomials([x;t],0:2));
s8 = casos.PS.sym('s8',monomials([x;t],0));
s9 = casos.PS.sym('s9',monomials(x,0:2));
s10 = casos.PS.sym('s10',monomials(x,0:2));
s11 = casos.PS.sym('s11',monomials(x,0:4));

% options
opts = struct('sossol','mosek');

% adjust optimality thresholds
% opts.conVioTol = 1e-2;
% opts.optTol    = 20;
opts.error_on_fail = 0;
opts.verbose = 1;
% opts.conViolCheck = 'pseudo';

sos = struct('x',[V; K; s1; s2; s3; s4; s5; s6;s7;s8;s9;s10;s11],...
              'f', dot(g-V,g-V) + dot(l-subs(V,t,T), l-subs(V,t,T)), ...
              'p',l0);



% constraints
sos.('g') = [s1;
             s2; 
             s3;
             s4; 
             s5;
             s6;
             s7;
             s8;
             s9;
             s10;
             s11;
             % dissipation inequality
             s1*(V) - s2*hT - nabla(V,t) - nabla(V,x)*(f+gx*K);
             % control constraints
             s3*(V) - s4*hT + K - umin;
             s5*(V) - s6*hT + umax - K;
             % constraint satisfaction
             s7*(V) - s8*hT - g; 
             % grow conditions
             s9*l0  - subs(V,t,0) ;
             % terminal conditions
             s10*(subs(V,t,T))    - l0 
             % contraction constraint
             s11*(subs(V,t,0.2)) + subs(V,t,0.2) - subs(V,t,0.1) - 1e-6*(x'*x) 
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% build sequential solver
buildTime_in = tic;
    solver_oneStepReach  = casos.nlsossol('S','sequential',sos,opts);
buildtime = toc(buildTime_in);


timeHor = 2;
N = timeHor/T;

for k = 1:N

if k == 1   
    sol = solver_oneStepReach('x0',casos.PD([ Vval;  ...
                                              x'*x; ...
                                              x'*x; ...
                                              x'*x; ...
                                              x'*x;
                                              x'*x;
                                              x'*x;
                                              x'*x;
                                              x'*x;
                                              x'*x;
                                              x'*x;
                                              x'*x;
                                              x'*x]), ...
                                              'p',casos.PD(l)); 
else
   sol = solver_oneStepReach('x0',casos.PD(sol.x), ...
                             'p',casos.PD(l)); 
end

l = subs(sol.x(1),t,0);


end

disp(['Solver buildtime: ' num2str(buildtime), ' s'])

Vval  = sol.x(1);
Kval  = sol.x(2);
% s1val = sol.x(3);
% s2val = sol.x(4);
% s3val = sol.x(5);
% s4val = sol.x(6);
% s5val = sol.x(7);
% s6val = sol.x(8);
% s7val = sol.x(9);
% s8val = sol.x(10);
% s9val = sol.x(11);
% s10val = sol.x(12);
% s11val = sol.x(13);



%% plotting
import casos.toolboxes.sosopt.*

figure(1)
pcontour(subs(sol.x(1),t,0),0,[-1 1 -1 1],'b')
hold on 
pcontour(g,0,[-1 1 -1 1],'k')
% pcontour(lT,0,[-1 1 -1 1],'r')

pcontour(subs(sol.x(1),t,T/2),0,[-1 1 -1 1],'g--')

% prepare reachable set and terminal set for MPC
V0 = subs(sol.x(1),t,0);
VT = subs(sol.x(1),t,0.1);

h_T = subs(sol.x(2),t,0.1);

Q = eye(2);
R = 1;

W = x'*Q*x + u'*R*u;

alpha0  = casos.PS.sym('alpha0');
s       = casos.PS.sym('s',monomials(x,1),'gram');


sos = struct('x',[alpha0;s],'f',alpha0);


% make use of Euler-discretization
dt = 0.1;
V1 = subs(VT,x,x+dt*(f + gx*h_T));

deltaV =cleanpoly( VT - V1, 1e-6);

% check set-inclusion visually
domain = [-10 10 -10 10];
figure(2)
clf
pcontour(VT,0,domain,'b')
hold on
pcontour(deltaV,0,domain,'k')
legend('V(1,x)','V(1,x)-V(1,f(x,h(1,x)) )')

domain = [-1 1 -1 1]*0.5;
figure(3)
clf
pcontour(VT,0,domain,'b')
hold on
pcontour(deltaV,0,domain,'k')
legend('V(1,x)','V(1,x)-V(1,f(x,h(1,x)) )')


sos.('g') = s*VT + alpha0*deltaV  - subs(W,u,h_T);

opts = struct();

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', 1,'sos', 1);
opts.Kc = struct('sos', 1);

% build sequential solver
buildTime_in = tic;
    solver_alpha0  = casos.sossol('S','scs',sos,opts);
buildtime = toc(buildTime_in)

tic
sol_alpha0 = solver_alpha0()
toc
alpha0 =full(sol_alpha0.x(1))

V0 = to_function(V0);
V1 = to_function(VT);

save('VDP_horizon5s.mat','V0',"V1","alpha0")

