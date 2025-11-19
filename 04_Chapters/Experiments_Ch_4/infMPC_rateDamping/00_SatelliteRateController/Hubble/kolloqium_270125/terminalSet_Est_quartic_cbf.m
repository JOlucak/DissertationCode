
clear
close all
clc

% system states
x = casos.PS('x',3,1);
u = casos.PS('u',3,1);

%% Hubble parameter
J = diag([31046;77217;78754]);

% simple bounds
omegaMax1 = 2*pi/180;
omegaMax2 = 1*pi/180;
omegaMax3 = 1*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';


% control constraint; assumption is that the box is inside the full
% pyramid. This is roughly estimated visually.
umin = [-1 -1 -1]'*1;
umax = [ 1  1  1]'*1;

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3))]);
Dxin = inv(Dx);

%% dynamics
% skew-symmetric matrix
skew = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% dynamics
f =  -J\skew(x(1:3))*J*x(1:3) + J\u; % omega_dot

% trim point
x0    = [0 0 0]';
u0    = [0,0,0]';

A = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

% LQR controller weights
Q = eye(3);
R = eye(3)*0.001;         

[K0,P] = lqr(full(A),full(B),Q,R);

% %scale initial lyapunov
Vval = x'*inv(Dx)'*P*inv(Dx)*x;


% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);

% state constraint
n = 4;
g0 = (x(1)^2/omegaMax1^2)^(n/2)+(x(2)^2/omegaMax2^2)^(n/2)+(x(3)^2/omegaMax3^2)^(n/2)-1;
g = subs(g0,x,Dx\x);


h  = casos.PS.sym('h',monomials(x,4));

P  = casos.PS.sym('p',monomials([x(1)^2 x(2)^2 x(3)^2]));
K  = casos.PS.sym('k',monomials(x(1),1),[3 1]);
for j = 1:3
    K(j) = casos.PS.sym('k',monomials(x(j),1));
end
s1 = casos.PS.sym('s1',monomials(x,4));
s2 = casos.PS.sym('s2',monomials(x,0:2));

s3 = casos.PS.sym('s3',monomials(x,0),[3 1]);
s4 = casos.PS.sym('s4',monomials(x,0),[3 1]);
s5 = casos.PS.sym('s5',monomials(x,0:2));

b = casos.PS.sym('b');

% options
opts = struct('sossol','mosek');

% adjust optimality thresholds
% opts.conVioTol     = 6e-8;
% opts.optTol        = 1e-1;
% opts.error_on_fail = 0;
opts.verbose       = 1;
opts.max_iter      = 100;


b = 1;
sos = struct('x',[h;P; K; s1; s2; s3; s4;s5],...
              'f',1/2*dot(g-(h-b),g-(h-b)), ... 
              'p',[]);


% constraints
sos.('g') = [
             s2;
             s3;
             s4;
             s5;
             s1*g  -  nabla(h,x)*subs(f,u,K) ;
             s2*g  -  (nabla(P,x)*subs(f,u,K) + (inv(Dx)*x)'*Q*(inv(Dx)*x) + K'*R*K);
             s3*g  + K - umin ; 
             s4*g  + umax - K ;
             s5*(h-b) - g
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% build sequential solver
buildTime_in = tic;
    solver_oneStepReach  = casos.nlsossol('S','filter-linesearch',sos,opts);
buildtime = toc(buildTime_in);

sol = solver_oneStepReach('x0',casos.PD([g;...
                                 Vval;
                                 -K0*x; ...         % unscaled initial guess
                                 x'*x; ...
                                 x'*x; ...
                                 ones(3,1)*(x'*x);...
                                 ones(3,1)*(x'*x); ...
                                 x'*x]));


%% check feasibility 
Psol    = sol.x(2); 
Ksol    = sol.x(3:5);
s1sol   = sol.x(6);
s2sol   = sol.x(7); 
s3sol   = sol.x(8:10); 
s5sol   = sol.x(11:13);   
bsol    = b;%sol.x(end);


%% plotting
import casos.toolboxes.sosopt.*

Psol_re = subs(Psol,x,Dx*x);
hsol_re = subs(sol.x(1),x,Dx*x)-bsol;
g_re    = subs(g0,x,x);

% plot in grad instead of rad; for pcontour the input is given in deg so we
% scale the input
figure(1)
deg2rad = diag([pi/180,pi/180,pi/180]);
clf
pcontour(subs(subs(hsol_re,x(3),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(3),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Invariant Set','Rate Bounds')


figure(2)
deg2rad = diag([pi/180,pi/180,pi/180]);
clf
pcontour(subs(subs(hsol_re,x(2),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(2),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Invariant Set','Rate Bounds')

figure(3)
deg2rad = diag([pi/180,pi/180,pi/180]);
clf
pcontour(subs(subs(hsol_re,x(1),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(1),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Invariant Set','Rate Bounds')

