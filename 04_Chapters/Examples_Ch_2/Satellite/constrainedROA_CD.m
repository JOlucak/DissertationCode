%% ------------------------------------------------------------------------
%
%
%   Short Descirption:  Calculate an inner-estimate of the
%                       region-of-attraction for the longitudinal motion 
%                       of the Nasa Generic Transport Model. To increase
%                       the size of the sublevel set we try to minimize the
%                       squared distance to a defined set. Additionally, we
%                       synthesis a linear control law at the same time.
%                       State and control constraints are considered.
%
%   Reference: Modified problem from:
%              Chakraborty, Abhijit and Seiler, Peter and Balas, Gary J.,
%              Nonlinear region of attraction analysis for flight control 
%              verification and validation, Control Engineering Practice,
%              2011, doi: 10.1016/j.conengprac.2010.12.001
%           
%
%--------------------------------------------------------------------------

clear 
close all
clc

import casos.toolboxes.sosopt.cleanpoly

% system states
x = casos.PS('x',6,1);
u = casos.PS('u',3,1);

%% Hubble telescope parameter
J = diag([3104;7721;7875]);

% simple bounds on rates;
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';


% control constraint; assumption is that the box is inside the full
% torque volume. This is roughly estimated visually.
umin = [-1 -1 -1]'*1.2;
umax = [ 1  1  1]'*1.2;

n = 2;
gu = (u(1)^2/umax(1)^2)^(n/2) + (u(2)^2/umax(2)^2)^(n/2) + (u(3)^2/umax(3)^2)^(n/2) - 1;

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3)),0.5,0.5,0.5]);

Dxin = inv(Dx);

%% dynamics
% skew-symmetric matrix
cpm = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

f =  [-J\cpm(x(1:3))*J*x(1:3) + J\u;
      1/4*B(x(4:6))*x(1:3)]; 

% generate an initial guess for CLF
x0    = [0 0 0 0 0 0]';
u0    = [0,0,0]';

A = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

Q = diag([1, 1, 1, 1,1 ,1]);
R = eye(3)*1;
[K0,P0] = lqr(full(A),full(B),Q,R);

% scaled initial guess for terminal penalty (Lyapunov linear system)
Vval = (inv(Dx)*x)'*P0*(inv(Dx)*x);

% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);

% allowable set (inner-approximation of box constraints via superquadric)
n = 4;
g0 = (x(1)^2/omegaMax1^2)^(n/2) + (x(2)^2/omegaMax2^2)^(n/2) + (x(3)^2/omegaMax3^2)^(n/2) + ...
     (x(4)^2/0.57^2)^(n/2) + (x(5)^2/0.57^2)^(n/2) + (x(6)^2/0.57^2)^(n/2) - 1;

% re-scale input of state constraints
g = subs(g0,x,Dx\x); 

% Lyapunov function candidate
V = casos.PS.sym('v',monomials(x,2));

% SOS multiplier
s1    = casos.PS.sym('s1',monomials(x,2));
s2    = casos.PS.sym('s2',monomials(x,0),[3 1]);
s3    = casos.PS.sym('s3',monomials(x,0),[3 1]);
s4    = casos.PS.sym('s4',monomials(x,0:2));

% control law(s)
K1  = casos.PS.sym('k1',monomials(x,0),[3 1]);
for j = 1:3
 K1(j) = casos.PS.sym('k2',monomials([x(j)]));
end

K2  = casos.PS.sym('k',monomials(x,0),[3 1]);
for j = 1:3
 K2(j) = casos.PS.sym('k',monomials([x(j+3)]));
end

kappa = K1+K2;

kappa = -K0*inv(Dx)*x;
b     = casos.PS.sym('b');

% enforce positivity
l = 1e-6*(x'*x);


%% setup solver
% options
% opts = struct('sossol','mosek');
% opts.verbose  = 1;
% opts.error_on_fail = 0;
% opts.conf_interval = [-0.01 0];

% cost
% cost = dot(g-(V-b),g-(V-b)) ;
b = 0.1;
sos = struct('x',[s1;s2;s3;s4],... % decision variables
              'p',V);                        % parameter

% SOS constraints
sos.('g') = [s1;
             % s2;
             % s3;
             % s4;
             s1*(V-b)-nabla(V,x)*subs(f,u,kappa)-l;
             % s2*(V-b) + kappa - umin;
             % s3*(V-b) + umax - kappa;
             % s4*(V-b) - g
             ];

% states + constraint cones
opts.Kx      = struct('lin', length(sos.x));
opts.Kc      = struct('sos', length(sos.g));

% setup solver
% S1 = casos.qcsossol('S1','bisection',sos,opts);
S1 = casos.sossol('S1','mosek',sos,opts);
%% solve problem


% solve problem
sol = S1('x0' ,Vval);

casos.postProcessSolver(S,true);

S.stats

S.stats.single_iterations{end}.conic

%% plotting
figure(1)

% re-scale solution
xd = Dx*x;

Vfun = to_function(subs(sol.x(1),x,xd));
gfun = to_function(subs(g,x,xd));

fcontour(@(x4,x5) full(Vfun(0,0,0, x4,x5,0) ), [-1 1 -4 4 ], 'b-', 'LevelList', full(sol.x(end)))
hold on
fcontour(@(x4,x5)  full(gfun(0,0,0, x4,x5,0) ), [-1 1 -4 4 ], 'r-', 'LevelList', 0)
hold off
legend('Lyapunov function','Safe set function')


figure(2)

% re-scale solution
xd = Dx*x;

Vfun = to_function(subs(sol.x(1),x,xd));
gfun = to_function(subs(g,x,xd));

fcontour(@(x1,x2) full(Vfun(x1,x2,0, 0,0,0) ), [-1 1 -4 4 ]*pi/180, 'b-', 'LevelList', full(sol.x(end)))
hold on
fcontour(@(x1,x2)  full(gfun(x1,x2,0, 0,0,0) ), [-1 1 -4 4 ]*pi/180, 'r-', 'LevelList', 0)
hold off
legend('Lyapunov function','Safe set function')

