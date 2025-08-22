%% ------------------------------------------------------------------------
%   
%
%   Short Description: Script to syntheszise the non-conflicting CBF/CLF for 
%                      a simple rendevouz problem. Only control constraints
%                      and simple box constraints are considered.
%
%
%   Needed software: - CasADi 3.6 (for CaSoS)
%                    - CaSoS
%
%
% ------------------------------------------------------------------------

clear
close all
clc

% system states
x = casos.PS('x',6,1);
u = casos.PS('u',3,1);

%% Chaser Parameter
m = 6500; % Kg
R0 = 705;


% control constraint; assumption is that the box is inside the full
% torque volume. This is roughly estimated visually.
umin = 0;
umax = 20;

% scaling matrix for system states
Dx =eye(6);
Dxin = inv(Dx);


n = 4;
g0 = (x(1)^2/1000^2)^(n/2) + (x(2)^2/1000^2)^(n/2) + (x(3)^2/1000^2)^(n/2)+ ...
     (x(4)^2/10^2)^(n/2) + (x(5)^2/10^2)^(n/2) + (x(6)^2/10^2)^(n/2)-1 ;

Dx   = diag([1/(1000),1/(1000),1/(1000),1/10,1/10,1/10]);
%% system dynamics
mu = 3.986*10^5;
n = sqrt(mu/R0^3);

A = [ 0     0   0   1    0  0;
      0     0   0   0    1  0;
      0     0   0   0    0  1;
     3*n^2  0   0   0   2*n 0;
      0     0   0 -2*n   0  0;
      0     0 -n^2  0    0  0];

B = [zeros(3,3);eye(3)./m];
     
f = A*x+B*u;

% cost function weights
Q = diag([1, 1, 1, 1, 1 ,1]);
R = eye(3);

% generate an initial guess for CLF
[K0,P0] = lqr(full(A),full(B),Q,R);

% scaled initial guess for CLF
Vval = (inv(Dx)*x)'*P0*(inv(Dx)*x);

% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);

% re-scale input of state constraints
g = subs(g0,x,Dx\x); 

%% setup SOS problem
% CLF
V  = casos.PS.sym('v',monomials(x,2)); 

% SOS mulitplier
s1 = casos.PS.sym('s1',monomials(x,2)); %
K  = casos.PS.sym('K',monomials(x,1),[3 1]);
s2 = casos.PS.sym('s2',monomials(x,2));
s3 = casos.PS.sym('s3',monomials(x,0));
s4 = casos.PS.sym('s4',monomials(x,0));
b  = casos.PS.sym('b');

% options for sequential sos
opts = struct('sossol','mosek');
opts.error_on_fail = 0;
opts.verbose       = 1;


sos = struct('x', [s1;s2;V;K;s3;s4;b],...  % decision variables
             'f',dot(g-(V-b),g-(V-b)),...
              'p',[]);                             % parameter


% constraints
sos.('g') = [
             s1;     
             s2;
             s3;
             s4;
             s2*g - nabla(V,x)*subs(f,u,K);  
             s3*g + umax^2 - K'*K
             s4*g +  K'*K - umin
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% solver setup
S  = casos.nlsossol('S','filter-linesearch',sos,opts);

x0 =casos.PD([x'*x; x'*x; x'*x; ones(3,1)*(x'*x); ones(2,1);1]);

%  solve
sol = S('x0',x0);

bsol = sol.x(end);

% re-scale invariant set, terminal penalty and local control law
Vsol_re = subs(sol.x(3),x,Dx*x) - full(casos.PD(bsol)); % write CBF as sublevel set 


%% plotting
import casos.toolboxes.sosopt.*

% slice for rates, i.e. sigma = [0,0,0]' satellite is aligned with inertial
% frame
figure(100)
pcontour(subs(Vsol_re,x(3:6),0),0,[-10 10 -10 10]*100,'k-')
hold on
pcontour(subs(g0,x(3:6),0),0,[-10 10 -10 10]*100,'r--')
