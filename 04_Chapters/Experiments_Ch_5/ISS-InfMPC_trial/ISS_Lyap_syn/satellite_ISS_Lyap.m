clc
clear

% system states
x = casos.PS('x',6);
u = casos.PS('u',3);
w = casos.PS('w',3);
s = casos.PS('s',1);

% Dynamics 


%% satellite simple

% simple bounds on rates;
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';

% scaling matrix for system states
Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3)),1,1,1]); 
% 
% Dxin = inv(Dx);

Dx = eye(6);

J = diag([1;1;1]);

% cross-product matrix
cpm = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% MRP derivative
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

% dynamics
f =  [-J\cpm(x(1:3))*J*x(1:3) + J\u; % omega_dot
      1/4*B(x(4:6))*x(1:3)];         % sigma_dot

Kd = diag([3;7;10]);
Kp = diag([0.6;1.1;1.55]);

K = -Kd*x(1:3) - Kp*x(4:6); 

f = subs(f,u,K);

f = Dx*subs(f,x,inv(Dx)*x);

% Lyapunov function candidate
V = casos.PS.sym('v',monomials(x,2:4));

% K_inf (see Example 1)
a      = casos.PS.sym('ca',monomials([s^2 s^4]));
a_ubar = casos.PS.sym('cau',monomials([s^2 s^4]));
a_bar  = casos.PS.sym('cao',monomials([s^2 s^4]));
sigma  = casos.PS.sym('si',monomials([s^2 s^4]));


% constraints
[c_a_ubar,~]  = poly2basis(a_ubar);% helper to get "norm"
c_a_ubar      = casos.PS(c_a_ubar);% casos complaint becaus coeffs. are SX


g1 = V - c_a_ubar(1)*(x'*x) - c_a_ubar(2)*(x'*x)^2;

[c_a_bar,~]  = poly2basis(a_bar);% helper to get "norm"
c_a_bar      = casos.PS(c_a_bar);% casos complaint becaus coeffs. are SX


g2 = c_a_bar(1)*(x'*x) + c_a_bar(2)*(x'*x)^2 - V; 

[c_sigma ,~] = poly2basis(sigma ); % helper to get "norm"
c_sigma      = casos.PS(c_sigma);  % casos complaint becaus coeffs. are SX
[c_a,~]      = poly2basis(a);      % helper to get "norm"
c_a          = casos.PS(c_a);      % casos complaint becaus coeffs. are SX

g3 = c_sigma(1)*(w'*w) + c_sigma(2)*(w'*w)^2 -nabla(V,x)*f - c_a(1)*(x'*x) - c_a(2)*(x'*x)^2  ;


g4 = s*nabla(a_bar,s);
g5 = s*nabla(a_ubar,s);
g6 = s*nabla(sigma,s);
g7 = s*nabla(a,s);

sos = struct('x',[V; a_bar; a_ubar; a; sigma], ...
              'g',[g1;g2;g3;g4;g5;g6;g7]);

% states + constraint are SOS cones
opts.Kx = struct('lin',5);
opts.Kc = struct('sos', 7);
opts.newton_solver = [];

% get solver
S = casos.sossol('S','mosek',sos,opts);

sol = S();
S.stats.UNIFIED_RETURN_STATUS


remove_coeffs(sol.x,sqrt(eps))