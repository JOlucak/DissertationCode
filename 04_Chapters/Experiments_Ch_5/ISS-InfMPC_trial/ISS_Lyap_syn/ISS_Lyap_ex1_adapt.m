clc
clear

% system states
x = casos.Indeterminates('x',3);
w = casos.Indeterminates('w',1);
s = casos.Indeterminates('s',1);

% Dynamics Example 1
f = [-x(1)   + x(1)^2*x(2);
     -x(1)^3 - x(2) + w;
     -x(3)   - x(1)^2/2];


% Lyapunov function candidate
V = casos.PS.sym('v',monomials(x,1:4));

% K_inf (see Example 1)
a      = casos.PS.sym('ca',monomials(s^4));
a_ubar = casos.PS.sym('cau',monomials(s^4));
a_bar  = casos.PS.sym('cao',monomials([s^2 s^4]));
sigma  = casos.PS.sym('si',monomials([s^2 s^4]));


% constraints
[c_a_ubar,~]  = poly2basis(a_ubar);% helper to get "norm"
c_a_ubar      = casos.PS(c_a_ubar);% casos complaint becaus coeffs. are SX

 % Eq.(19)
g1 = V - c_a_ubar*(x'*x);


[c_a_bar,~]  = poly2basis(a_bar);% helper to get "norm"
c_a_bar      = casos.PS(c_a_bar);% casos complaint becaus coeffs. are SX

% Eq.(20)
g2 = c_a_bar(1)*(x'*x) + c_a_bar(2)*(x'*x)^2 - V; 

[c_sigma ,~] = poly2basis(sigma ); % helper to get "norm"
c_sigma      = casos.PS(c_sigma);  % casos complaint becaus coeffs. are SX
[c_a,~]      = poly2basis(a);      % helper to get "norm"
c_a          = casos.PS(c_a);      % casos complaint becaus coeffs. are SX

% Eq.(21) rewritten; see eq.(5) --> just makt the dissipation inequality
% hold
g3 = c_sigma(1)*(w'*w) + c_sigma(2)*(w'*w)^2 -nabla(V,x)*f - c_a*(x'*x)  ;


% Eq.(22)
g4 = s*nabla(a_bar,s);
g5 = s*nabla(a_ubar,s);
g6 = s*nabla(sigma,s);
g7 = s*nabla(a,s);

sos = struct('x',[V;a_bar;a_ubar;a;sigma], ...
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