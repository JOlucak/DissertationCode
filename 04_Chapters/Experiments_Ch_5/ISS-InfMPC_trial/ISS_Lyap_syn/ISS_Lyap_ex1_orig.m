

% system states
x = casos.Indeterminates('x',3);
w = casos.Indeterminates('w',1);
s = casos.Indeterminates('s',1);

% Dynamics Example 1
f = [-x(1) + x(1)^2*x(2);
     -x(1)^3-x(2)+w;
     -x(3)-x(1)^2/2];


% Lyapunov function candidate
V = casos.PS.sym('v',monomials(x,2:4));
s1 = casos.PS.sym('s1',monomials(x,2));
s2 = casos.PS.sym('s2',monomials(x,2));
s3 = casos.PS.sym('s3',monomials([x;w],2));

% K_inf (see Example 1)
a      = casos.PS.sym('ca',monomials(s^2));
a_ubar = casos.PS.sym('cau',monomials(s^2));
a_bar  = casos.PS.sym('cao',monomials([s^2 s^4]));
sigma  = casos.PS.sym('si',monomials([s^2 s^4]));


s4 = casos.PS.sym('s4',monomials(s,2));
s5 = casos.PS.sym('s5',monomials(s,2));
s6 = casos.PS.sym('s6',monomials(s,2));
s7 = casos.PS.sym('s7',monomials(s,2));


% constraints
% Eq.(19)
[c_a_ubar,~]  = poly2basis(a_ubar);% helper to get "norm"
c_a_ubar      = casos.PS(c_a_ubar);% casos complaint becaus coeffs. are SX

g1 = V-c_a_ubar*(x'*x) - s1;

% Eq.(20)
[c_a_bar,~]  = poly2basis(a_bar);% helper to get "norm"
c_a_bar      = casos.PS(c_a_bar);% casos complaint becaus coeffs. are SX

g2 = c_a_bar(1)*(x'*x) + c_a_bar(2)*(x'*x)^2 - V - s2;

% Eq.(21)
[c_sigma ,~] = poly2basis(sigma ); % helper to get "norm"
c_sigma      = casos.PS(c_sigma); % casos complaint becaus coeffs. are SX

[c_a,~]  = poly2basis(a);% helper to get "norm"
c_a      = casos.PS(c_a);% casos complaint becaus coeffs. are SX

g3 = nabla(V,x)*f + c_a*(x'*x) - c_sigma(1)*(w'*w) - c_sigma(2)*(w'*w)^2 + s3;

% Eq.(22)
g4 = s*nabla(a_bar,s) - s4;
g5 = s*nabla(a_ubar,s) - s5; 
g6 = s*nabla(sigma,s) - s6;
g7 = s*nabla(a,s) - s7;


g_lin = [g1; g2; g3; g4; g5; g6; g7];
g_sos = [s1; s2; s3; s4; s5; s6; s7];

sos = struct('x',[V;a_bar;a_ubar;a;sigma;s1;s2;s3;s4;s5;s6;s7], ...
              'g',[g_lin;g_sos]);

% states + constraint are SOS cones
opts.Kx = struct('lin',12);
opts.Kc = struct('lin',7,'sos',7);
opts.newton_solver = [];
% get solver
S = casos.sossol('S','mosek',sos,opts);

sol = S();

sol.x