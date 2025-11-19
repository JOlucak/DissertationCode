%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak and Torbjørn Cunis
%
%   Short Description: Script to syntheszise caompatible robust CBF and
%                      ISS-CLF for the Van-der-Pol Oscillator. 
%                      
%
%   Needed software: - CasADi 3.6 (for CaSoS)
%                    - CaSoS
%
% ------------------------------------------------------------------------

clear
close all
clc


nx = 2;
nu = 1;
nw = 1;

% system states
x = casos.PS('x',nx,1);
% controls
u = casos.PS('u',nu,1);
% input disturbance
w = casos.PS('w',nw,1);
% for K_inf. functions, i.e., scalar real even univariate polynomials
s = casos.Indeterminates('s',1);

% control constraint
umin = -1;
umax =  1;

% max. disturbance
wmax = 0.07;

%% system dynamics

% dynamics
mu = 1;

% nominal
f0 =  [x(2);
      mu*(1-x(1)^2)*x(2) - x(1) - u];

% full (affine in u and w)
f = f0 +[0;1]*w;

% trim point/equilibrium point
x0    = [0 0]';
u0    = 0;

A0 = full(casos.PD(subs(nabla(f0,x),[x;u],[x0;u0])));
B0 = full(casos.PD(subs(nabla(f0,u),[x;u],[x0;u0])));

% cost function weights
Q = diag([1, 1]);
R = 1;

% generate an initial guess for CLF
[K0,P0] = lqr(full(A0),full(B0),Q,R);

% initial guess for CLF
hval = x'*P0*x;

% state constraint 
x_1 = x(1);
x_2 = x(2);

 g = -0.1 - 20.9899*x_1^2 + 40.033*x_2^2 + 25.6545*x_1^4 - 25.5822*x_1^2 ...
  *x_2^2 + 12.9417*x_2^4;

%% setup SOS problem
load initGuess_termSet.mat

% CBF
h  = casos.PS.sym('h',monomials(x,2:4));

% CLF
V  = casos.PS.sym('v',monomials(x,2)); 

% Control law 
K = casos.PS.sym('k',monomials(x,1));

% ------------------ SOS multiplier ------------------
s1 = casos.PS.sym('s1',monomials(x,0)); % state constraint

s2  = casos.PS.sym('s2',monomials([x;w],2)); % CBF
s2w = casos.PS.sym('s1',monomials([x;w],2));

% control constraints
s3 = casos.PS.sym('s3',monomials(x,2:4),[length(u) 1]);
s4 = casos.PS.sym('s4',monomials(x,2:4),[length(u) 1]);

% CLF
s5  = casos.PS.sym('s5',monomials([x;w],2));
s5w = casos.PS.sym('s5w',monomials([x;w],2));

% Kinf functions (see Paper Ichihara)
a      = casos.PS.sym('ca',monomials(s^4));
a_ubar = casos.PS.sym('cau',monomials(s^4));
a_bar  = casos.PS.sym('cao',monomials([s^2 s^4]));
sigma  = casos.PS.sym('si',monomials([s^2 s^4]));

% -------------------------------------------------------------------------
% Work around for norm to setup the K-inf function manually
% -------------------------------------------------------------------------

[c_a_ubar,~]  = poly2basis(a_ubar);% helper to get "norm"
c_a_ubar      = casos.PS(c_a_ubar);% casos complaint because coeffs. are SX

[c_a_bar,~]  = poly2basis(a_bar);% helper to get "norm"
c_a_bar      = casos.PS(c_a_bar);% casos complaint because coeffs. are SX


[c_sigma ,~] = poly2basis(sigma ); % helper to get "norm"
c_sigma      = casos.PS(c_sigma);  % casos complaint because coeffs. are SX

[c_a,~]      = poly2basis(a);      % helper to get "norm"
c_a          = casos.PS(c_a);      % casos complaint because coeffs. are SX

% fixed level for CBF
b = 0.1;

% options for sequential sos
opts = struct('sossol','mosek');

opts.verbose       = 1;
opts.max_iter      = 100;

tau = ( nabla(V,x)*subs(f,u,K)    + K'*R*K + x'*Q*x - c_sigma(1)*(w'*w) - c_sigma(2)*(w'*w)^2 + c_a*(x'*x) ); 

cost = dot(g-(h-b), g-(h-b));

sos = struct('x', [h;V;K;s1;s2w;s2;s3; s4;s5;s5w;a_bar;a_ubar;a;sigma],...  % decision variables
              'f', cost ,...                                                % cost function
              'p',[]);                                                      % parameter

% K-function parameter for CBF, i.e., gamma(W(x)) = gamma*W(x)
gamma = 0.5;

% SOS multiplier from S-procedure
multiplier_constraints = [s1; 
                          s2w;
                          s2;
                          s3;
                          s4;
                          s5;
                          s5w];

CLF_constraints = [V - c_a_ubar*(x'*x);                         % CLF lower bound  
                  c_a_bar(1)*(x'*x) + c_a_bar(2)*(x'*x)^2 - V;  % CLF upper bound
                  s5*(h-b) -  tau  + s5w*(w'*w - wmax^2)
                  ];  

% K_inf functions (see Ichihara paper)
K_inf_constraints = [s*nabla(a_bar,s);
                     s*nabla(a_ubar,s);
                     s*nabla(sigma,s);
                     s*nabla(a,s)];
% constraints
sos.('g') = [
              multiplier_constraints;                                                       % SOS multiplier
              CLF_constraints                                                               % CLF constraints
              s1*(h-b)  - g;                                                                % State constraint(s)
              s2*(h-b)  -  nabla(h,x)*subs(f,u,K) - gamma*(h-b) + s2w*(w'*w - wmax^2); % CBF
              s3*(h-b)  + K-umin;                                                          % control constraints
              s4*(h-b)  + umax-K ;
              K_inf_constraints                                                              % K_inf
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% turn off Newton
opts.sossol_options.newton_solver = [];

% solver setup
S  = casos.nlsossol('S','sequential',sos,opts);

% initial guess for sequential
x0 = casos.PD([ g; ...
                hval;
                1
                x'*x;
                x'*x;
                x'*x;
                x'*x;
                x'*x;
                x'*x;
                x'*x; ...
                s'*s; ...
                s'*s; ...
                s'*s; ...
                s'*s]);

% % load initial guess
% load initGuess_termSet.mat
% x0 = casos.PD(monoGuess,coeffGuess);

%  solve
sol = S('x0',x0);

% bsol = full(sol.x(end));
bsol = b;

% re-scale invariant set, terminal penalty and local control lah
hsol_re = subs(sol.x(1),x,x) - full(casos.PD(bsol)); % hrite CBF as sublevel set 
Vsol_re = subs(sol.x(2),x,x);                        % CLF

% Control lah (parameter found on scaled states)

Ksol_re = subs(sol.x(3),x,x);

%% plotting
import casos.toolboxes.sosopt.*

figure(1)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour(hsol_re,0,[-1 1 -1 1],'g')
hold on 
pcontour(g,0,[-1 1 -1 1],'k--')
legend('Terminal Set','Safe Set')


conVio =  full(S.stats.single_iterations{end}.constraint_violation);
%% verification 

% ------------------------------------------------------------------------
% Comment: Altough we have the guarantee all SOS constraints are met if 
% feasible, we can run a simple sampling approach to check if the 
% sufficient conditions are met.
% ------------------------------------------------------------------------


% continous-time penalty, invariant set and control law as functions
penalty     = to_function(nabla(Vsol_re,x)*subs(f,u,Ksol_re) + Ksol_re'*R*Ksol_re + x'*Q*x) ;  % CLF/optimal value function
safety_fun  = to_function(hsol_re);  %CBF
Kfun        = to_function(Ksol_re);

% generate sample rate with the individual boubds
nSample = 1000000;          % number of samples

a1 = -1;
b1 =  1;
samples = (b1-a1)*rand(2,nSample)+a1;

% get all samples that lies within the invariant set, i.e.,  W(x_samp) <= 0
idx = find(full(safety_fun(samples(1,:),samples(2,:))) <= 0);

% evaluate control law and check if commanded control torques lie in bounds
uval = full(Kfun(samples(1,idx),samples(2,idx)));

if any(uval(1,:) > umax(1)) || any(uval(1,:) <  umin(1)) 

        fprintf('Control Constraints are not met at sampling points!!!!!\n' )
else
    fprintf('Control Constraints are met at sampling points\n' )

end

% evaluate the CLF at sample points that lie in safe set
% penalties = full(penalty(samples(1,idx),samples(2,idx)));
% 
% 
% % visualize samples of penalty as histogram
% figure(3)
% histogram(penalties.*penalties)
% ylabel('No. of samples')
% xlabel('Squared distance')
% 
% % maximum value 
% max_penalty = max(penalties);
% 
% if max_penalty > 0 
%     fprintf('Something went wrong. The maximum penalty is %d\n',max_penalty )
% else
%     fprintf('Everything seems fine! The maximum penalty is %d\n',max_penalty )
% end

%% store initial guess
[coeffGuess,monoGuess] = poly2basis(remove_coeffs(sol.x,1e-6));

% store monomial basis
[~,monoGuessV]  = poly2basis(remove_coeffs(sol.x(2),1e-6));
[~,monoGuessS5] = poly2basis(remove_coeffs(sol.x(end),1e-6));

save('initGuess_termSet.mat','coeffGuess','monoGuess','monoGuessV','monoGuessS5')

%% save ingredients for MPC
h_fun = to_function(hsol_re);
V_fun = to_function(Vsol_re);
K_fun = to_function(Ksol_re);
g_fun =  to_function(g);
% dynamics
xdot_nom = to_function(f0);
x_dot    = to_function(f);

% save for infinitesimal MPC in current folder
save('terminalIngredients.mat','h_fun','V_fun', 'K_fun','gamma', ... % polynomials
    'Q','R','g_fun', ...                                             % weights
    'x_dot','xdot_nom','nx','nu','nw',...
    'umin','umax')                                           % parameter


