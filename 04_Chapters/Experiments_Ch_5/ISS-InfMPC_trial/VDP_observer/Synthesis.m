%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Script to syntheszise the non-conflicting CBF/CLF for 
%                      the comparative study for the infinetesimal-MPC approach. 
%                      The CBF/CLF is pre-computed  and  stored in .mat file, 
%                      which is the used in   inf_MPC_simulation. The 
%                      synthesized control law is also  stored and used for 
%                      comparison, but in a separate folder.
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
x = casos.PS('x',2,1);
u = casos.PS('u',1,1);
w = casos.PS('w',1,1);
s = casos.Indeterminates('s',1);


% control constraint
umin = -1;
umax =  1;

wmax = 0.2;

%% system dynamics

% dynamics
mu =1;
% nominal
f0 =  [x(2);
      mu*(1-x(1)^2)*x(2) - x(1) - u];

% full
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

% scaled initial guess for CLF
hval = x'*P0*x;

% state constraint 
g = x(1)^2+3*x(2)^2-1;


%% setup SOS problem
load initGuess_termSet.mat

% CBF
h  = casos.PS.sym('h',monomials(x,1:2));

% CLF
V  = casos.PS.sym('v',monomials(x,2)); 

% Control law 
K = casos.PS.sym('k',monomials([x;w],1));

% SOS mulitplier
s1 = casos.PS.sym('s1',monomials(x,2));
s2w = casos.PS.sym('s1',monomials([x;w],2));
s2 = casos.PS.sym('s2',monomials([x;w],0:2));

s3 = casos.PS.sym('s3',monomials([x;w],0),[length(u) 1]);
s4 = casos.PS.sym('s4',monomials([x;w],0),[length(u) 1]);


s3w = casos.PS.sym('s3w',monomials([x;w],2),[length(u) 1]);
s4w = casos.PS.sym('s4w',monomials([x;w],2),[length(u) 1]);

s5 = casos.PS.sym('s5',monomials([x;w],2));
s5w = casos.PS.sym('s5w',monomials([x;w],2));


% Kinf functions
a      = casos.PS.sym('ca',monomials(s^4));
a_ubar = casos.PS.sym('cau',monomials(s^4));
a_bar  = casos.PS.sym('cao',monomials([s^2 s^4]));
sigma  = casos.PS.sym('si',monomials([s^2 s^4]));


[c_a_ubar,~]  = poly2basis(a_ubar);% helper to get "norm"
c_a_ubar      = casos.PS(c_a_ubar);% casos complaint becaus coeffs. are SX

[c_a_bar,~]  = poly2basis(a_bar);% helper to get "norm"
c_a_bar      = casos.PS(c_a_bar);% casos complaint becaus coeffs. are SX


[c_sigma ,~] = poly2basis(sigma ); % helper to get "norm"
c_sigma      = casos.PS(c_sigma);  % casos complaint becaus coeffs. are SX
[c_a,~]      = poly2basis(a);      % helper to get "norm"
c_a          = casos.PS(c_a);      % casos complaint becaus coeffs. are SX


% fixed level for CBF
b = 1;

% options for sequential sos
opts = struct('sossol','mosek');

opts.verbose       = 1;
opts.max_iter      = 100;


tau = ( nabla(V,x)*subs(f,u,K)    + K'*R*K + x'*Q*x); 

cost = dot(g-(h-b), g-(h-b));

sos = struct('x', [h;V;K;s1;s2w;s2;s3; s4;s3w; s4w;s5;s5w;a_bar;a_ubar;a;sigma],...  % decision variables
              'f', cost ,...                                % cost function
              'p',[]);                                      % parameter

% K-function parameter for CBF, i.e., gamma(W(x)) = gamma*W(x)
gamma = 0.1; % heuristically found

% constraints
sos.('g') = [
             s1
             s2w
             s2;
             s3;
             s4;
             s3w;
             s4w;
             s5;
             s5w
             V - c_a_ubar*(x'*x);                                                                           % CLF positivity  
             c_a_bar(1)*(x'*x) + c_a_bar(2)*(x'*x)^2 - V
             s1*(h-b)  - g;                                                                                 % State constraints
             s2*(h-b)  -  nabla(h,x)*subs(f,u,K) - gamma*(h-b) + s2w*(w'*w - wmax^2);                       % CBF
             s3*g  + K-umin + s3w*(w'*w - wmax^2);                                                          % control constraints
             s4*g  + umax-K + s4w*(w'*w - wmax^2);   
             s5*(h-b) -  tau + c_sigma(1)*(w'*w) + c_sigma(2)*(w'*w)^2 - c_a*(x'*x) + s5w*(w'*w - wmax^2);   
             s*nabla(a_bar,s);
             s*nabla(a_ubar,s);
             s*nabla(sigma,s);
             s*nabla(a,s);
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));
opts.sossol_options.newton_solver = [];
% profile on
% solver setup
S  = casos.nlsossol('S','sequential',sos,opts);
% profile viewer
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

% % evaluate control law and check if commanded control torques lie in bounds
% uval = full(Kfun(samples(1,idx),samples(2,idx)));
% 
% if any(uval(1,:) > umax(1)) || any(uval(1,:) <  umin(1)) 
% 
%         fprintf('Control Constraints are not met at sampling points!!!!!\n' )
% else
%     fprintf('Control Constraints are met at sampling points\n' )
% 
% end

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

% %% save ingredients for MPC
W_fun = to_function(hsol_re);
V_fun = to_function(Vsol_re);
K_fun = to_function(Ksol_re);

% save for infinitesimal MPC in current folder
save('terminalIngredients.mat','W_fun','V_fun', 'K_fun', ... % polynomials
    'Q','R', ...                                             % weights
    'umin','umax')                                           % parameter


