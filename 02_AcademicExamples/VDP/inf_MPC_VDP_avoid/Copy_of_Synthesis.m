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


% control constraint; assumption is that the box is inside the full
% torque volume. This is roughly estimated visually.
umin = -1;
umax = 1;


%% system dynamics


% dynamics
mu =1;
f =  [x(2);
      mu*(1-x(1)^2)*x(2) - x(1) + u];

% trim point/equilibrium point
x0    = [0 0 ]';
u0    = 0;

A0 = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B0 = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

% cost function weights
Q = diag([1, 1]);
R = 2.5;

% generate an initial guess for CLF
[K0,P0] = lqr(full(A0),full(B0),Q,R);

% scaled initial guess for CLF
Wval = x'*P0*x;

% state constraint 
g = x(1)^2+3*x(2)^2-1;


%% setup SOS problem
load initGuess_termSet.mat

% CBF
W  = casos.PS.sym('w',monomials(x,2));

% CLF
V  = casos.PS.sym('v',monomials(x,2)); % we used adjusted monomial here i.e, we removed very small terms found by the intial guess

% Control law 
K = casos.PS.sym('k',monomials(x,1));

% SOS mulitplier
s1 = casos.PS.sym('s1',monomials(x,2));
s2 = casos.PS.sym('s2',monomials(x,2));
s3 = casos.PS.sym('s3',monomials(x,0),[3 1]);
s4 = casos.PS.sym('s4',monomials(x,0),[3 1]);
s5 = casos.PS.sym('s5',monomials(x,4));

% fixed level for CBF
b = 1;

% options for sequential sos
opts = struct('sossol','mosek');

opts.verbose       = 1;
opts.max_iter      = 100;


tau = ( nabla(V,x)*subs(f,u,K)    + K'*R*K + x'*Q*x); 

cost = dot(g-W,g-W);

sos = struct('x', [W;V;K;s1;s2;s3; s4;s5],...  % decision variables
              'f', cost ,...                       % cost function
              'p',[]);                             % parameter

% K-function parameter for CBF, i.e., gamma(W(x)) = gamma*W(x)
gamma = 1; % heuristically found

% constraints
sos.('g') = [
             s1
             s2;
             s3;
             s4;
             s5;
             V - 1e-6*(x'*x);                                    % CLF positivity              
             s1*(W-b) - g;                                       % State constraints
             s2*(W-b)  -  nabla(W,x)*subs(f,u,K)- gamma*(W-b);   % CBF
             s3*(W-b)  + K-umin;                                 % control constraints
             s4*(W-b)  + umax-K;
             s5*(W-b) -  tau ;                                   % CLF dissipation
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% solver setup
S  = casos.nlsossol('S','filter-linesearch',sos,opts);

% initial guess for sequential
x0 = casos.PD([ g; ...
                Wval;
                1
                ones(3,1);
                ones(3,1);
                x'*x; ...
                x'*x; ...
                x'*x]);

% load initial guess
% load initGuess_termSet.mat
% x0 = casos.PD(monoGuess,coeffGuess);

%  solve
sol = S('x0',x0);

bsol = b;

% re-scale invariant set, terminal penalty and local control law
Wsol_re = subs(sol.x(1),x,x) - full(casos.PD(bsol)); % write CBF as sublevel set 
Vsol_re = subs(sol.x(2),x,x);                        % CLF

% Control law (parameter found on scaled states)

Ksol_re = subs(sol.x(3),x,x);

%% plotting
import casos.toolboxes.sosopt.*

figure(1)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour(Wsol_re,0,[-1 1 -1 1],'g')
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
safety_fun  = to_function(Wsol_re);  %CBF
Kfun        = to_function(Ksol_re);

% generate sample rate with the individual boubds
nSample = 1000000;          % number of samples

a1 = -1*pi/180;
b1 = 1*pi/180;
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
penalties = full(penalty(samples(1,idx),samples(2,idx)));


% visualize samples of penalty as histogram
figure(3)
histogram(penalties.*penalties)
ylabel('No. of samples')
xlabel('Squared distance')

% maximum value 
max_penalty = max(penalties);

if max_penalty > 0 
    fprintf('Something went wrong. The maximum penalty is %d\n',max_penalty )
else
    fprintf('Everything seems fine! The maximum penalty is %d\n',max_penalty )
end

%% store initial guess
[coeffGuess,monoGuess] = poly2basis(remove_coeffs(sol.x,1e-6));

% store monomial basis
[~,monoGuessV]  = poly2basis(remove_coeffs(sol.x(2),1e-6));
[~,monoGuessS5] = poly2basis(remove_coeffs(sol.x(end),1e-6));

save('initGuess_termSet.mat','coeffGuess','monoGuess','monoGuessV','monoGuessS5')

%% save ingredients for MPC
W_fun = to_function(Wsol_re);
V_fun = to_function(Vsol_re);
K_fun = to_function(Ksol_re);

% save for infinitesimal MPC in current folder
save('terminalIngredients.mat','W_fun','V_fun', 'K_fun', ... % polynomials
    'Q','R', ...                                             % weights
    'umin','umax')   % parameter

% % store also in poly_controlLaw folder
% cd ..\poly_controlLaw\
% 
% save('terminalIngredients.mat','W_fun','V_fun', 'K_fun', ... % polynomials
%     'Q','R', ...                                             % weights
%     'umin','umax','omegaMax1','omegaMax2','omegaMax3','J')   % parameter
% 
% % come back to current folder
% cd ..\inf_MPC\
% 
