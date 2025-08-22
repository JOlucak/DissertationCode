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


x1_max =  1000;
x1_min = -1000;

x2_max =  1000;
x2_min = -1000;

x3_max =  1000;
x3_min = -1000;


x4_max =  10;
x4_min = -10;

x5_max =  10;
x5_min = -10;

x6_max =  10;
x6_min = -10;

% control constraint; assumption is that the box is inside the full
% torque volume. This is roughly estimated visually.
umin = [0 0 0]'*10;
umax = [ 1  1  1]'*10;

% scaling matrix for system states
Dx =eye(6);
Dxin = inv(Dx);

% Dx = diag([1./(x1_max-x1_min),1./(x2_max-x2_min), 1./(x3_max-x3_min)...
%           1./(x4_max-x4_min),1./(x5_max-x5_min),1./(x5_max-x5_min)]);

%% system dynamics
mu = 3.986*10^5;
n = sqrt(mu/R0^3);


f = [x(3);
     x(4);
     x(5);
     u(1)/m+3*n^2*x(1)+ 2*n*x(4);
     u(2)/m-2*n*x(4);
     u(3)/m-2*n*x(3)];


% trim point/equilibrium point
x0    = [0 0 0 0 0 0]';
u0    = [0,0,0]';

A0 = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B0 = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

% cost function weights
Q = diag([1, 1, 1, 1, 1 ,1]);
R = eye(3)*1;

% generate an initial guess for CLF
[K0,P0] = lqr(full(A0),full(B0),Q,R);

% scaled initial guess for CLF
Wval = (inv(Dx)*x)'*P0*(inv(Dx)*x);

% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);

% state constraint as super-ellipsoid
% Remark: The MRP parameter are selected such that \sigma'*\sigma <= 1
n = 2;
g0 = (x(1)^2/x1_max^2)^(n/2) + (x(2)^2/x2_max^2)^(n/2) + (x(3)^2/x3_max^2)^(n/2) + ...
     (x(4)^2/10^2)^(n/2) + (x(5)^2/10^2)^(n/2) + (x(6)^2/10^2)^(n/2) - 1;

% re-scale input of state constraints
g = subs(g0,x,Dx\x); 

%% setup SOS problem
load initGuess_termSet.mat

% CBF
W  = casos.PS.sym('w',monomials(x,2));

% CLF
V  = casos.PS.sym('v',monoGuessV); % we used adjusted monomial here i.e, we removed very small terms found by the intial guess

% Control law 
Kd = casos.PS.sym('kd',[3,1]);
Kp = casos.PS.sym('kp',[3,1]);

K= casos.PS.sym('kp',monomials(x,1),[3,1]);


% SOS mulitplier
s1 = casos.PS.sym('s1',monomials(x,2));
s2 = casos.PS.sym('s2',monomials(x,2));
s3 = casos.PS.sym('s3',monomials(x,0),[3 1]);
s4 = casos.PS.sym('s4',monomials(x,0),[3 1]);
s5 = casos.PS.sym('s5',monomials(x,2));

% fixed level for CBF
b = 0.9;

% options for sequential sos
opts = struct('sossol','mosek');

opts.verbose       = 1;
opts.max_iter      = 100;


tau = ( nabla(V,x)*subs(f,u,K)    + K'*R*K + (inv(Dx)*x)'*Q*(inv(Dx)*x)); 

cost = dot(g-W,g-W);
sos = struct('x', [W;V;K;s1;s2;s3; s4;s5],...  % decision variables
              'f', cost ,...                       % cost function
              'p',[]);                             % parameter

% K-function parameter for CBF, i.e., gamma(W(x)) = gamma*W(x)
gamma = 0.0001; % heuristically found

% constraints
sos.('g') = [
             s1
             s2;
             s3;
             s4;
             s5;
             % V - 1e-6*(x'*x);                                    % CLF positivity              
             s1*(W-b) - g;                                       % State constraints
             % s2*(W-b)  -  nabla(W,x)*subs(f,u,K)- gamma*(W-b);   % CBF
             % s3*(W-b)  + K-umin;                                 % control constraints
             % s4*(W-b)  + umax-K;
             % s5*(W-b) -  tau ;                                   % CLF dissipation
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% solver setup
S  = casos.nlsossol('S','filter-linesearch',sos,opts);

% initial guess for sequential
x0 = casos.PD([ g; ...
                Wval;
                -K0*x;
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
Wsol_re = subs(sol.x(1),x,Dx*x) - full(casos.PD(bsol)); % write CBF as sublevel set 
Vsol_re = subs(sol.x(2),x,Dx*x);                        % CLF

% Control law (parameter found on scaled states)
% K       = -sol.x(3:5).*x(1:3)-sol.x(6:8).*x(4:6);
% Ksol_re = subs(K,x,Dx*x);

%% plotting
import casos.toolboxes.sosopt.*

% slice for rates, i.e. sigma = [0,0,0]' satellite is aligned with inertial
% frame
figure(1)
clf
pcontour(subs(Wsol_re,x(3:end),zeros(4,1)),0,[-x1_max x1_max -x2_max -x2_max]*1.01,'g')
hold on 
pcontour(subs(g0,x(3:end),zeros(4,1)),0,[-x1_max x1_max -x2_max -x2_max]*1.01,'k--')
legend('Terminal Set','Safe Set')

% 3D slice for Modified rodrigues parameter for rest-to-rest
figure(2)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour3(subs(Wsol_re,x(1:3),zeros(3,1)),0,[-1 1 -1 1 -1 1])
hold on 
legend('Terminal Set')


%% verification 

% ------------------------------------------------------------------------
% Comment: Altough we have the guarantee all SOS constraints are met if 
% feasible, we can run a simple sampling approach to check if the 
% sufficient conditions are met.
% ------------------------------------------------------------------------

% unscaled dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

f =  [-J\cpm(x(1:3))*J*x(1:3) + J\u;
      1/4*B(x(4:6))*x(1:3)];

% continous-time penalty, invariant set and control law as functions
penalty     = to_function(nabla(Vsol_re,x)*subs(f,u,Ksol_re) + Ksol_re'*R*Ksol_re + x'*Q*x) ;  % CLF/optimal value function
safety_fun  = to_function(Wsol_re);  %CBF
Kfun        = to_function(Ksol_re);

% generate sample rate with the individual boubds
nSample = 1000000;          % number of samples
samples = zeros(6,nSample); % pre-allocation

a1 = -0.5*pi/180;
b1 =  0.5*pi/180;
samples(1,:) = (b1-a1)*rand(1,nSample)+a1;

a2 = -0.2*pi/180;
b2 =  0.2*pi/180;
samples(2,:) = (b2-a2)*rand(1,nSample)+a2;

a3 = -0.2*pi/180;
b3 =  0.2*pi/180;

samples(3,:) = (b3-a3)*rand(1,nSample)+a3;

a4 = -0.6;
b4 =  0.6;

samples(4:6,:) = (b4-a4)*rand(3,nSample)+a4;


% get all samples that lies within the invariant set, i.e.,  W(x_samp) <= 0
idx = find(full(safety_fun(samples(1,:),samples(2,:),samples(3,:),samples(4,:),samples(5,:),samples(6,:))) <= 0);

% evaluate control law and check if commanded control torques lie in bounds
uval = full(Kfun(samples(1,idx),samples(2,idx),samples(3,idx),samples(4,idx),samples(5,idx),samples(6,idx)));

if any(uval(1,:) > umax(1)) || any(uval(1,:) <  umin(1)) || ...
   any(uval(2,:) > umax(2)) || any(uval(2,:) <  umin(2) )|| ...
   any(uval(3,:) > umax(3)) || any(uval(3,:) <  umin(1)) 
        fprintf('Control Constraints are not met at sampling points!!!!!\n' )
else
    fprintf('Control Constraints are met at sampling points\n' )

end

% evaluate the CLF at sample points that lie in safe set
penalties = full(penalty(samples(1,idx),samples(2,idx),samples(3,idx),samples(4,idx),samples(5,idx),samples(6,idx)));


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
% [coeffGuess,monoGuess] = poly2basis(remove_coeffs(sol.x,1e-6));
% 
% % store monomial basis
% [~,monoGuessV]  = poly2basis(remove_coeffs(sol.x(2),1e-6));
% [~,monoGuessS5] = poly2basis(remove_coeffs(sol.x(end),1e-6));
% 
% save('initGuess_termSet.mat','coeffGuess','monoGuess','monoGuessV','monoGuessS5')

%% save ingredients for MPC
W_fun = to_function(Wsol_re);
V_fun = to_function(Vsol_re);
K_fun = to_function(Ksol_re);

% save for infinitesimal MPC in current folder
save('terminalIngredients.mat','W_fun','V_fun', 'K_fun', ... % polynomials
    'Q','R', ...                                             % weights
    'umin','umax','omegaMax1','omegaMax2','omegaMax3','J')   % parameter

% store also in poly_controlLaw folder
cd ..\poly_controlLaw\

save('terminalIngredients.mat','W_fun','V_fun', 'K_fun', ... % polynomials
    'Q','R', ...                                             % weights
    'umin','umax','omegaMax1','omegaMax2','omegaMax3','J')   % parameter

% come back to current folder
cd ..\inf_MPC\

