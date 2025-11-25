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

nx = 6;
nu = 3;
nw = 3;

% system states
x = casos.PS('x',nx,1);
% controls
u = casos.PS('u',nu,1);
% input disturbance
w = casos.PS('w',nw,1);
% for K_inf. functions, i.e., scalar real even univariate polynomials
s = casos.Indeterminates('s',1);

%% Hubble parameter
J = diag([31;77;78]);

% simple bounds
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';


% control constraint; assumption is that the box is inside the full
% pyramid. This is roughly estimated visually.
umin = [-1 -1 -1]'*0.012;
umax = [ 1  1  1]'*0.012;

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3)),1,1,1]);
Dxin = inv(Dx);

wmax = 0.1*umax(1);

%% dynamics
% skew-symmetric matrix
skew = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*skew(sigma)+ 2*sigma*sigma';

f0 =  [-J\skew(x(1:3))*J*x(1:3) + J\u ; % omega_dot
        1/4*B(x(4:6))*x(1:3)] ;         % sigma_dot

% full i.e. add disturbance torque/angular acceleration
f = f0 + [J\w;
          zeros(3,1)];

fs = Dx*subs(f,x,inv(Dx)*x);

% trim point/equilibrium point
x0    = [0 0 0 0 0 0]';
u0    = [0 0 0 ]';

A0 = full(casos.PD(subs(nabla(f0,x),[x;u],[x0;u0])));
B0 = full(casos.PD(subs(nabla(f0,u),[x;u],[x0;u0])));

% cost function weights (weights can be adjusted individually)
Q = diag([1, 1, 1, 1, 1, 1]);
R = diag([1, 1, 1])*100;

% generate an initial guess for CLF
[K0,P0] = lqr(full(A0),full(B0),Q,R);

% scaled initial guess for CLF
Vval = x'*P0*x;

% state constraint 
n = 2;
g = (x(1)^2/omegaMax1^2)^(n/2)+(x(2)^2/omegaMax2^2)^(n/2)+(x(3)^2/omegaMax3^2)^(n/2) ...
    + (x(4)^2/0.57^2)^(n/2) + (x(5)^2/0.57^2)^(n/2) + (x(6)^2/0.57^2)^(n/2) - 1;

g = subs(g,x,Dx\x);

%% setup SOS problem
useInitguess  = 0;

if ~useInitguess 
% CBF
h  = casos.PS.sym('h',monomials(x,2));

% CLF
V  =  casos.PS.sym('v',monomials(x,2)); 

% Control law 
K = casos.PS.sym('k',monomials(x,1),[nu 1]);

% ------------------ SOS multiplier ------------------
s1 = casos.PS.sym('s1',monomials(x,0)); % state constraint

s2  = casos.PS.sym('s2',monomials([x;w],2)); % CBF
s2w = casos.PS.sym('s1',monomials([x;w],2));

% control constraints
s3 = casos.PS.sym('s3',monomials(x,2),[length(u) 1]);
s4 = casos.PS.sym('s4',monomials(x,2),[length(u) 1]);

% CLF
s5  = casos.PS.sym('s5',monomials([x;w],2));
s5w = casos.PS.sym('s5w',monomials([x;w],2));

else

load initGuess_termSet.mat
% CBF
h  = casos.PS.sym('h',monoGuessh);

% CLF
V  = casos.PS.sym('v',monomials(x,2)); 

% Control law 
K = casos.PS.sym('k',monoGuessK);


% ------------------ SOS multiplier ------------------
s1 = casos.PS.sym('s1',monoGuessS1); % state constraint

s2  = casos.PS.sym('s2',monoGuessS2); % CBF
s2w = casos.PS.sym('s1',monoGuessS2w);

% control constraints
s3 = casos.PS.sym('s3',monoGuessS3);
s4 = casos.PS.sym('s4',monoGuessS4);

% CLF
s5  = casos.PS.sym('s5',monoGuessS5);
s5w = casos.PS.sym('s5w',monoGuessS5w);

end

% Kinf functions (see Paper Ichihara)
a      = casos.PS.sym('ca',monomials(s^4));
a_ubar = casos.PS.sym('cau',monomials(s^4));
a_bar  = casos.PS.sym('cao',monomials([s^2 s^4]));
sigma  = casos.PS.sym('si',monomials([s^2 s^4]));

% -------------------------------------------------------------------------
% Work around for norm to setup the K-inf function manually
% -------------------------------------------------------------------------

[c_a_ubar,~]  = poly2basis(a_ubar); % helper to get "norm"
c_a_ubar      = casos.PS(c_a_ubar); % casos complaint because coeffs. are SX

[c_a_bar,~]  = poly2basis(a_bar); % helper to get "norm"
c_a_bar      = casos.PS(c_a_bar); % casos complaint because coeffs. are SX


[c_sigma ,~] = poly2basis(sigma ); % helper to get "norm"
c_sigma      = casos.PS(c_sigma);  % casos complaint because coeffs. are SX

[c_a,~]      = poly2basis(a);      % helper to get "norm"
c_a          = casos.PS(c_a);      % casos complaint because coeffs. are SX

% fixed level for CBF
b = 0.9;

% options for sequential sos
opts = struct('sossol','mosek');

opts.verbose       = 1;
opts.max_iter      = 100;

tau = ( nabla(V,x)*subs(fs,u,K)    + K'*R*K + x'*inv(Dx)'*Q*(inv(Dx)*x) - c_sigma(1)*(w'*w) - c_sigma(2)*(w'*w)^2 + c_a*(x'*x)^2 ); 

cost = dot(g-(h-b), g-(h-b));

sos = struct('x', [h;V;K;s1;s2w;s2;s3; s4;s5;s5w;a_bar;a_ubar;a;sigma],...  % decision variables
              'f', cost ,...                                                % cost function
              'p',[]);                                                      % parameter

% K-function parameter for CBF, i.e., gamma(W(x)) = gamma*W(x)
gamma = 1;

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
              multiplier_constraints;                                                      % SOS multiplier
              CLF_constraints                                                              % CLF constraints
              s1*(h-b)  - g;                                                               % State constraint(s)
              s2*(h-b)  -  nabla(h,x)*subs(fs,u,K) - gamma*(h-b) + s2w*(w'*w - wmax^2);  % CBF
              s3*(h-b)  + K-umin;                                                              % control constraints
              s4*(h-b)  + umax-K ;
              K_inf_constraints                                                            % K_inf
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% turn off Newton
opts.sossol_options.newton_solver = [];
opts.tolerance_con  = 1e-3;
opts.tolerance_opt  = 1e-3;

% solver setup
S  = casos.nlsossol('S','sequential',sos,opts);

% initial guess for sequential
% if ~useInitguess 
x0 = casos.PD([ g; ...
                Vval;
                1
                x'*x;
                x'*x;
                x'*x;
                x'*x;
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
% else
% % % load initial guess
load initGuess_termSet.mat
x0 = casos.PD(monoGuess,coeffGuess);
% end
% solve
sol = S('x0',x0);

% bsol = full(sol.x(end));
bsol = b;

% re-scale invariant set, terminal penalty and local control lah
hsol_re = subs(sol.x(1),x,Dx*x) - full(casos.PD(bsol)); % CBF as sublevel set 
Vsol_re = subs(sol.x(2),x,Dx*x);                        % CLF

Ksol_re = subs(sol.x(3:5),x,Dx*x);

%% plotting
g_re    = subs(g,x,Dx*x);
import casos.toolboxes.sosopt.*
% plot in grad instead of rad; for pcontour the input is given in deg so we
% scale the input

% slice for rates, i.e. sigma = [0,0,0]' satellite is aligned with inertial
% frame
figure(1)
subplot(311)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);

pcontour(subs(subs(hsol_re,x(3:end),zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(3:end),zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Terminal Set','Safe Set')
subplot(312)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
pcontour(subs(subs(hsol_re,[x(2);x(4:end)],zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,[x(2);x(4:end)],zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Terminal Set','Safe Set')

subplot(313)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
pcontour(subs(subs(hsol_re,[x(1);x(4:end)],zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,[x(1);x(4:end)],zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Terminal Set','Safe Set')

% 3D slice for Modified rodrigues parameter for rest-to-rest
figure(2)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour3(subs(hsol_re,x(1:3),zeros(3,1)),0,[-1 1 -1 1 -1 1])
hold on 
legend('Terminal Set')



conVio =  full(S.stats.single_iterations{end}.constraint_violation);


%% store initial guess
[coeffGuess,monoGuess] = poly2basis(remove_coeffs(sol.x,1e-6));

% store monomial basis
[~,monoGuessh]    = poly2basis(remove_coeffs(sol.x(1),1e-6));
[~,monoGuessV]    = poly2basis(remove_coeffs(sol.x(2),1e-6));
[~,monoGuessK]    = poly2basis(remove_coeffs(sol.x(3:5),1e-6));
[~,monoGuessS1]   = poly2basis(remove_coeffs(sol.x(6),1e-6));
[~,monoGuessS2]   = poly2basis(remove_coeffs(sol.x(7),1e-6));
[~,monoGuessS2w]  = poly2basis(remove_coeffs(sol.x(8),1e-6));
[~,monoGuessS3]   = poly2basis(remove_coeffs(sol.x(9:11),1e-6));
[~,monoGuessS4]   = poly2basis(remove_coeffs(sol.x(12:14),1e-6));
[~,monoGuessS5]   = poly2basis(remove_coeffs(sol.x(15),1e-6));
[~,monoGuessS5w]  = poly2basis(remove_coeffs(sol.x(16),1e-6));

save('initGuess_termSet.mat','coeffGuess','monoGuess', ...
    'monoGuessh','monoGuessK','monoGuessV','monoGuessS5',...
    'monoGuessS1','monoGuessS2','monoGuessS3','monoGuessS4',...
    'monoGuessS2w','monoGuessS5','monoGuessS5w')

%% save ingredients for MPC
h_fun = to_function(hsol_re);
V_fun = to_function(Vsol_re);
K_fun = to_function(Ksol_re);
g_fun =  to_function(g);


a_bar_fun = to_function(sol.x(end-3));
a_ubar_fun = to_function(sol.x(end-2));
a_fun = to_function(sol.x(end-1));
sigma_fun = to_function(sol.x(end));

% dynamics
xdot_nom = to_function(f0);
x_dot    = to_function(f);

% save for infinitesimal MPC in current folder
save('terminalIngredients.mat','h_fun','V_fun', 'K_fun','gamma', ... % polynomials
    'Q','R','g_fun','a_bar_fun','a_ubar_fun','a_fun','sigma_fun', ...                                             % weights
    'x_dot','xdot_nom','nx','nu','nw',...
    'umin','umax','x_low','x_up')                                           % parameter
