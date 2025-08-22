
clear
close all
clc

% system states
x = casos.PS('x',6,1);
u = casos.PS('u',3,1);

%% Hubble telescope parameter
J = diag([31046;77217;78754]);

% simple bounds
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';


% control constraint; assumption is that the box is inside the full
% pyramid. This is roughly estimated visually.
umin = [-1 -1 -1]'*2;
umax = [ 1  1  1]'*2;

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3)),0.5,.5,.5]);

Dxin = inv(Dx);

%% dynamics
% skew-symmetric matrix
cpm = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

f =  [-J\cpm(x(1:3))*J*x(1:3) + J\u;
      1/4*B(x(4:6))*x(1:3)];            % omega_dot

% trim point
x0    = [0 0 0 0 0 0]';
u0    = [0,0,0]';

A = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

% LQR controller weights
Q = diag([1,1,1,10,10,10]);
R = eye(3);         

[K0,P0] = lqr(full(A),full(B),Q,R);

% scaled initial guess for terminal penalty (Lyapunov linear system)
Vval = (inv(Dx)*x)'*P0*(inv(Dx)*x);

% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);

% state constraint
n = 4;
x_1 = x(1);
x_2 = x(2);
x_3 = x(3);
x_4 = x(4);
x_5 = x(5);
x_6 = x(6);


% re-scale input of state constraints
% gcost = subs(gcost,x,Dx\x); 

g0 =      -0.0206358 + 0.0432121*x_6 + 25.3714*x_1^2 - 32.8037*x_2^2 - 32.8203...
  *x_3^2 + 0.010125*x_4^2 + 0.0331467*x_4*x_5 + 0.010125*x_5^2 + 0.0648211...
  *x_6^2 - 110.007*x_1^2*x_6 - 1153.84*x_2^2*x_6 - 1153.82*x_3^2*x_6 ...
  + 0.012475*x_4^2*x_6 + 0.0585832*x_4*x_5*x_6 + 0.012475*x_5^2*x_6 ...
  + 0.0327014*x_6^3 + 4.21006e+06*x_1^4 - 2.57726e+06*x_1^2*x_2^2 ...
  + 1.68913e+08*x_2^4 - 2.57721e+06*x_1^2*x_3^2 - 1.4034e+07*x_2^2*x_3^2 ...
  + 1.68914e+08*x_3^4 - 37.7648*x_1^2*x_4^2 - 190.765*x_2^2*x_4^2 ...
  - 190.757*x_3^2*x_4^2 + 0.00792347*x_4^4 - 57.4672*x_1^2*x_4*x_5 ...
  - 428.383*x_2^2*x_4*x_5 - 428.381*x_3^2*x_4*x_5 + 0.00435582*x_4^3*x_5 ...
  - 37.7648*x_1^2*x_5^2 - 190.765*x_2^2*x_5^2 - 190.757*x_3^2*x_5^2 ...
  + 0.0271329*x_4^2*x_5^2 + 0.00435582*x_4*x_5^3 + 0.00792348*x_5^4 ...
  - 24.4517*x_1^2*x_6^2 - 287.507*x_2^2*x_6^2 - 287.477*x_3^2*x_6^2 ...
  + 0.0126561*x_4^2*x_6^2 + 0.0236039*x_4*x_5*x_6^2 + 0.0126561*x_5^2 ...
  *x_6^2 + 0.0141843*x_6^4;

% re-scale input of state constraints
g = subs(g0,x,Dx\x); 

%% setup SOS problem

% terminal set (invariant set)
B  = casos.PS.sym('v',monomials(x,1:4));

%terminal penalty
V  = casos.PS.sym('p',monomials(x,2));

% control law(s)
K1  = casos.PS.sym('k',monomials(x,0:2),[3 1]);
for j = 1:3
 K1(j) = casos.PS.sym('k',monomials([x(j)]));
end

K2  = casos.PS.sym('k',monomials(x,0:2),[3 1]);
for j = 1:3
 K2(j) = casos.PS.sym('k',monomials([x(j+3)]));
end

K = K1+K2;

% SOS mulitplier
s1 = casos.PS.sym('s1',monomials(x,4));
s2 = casos.PS.sym('s2',monomials(x,0:4));
s3 = casos.PS.sym('s3',monomials(x,0),[3 1]);
s5 = casos.PS.sym('s5',monomials(x,0),[3 1]);
s8 = casos.PS.sym('s8',monomials(x,2));
s9 = casos.PS.sym('s9',monomials(x,2));

% b = casos.PS.sym('b');

% fixed level set of terminal set
b = 0.02;

% options for sequential sos
opts = struct('sossol','mosek');

opts.verbose       = 1;
opts.max_iter      = 100;


cost = dot(K,K);
sos = struct('x', [V;K1;K2;s1;s3; s5;s8],... % decision variables
              'f', cost,...                  % cost function
              'p',[]);                       % parameter

% Q = eye(6);
% R = eye(3);

L = (inv(Dx)*x)'*Q*inv(Dx)*x + K'*R*K;

% constraints
sos.('g') = [
             s1;
             s3;
             s5;
             s8;
             s1*g -  nabla(V,x)*subs(f,u,K) - L;
             s3*(V-b)  + K-umin; 
             s5*(V-b)  + umax-K;
             % s8*(V-b) - g;
             ];




% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% solver setup
solver_terminalIngredients  = casos.nlsossol('S','filter-linesearch',sos,opts);

% initial guess for sequential
x0 = casos.PD([ Vval;  ...
                x'*x
                -eye(3)*x(1:3); ...         
                -eye(3)*x(4:6); ...
                ones(3,1)*(x'*x);
                ones(3,1)*(x'*x);x'*x]);

% solve
sol = solver_terminalIngredients('x0',x0);

bsol = b; %sol.x(end);

% re-scale invariant set, terminal penalty and local control law
Vsol_re = subs(sol.x(1),x,Dx*x) - full(casos.PD(bsol));
Ksol_re = subs(sol.x(3:5),x,Dx*x) + subs(sol.x(6:8),x,Dx*x);


%% plotting
import casos.toolboxes.sosopt.*

% plot in grad instead of rad; for pcontour the input is given in deg so we  scale the input

% slice for rates
figure(1)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour(subs(subs(Vsol_re,x(3:end),zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g0,x(3:end),zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Terminal Set','Safe Set')

% 3D slice for Modified rodrigues parameter
figure(2)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour3(subs(Vsol_re,x(1:3),zeros(3,1)),0,[-2 2 -2 2 -2 2]*2,'g')
hold on 
legend('Terminal Set')


%% verification

% unscaled dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

f =  [-J\cpm(x(1:3))*J*x(1:3) + J\u;
      1/4*B(x(4:6))*x(1:3)];

% continous-time penalty, invariant set and control law as functions
penalty     = to_function(nabla(Psol_re,x)*subs(f,u,Ksol_re)+x'*Q*x+Ksol_re'*R*Ksol_re) ;
safety_fun  = to_function(Vsol_re);
Kfun        = to_function(Ksol_re);

% generate sample rate with the individual boubds
nSample = 500000;           % number of samples
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

a4 = -1;
b4 =  1;

samples(4:6,:) = (b4-a4)*rand(3,nSample)+a4;


% get all samples that lies within the invariant set, i.e.,  V(x_samp) <= 0
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
