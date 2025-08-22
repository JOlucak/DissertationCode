
clear
close all
clc

% system states
x = casos.PS('x',3,1);
u = casos.PS('u',3,1);

%% Hubble parameter
J = diag([31046;77217;78754]);

% simple bounds
omegaMax1 = 2*pi/180;
omegaMax2 = 1*pi/180;
omegaMax3 = 1*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';


% control constraint; assumption is that the box is inside the full
% pyramid. This is roughly estimated visually.
umin = [-1 -1 -1]'*1;
umax = [ 1  1  1]'*1;

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3))]);
Dxin = inv(Dx);

%% dynamics
% cross-product matrix
cpm = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% dynamics
f =  -J\cpm(x(1:3))*J*x(1:3) + J\u; 

% trim point
x0    = [0 0 0]';
u0    = [0,0,0]';

A = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

% LQR controller weights
Q = eye(3)*100;
R = eye(3)*0.001;         

[K0,P] = lqr(full(A),full(B),Q,R);

% %scale initial lyapunov
Vval = x'*inv(Dx)'*P*inv(Dx)*x;


% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);

% state constraint
n = 4;
g = (x(1)^2/omegaMax1^2)^(n/2)+(x(2)^2/omegaMax2^2)^(n/2)+(x(3)^2/omegaMax3^2)^(n/2)-1;
g = subs(g,x,Dx\x);


V  = casos.PS.sym('v',monomials( [x(1)^4 x(1)^2*x(2)^2 x(2)^4 x(1)^2*x(3)^2 x(2)^2*x(3)^2 x(3)^4]));
K  = casos.PS.sym('k',monomials(x(1),1),[3 1]);
for j = 1:3
    K(j) = casos.PS.sym('k',monomials(x(j),1));
end
s1 = casos.PS.sym('s1',monomials( [x(1)^4 x(1)^2*x(2)^2 x(2)^4 x(1)^2*x(3)^2 x(2)^2*x(3)^2 x(3)^4]));

s3 = casos.PS.sym('s3',monomials([x(1)^2 x(2)^2 x(3)^2]),[3 1]);
s5 = casos.PS.sym('s5',monomials([x(1)^2 x(2)^2 x(3)^2]),[3 1]);
s8= casos.PS.sym('s7',monomials([x(1)^2 x(2)^2 x(3)^2]) );
s9 = casos.PS.sym('s9',monomials([x(1)^2 x(2)^2 x(3)^2]) );


% options
opts = struct('sossol','mosek');
opts.verbose       = 1;
opts.max_iter      = 100;


b = 1;
sos = struct('x',[V; K; s1; s3; s5;s8],...
              'f',dot(g-(V),g-(V)), ... 
              'p',[]);


% constraints
sos.('g') = [
             s3;
             s5;
             s8;
             s1*g  -  nabla(V,x)*subs(f,u,K) ;
             s3*(g)  + K - umin ; 
             s5*(g)  + umax - K ;
             s8*(V-b) - g;
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% build sequential solver
buildTime_in = tic;
    solver_terminalSet  = casos.nlsossol('S','filter-linesearch',sos,opts);
buildtime = toc(buildTime_in);

sol = solver_terminalSet('x0',casos.PD([ g;  ...
                                 -K0*x; ...         % unscaled initial guess
                                 x'*x; ...
                                 x'*x; ...
                                 ones(3,1)*(x'*x);
                                 ones(3,1)*(x'*x)]));

Vsol    = sol.x(1); 
Ksol    = sol.x(2:4);
s1sol   = sol.x(5);
s3sol   = sol.x(6:8); 
s5sol   = sol.x(9:11);   
s8sol   = sol.x(12);
bsol    = b;


%% estimate terminal penalty for fixed control law
% constraints
opts = [];
sos = [];
P  = casos.PS.sym('p',monomials(x,2));
s2 = casos.PS.sym('s2',monomials(x,1),'gram');

sos.x = [P;s2];
sos.g = s2*g  -  (nabla(P,x)*subs(f,u,Ksol) + (inv(Dx)*x)'*Q*(inv(Dx)*x) + Ksol'*R*Ksol);


% states + constraint are linear/SOS cones
opts.Kx = struct('lin', 1, 'sos',1);
opts.Kc = struct('sos', length(sos.g));

% build sequential solver
solver_oneStepReach  = casos.sossol('S','mosek',sos,opts);


sol = solver_oneStepReach();

Psol = sol.x(1);



%% plotting
import casos.toolboxes.sosopt.*

% re-scale
Vsol_re = subs(Vsol,x,Dx*x)-full(casos.PD(bsol));
Psol_re = subs(Psol,x,Dx*x);
g_re    = subs(g,x,Dx*x);
Ksol_re = subs(Ksol,x,Dx*x);
% plot in grad instead of rad; for pcontour the input is given in deg so we
% scale the input
figure(1)
deg2rad = diag([pi/180,pi/180,pi/180]);
clf
pcontour(subs(subs(Vsol_re,x(3),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(3),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Invariant Set','Rate Bounds')


figure(2)
deg2rad = diag([pi/180,pi/180,pi/180]);
clf
pcontour(subs(subs(Vsol_re,x(2),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(2),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Invariant Set','Rate Bounds')

figure(3)
deg2rad = diag([pi/180,pi/180,pi/180]);
clf
pcontour(subs(subs(Vsol_re,x(1),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(1),0),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Invariant Set','Rate Bounds')


%% Verification
f =  -J\cpm(x(1:3))*J*x(1:3) + J\u;

safety_fun = to_function(Vsol_re);

% continous-time penalty
penalty     = to_function(nabla(Psol_re,x)*subs(f,u,Ksol_re)+x'*Q*x+Ksol_re'*R*Ksol_re) ;

% generate sample rate with the individual bounds
nSample = 100000;
samples = zeros(3,nSample);
a1 = -2*pi/180;
b1 =  2*pi/180;
samples(1,:) = (b1-a1)*rand(1,nSample)+a1;
a2 = -1*pi/180;
b2 =  1*pi/180;
samples(2,:) = (b2-a2)*rand(1,nSample)+a2;
a3 = -1*pi/180;
b3 =  1*pi/180;
samples(3,:) = (b3-a3)*rand(1,nSample)+a3;

% get all samples that lies within the invariant set, i.e.,  V(x_samp) <= 0
idx = find(full(safety_fun(samples(1,:),samples(2,:),samples(3,:))) <= 0);

% evaluate the continous-time penalty at these sample points and get the
% maximum value
max_penalty = max(full(penalty(samples(1,idx),samples(2,idx),samples(3,idx))));


if max_penalty > 0 
    fprintf('Something went wrong. The maximum penalty is %d\n',max_penalty )
else
    fprintf('Everything seems fine! The maximum penalty is %d\n',max_penalty )
end

