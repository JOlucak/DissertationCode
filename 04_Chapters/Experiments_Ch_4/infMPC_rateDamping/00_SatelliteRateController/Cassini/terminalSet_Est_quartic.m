
clear
close all
clc

% system states
x = casos.PS('x',3,1);
u = casos.PS('u',3,1);

%% Cassini parameter
J = diag([8970;9230;3830]);

% simple bounds; from slew rate [deg/min] to [rad/s]
omegaMax1 = 0.1*pi/180;
omegaMax2 = 0.1*pi/180;
omegaMax3 = 0.2*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';


% control constraint; assumption is that the box is inside the full
% pyramid. This is roughly estimated visually.
umin = [-1 -1 -1]'*0.2;
umax = [ 1  1  1]'*0.2;


Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3))]);
Dxin = inv(Dx);

%% dynamics
% skew-symmetric matrix
skew = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% dynamics
f =  -J\skew(x(1:3))*J*x(1:3) + J\u; % omega_dot

% trim point
x0    = [0 0 0]';
u0    = [0,0,0]';

A = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

% LQR controller weights
Q = eye(3)*10;
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


V  = casos.PS.sym('v',monomials(x,4));
P  = casos.PS.sym('p',monomials(x,2));
K  = casos.PS.sym('k',monomials(x,0:2),[3 1]);
s1 = casos.PS.sym('s1',monomials(x,0:4));
s2 = casos.PS.sym('s2',monomials(x,2));
s3 = casos.PS.sym('s3',monomials(x,0:2),[3 1]);
s5 = casos.PS.sym('s5',monomials(x,0:2),[3 1]);
s7 = casos.PS.sym('s7',monomials(x,0:2));
s8 = casos.PS.sym('s8',monomials(x,0:4));
s9 = casos.PS.sym('s8',monomials(x,0:2));
b = casos.PS.sym('b');

% options
opts = struct('sossol','mosek');

% adjust optimality thresholds
opts.conVioTol     = 6e-8;
opts.optTol        = 1e-1;
opts.error_on_fail = 0;
opts.verbose       = 1;
opts.max_iter      = 20;


b = 0.9;
sos = struct('x',[V; P; K; s1; s2; s3; s5;s8;s9],...
              'f',dot(g-(V-b),g-(V-b)) , ... 
              'p',[]);


% constraints
sos.('g') = [s1;
             s2;
             s3;
             s5;
             s8;
             s9;
             s1*(V-b)  -  nabla(V,x)*subs(f,u,K);
             s2*(V-b)  -  (nabla(P,x)*subs(f,u,K) + (inv(Dx)*x)'*Q*(inv(Dx)*x) + K'*R*K);
             s3*(V-b)  + K-umin; 
             s5*(V-b)  + umax-K;
             s8*(V-b) - g;
             s9*(V-b) + P
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% build sequential solver
buildTime_in = tic;
    solver_oneStepReach  = casos.nlsossol('S','sequential',sos,opts);
buildtime = toc(buildTime_in);

sol = solver_oneStepReach('x0',[ g;  ...
                                 Vval;
                                 -K0*x; ...         % unscaled initial guess
                                 x'*x; ...
                                 x'*x; ...
                                 ones(3,1)*(x'*x);
                                 ones(3,1)*(x'*x);
                                 x'*x;x'*x]);

sol = solver_oneStepReach('x0',sol.x);

disp(['Solver buildtime: ' num2str(buildtime), ' s'])

%% check feasibility
Vsol    = sol.x(1); 
Psol    = sol.x(2); 
Ksol    = sol.x(3:5);
s1sol   = sol.x(6);
s2sol   = sol.x(7); 
s3sol   = sol.x(8:10); 
s5sol   = sol.x(11:13);   
s8sol   = sol.x(14);
s9sol   = sol.x(15);
% currently set constant
bsol    = b;

gval = [    
            % s1sol;
            % s2sol;
            % s3sol;
            % s5sol;
            % s8sol;
            s1sol*(Vsol-bsol)  -  nabla(Vsol,x)*(subs(f,u,Ksol));
            s2sol*(Vsol-bsol)  -  (nabla(Psol,x)*subs(f,u,Ksol) + (inv(Dx)*x)'*Q*(inv(Dx)*x) +  Ksol'*R*Ksol);
            s3sol*(Vsol-bsol)  + Ksol-umin; 
            s5sol*(Vsol-bsol)  + umax-Ksol;
            s8sol*(Vsol-bsol) - g;
             s9sol*(Vsol-bsol) + Psol
         ];

for j = 1:length(gval)
    feas = isSOS(gval(j));

    if ~feas
        disp(['Constraint ' num2str(j) ' is violated'])
    end
end

%% plotting
import casos.toolboxes.sosopt.*

% re-scale
Vsol_re = subs(Vsol,x,Dx*x)-full(casos.PD(bsol));
Psol_re = subs(Psol,x,Dx*x);
g_re    = subs(g,x,Dx*x);

% plot in grad instead of rad; for pcontour the input is given in deg so we
% scale the input
figure(1)
deg2rad = diag([pi/180,pi/180,pi/180]);
clf
pcontour(subs(subs(Vsol_re,x(3),0),x,deg2rad*x),0,[-omegaMax3 omegaMax3 -omegaMax3 omegaMax3]*180/pi,'g')
hold on 
pcontour(subs(subs(g_re,x(3),0),x,deg2rad*x),0,[-omegaMax3 omegaMax3 -omegaMax3 omegaMax3]*180/pi,'k--')
legend('Invariant Set','Rate Bounds')

