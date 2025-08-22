
clear
close all
clc

% system states
x = casos.PS('x',3,1);
u = casos.PS('u',3,1);

%% Hubble parameter
J = diag([1;1;1]);

% simple bounds
omegaMax1 = 1*pi/180;
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
Q = eye(3);
R = eye(3)*0.1;         

[K0,P] = lqr(full(A),full(B),Q,R);

% %scale initial lyapunov
Vval = x'*inv(Dx)'*P*inv(Dx)*x;


% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);

% state constraint
n = 4;
g0 = (x(1)^2/omegaMax1^2)^(n/2)+(x(2)^2/omegaMax2^2)^(n/2)+(x(3)^2/omegaMax3^2)^(n/2)-1;
g = subs(g0,x,Dx\x);


V  = casos.PS.sym('v',monomials( [x(1)^4 x(1)^2*x(2)^2 x(2)^4 x(1)^2*x(3)^2 x(2)^2*x(3)^2 x(3)^4]));

 % % [x(1)^4 x(1)^2*x(2)^2 x(2)^4 x(1)^2*x(3)^2 x(2)^2*x(3)^2 x(3)^4]

% P  = casos.PS.sym('p',monomials(x,2));
P  = casos.PS.sym('p',monomials([x(1)^2 x(2)^2 x(3)^2]));
K  = casos.PS.sym('k',monomials(x(1),1),[3 1]);
for j = 1:3
    K(j) = casos.PS.sym(['k' num2str(j)],monomials(x(j),1));
end
s1 = casos.PS.sym('s1',monomials( [x(1)^4 x(1)^2*x(2)^2 x(2)^4 x(1)^2*x(3)^2 x(2)^2*x(3)^2 x(3)^4]));
s2 = casos.PS.sym('s2',monomials([x(1)^2 x(2)^2 x(3)^2]));
s3 = casos.PS.sym('s3',monomials([x(1)^2 x(2)^2 x(3)^2]),[3 1]);
s5 = casos.PS.sym('s5',monomials([x(1)^2 x(2)^2 x(3)^2]),[3 1]);
% s3 = casos.PS.sym('s3',monomials(x,0:2),[3 1]);
% s5 = casos.PS.sym('s5',monomials(x,0:2),[3 1]);
% s8 = casos.PS.sym('s8',monomials(x,2));
% s9 = casos.PS.sym('s9',monomials(x,2));
s8= casos.PS.sym('s7',monomials([x(1)^2 x(2)^2 x(3)^2]) );
s9 = casos.PS.sym('s9',monomials([x(1)^2 x(2)^2 x(3)^2]) );
b = casos.PS.sym('b');

% options
opts = struct('sossol','mosek');

% adjust optimality thresholds
% opts.conVioTol     = 6e-8;
% opts.optTol        = 1e-1;
% opts.error_on_fail = 0;
opts.verbose       = 1;
opts.max_iter      = 100;


b = 1;
sos = struct('x',[V; P; K; s1; s2; s3; s5;s8],...
              'f',dot(g-V,g-V), ... 
              'p',[]);


% constraints; with this scaling, we have the real control inputs in terms
% of x instead of the scaled states
sos.('g') = [
             s1;
             s2;
             s3;
             s5;
             s8;
             s1*g  -  nabla(V,x)*subs(f,u,subs(K,x,Dx\x)) ;
             s2*g  -  (nabla(P,x)*subs(f,u,subs(K,x,Dx\x)) + (inv(Dx)*x)'*Q*(inv(Dx)*x) + subs(K,x,Dx\x)'*R*subs(K,x,Dx\x));
             s3*g  + subs(K,x,Dx\x) - umin ; 
             s5*g  + umax - subs(K,x,Dx\x) ;
             s8*(V-b) - g;
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% build sequential solver
buildTime_in = tic;
    solver_oneStepReach  = casos.nlsossol('S','filter-linesearch',sos,opts);
buildtime = toc(buildTime_in);

sol = solver_oneStepReach('x0',casos.PD([ g;  ...
                                 Vval;
                                 -K0*x; ...         % scaled initial guess
                                 x'*x; ...
                                 x'*x; ...
                                 ones(3,1)*(x'*x);
                                 ones(3,1)*(x'*x);
                                 x'*x]));


%% check feasibility
Vsol    = sol.x(1); 
Psol    = sol.x(2); 
Ksol    = sol.x(3:5);
s1sol   = sol.x(6);
s2sol   = sol.x(7); 
s3sol   = sol.x(8:10); 
s5sol   = sol.x(11:13);   
s8sol   = sol.x(14);
% s9sol   = sol.x(15);
% currently set constant
bsol    = b;

gval = [    
            % s1sol;
            % s2sol;
            % s3sol;
            % s5sol;
            % s8sol;
            % s1sol*(Vsol-bsol)  -  nabla(Vsol,x)*(subs(f,u,Ksol));
            % s2sol*(Vsol-bsol)  -  (nabla(Psol,x)*subs(f,u,Ksol) + (inv(Dx)*x)'*Q*(inv(Dx)*x) +  Ksol'*R*Ksol);
            % s3sol*(Vsol-bsol)  + Ksol-umin; 
            % s5sol*(Vsol-bsol)  + umax-Ksol;
            % s8sol*(Vsol-bsol) - g;
            % % s9sol*(Vsol-bsol) + Psol
         ];
% 
% for j = 1:length(gval)
%     feas = isSOS(gval(j));
% 
%     if ~feas
%         disp(['Constraint ' num2str(j) ' is violated'])
%     end
% end

%% plotting
import casos.toolboxes.sosopt.*

% re-scale
Vsol_re = subs(Vsol,x,Dx*x)-full(casos.PD(bsol))
Psol_re = subs(Psol,x,Dx*x)
g_re    = subs(g0,x,x);
Ksol_re = sol.x(3:5)
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

