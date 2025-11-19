% One-step controllable set calculation
clear
close all
clear

import casos.toolboxes.sosopt.*

x = casos.Indeterminates('x',2);
u = casos.Indeterminates('u',1); 

Ts = 10e-2; % sampling time

% system dynamics in continuous-time
f = [x(2);
        (1-x(1)^2)*x(2)-x(1)];

gx = [0;1];

% open system as Euler-forward
f_disc = x+Ts*(f+gx*u);

Ad = full(subs(nabla(f_disc,x),[x;u],[zeros(2,1);0]));
Bd = full(subs(nabla(f_disc,u),[x;u],[zeros(2,1);0]));

%% Terminal Penalty & Constraints
% terminal LQR
[k0,P] = dlqr(Ad, Bd, eye(2), 2.5);

% terminal set constraint
p0 = x'*P*x - 1;

ulim = [-1;1];

g = 1.78*x(1)^2 + 4*x(2)^2 - 1;


% decision variables
V  = casos.PS.sym('V',  monomials(x,0:4),[2 1]);
L  = casos.PS.sym('l',  monomials(x,0:2));
s0 = casos.PS.sym('s0', monomials(x,0:2),[1 1]);
s1 = casos.PS.sym('s1', monomials(x,0),[1 1]);
s11 = casos.PS.sym('s11', monomials(x,0),[1 1]);
s2 = casos.PS.sym('s2', monomials(x,0:2),[1 1]);
s3 = casos.PS.sym('s3', monomials(x,0:2),[1 1]);
s4 = casos.PS.sym('s4', monomials(x,0:2),[1 1]);
K  = casos.PS.sym('k',  monomials(x,1)  ,[1 1]);

% options
opts = struct('sossol','mosek');
opts.verbose = 1;


sos = struct('x',[V; K; s0; s1;s11; s2; s3; s4],...
              'f',dot(g-V(1), g-V(1)), ...
              'p',L);
b = 0.1;
% constraints
sos.('g') = [s0;
             s1; 
             s2;
             % s3; 
             s4;
             s0*(V(1)-b) -  (subs(V(2),x,subs(f_disc,x,K)) - V(1)) ; 
             s1*(V(1)-b) + (K - ulim(1));
             s11*(V(1)-b) + (ulim(2)- K);
             s2*(V(2)-b) - L;
             % s3*(V(2)-b) - (V(1))+b
             s4*(V(1)-b) - g
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));

% build sequential solver
buildTime_in = tic;
    solver_oneStepReach  = casos.nlsossol('S','sequential',sos,opts);
buildtime = toc(buildTime_in);


disp(['Solver buildtime: ' num2str(buildtime), ' s'])



timeHor = 1;
N      = timeHor/Ts;

for k = 1:N

    if k == 1 
        % user initial guess and "actual" terminal set
        x0 = casos.PD([ p0;  ...
                        p0; ...
                        -k0*x; ...
                        x'*x;
                        x'*x;
                        x'*x;
                        x'*x;
                        x'*x;
                        x'*x]);

        sol = solver_oneStepReach('x0',x0,'p',casos.PD(p0)); 
    else
    
        % we use previous solution as an initia guess and "new" terminal
        % set
        sol = solver_oneStepReach('x0',sol.x,'p',casos.PD(l0)); 
    end

      switch (solver_oneStepReach.stats.solutionStatus)
        case 'Optimal solution'
            % set current reachable set as terminal set
            l0 = casos.PD(sol.x(1))-b;
            l0 = l0(1);
          case 'Feasible Solution'
            % set current reachable set as terminal set
            l0 = casos.PD(sol.x(1))-b;
            l0 = l0(1);
          otherwise
             fprintf('Failed in step %g.\n', k)
          break
      end

end



%% plotting

% descale solution 
V0 = subs(sol.x(1),x,x)-b;
V1 = subs(sol.x(2),x,x)-b;

% p0 = subs(p0,x,x);
g0 = subs(g,x,x);


import casos.toolboxes.sosopt.*
prob.domain = [-1 1 -1 1]*0.7;
figure(1)
pcontour(p0,0,prob.domain,'b')
hold on
pcontour(g0,0,prob.domain,'k')
pcontour(V0,0,prob.domain,'g')
pcontour(V1,0,prob.domain,'r')
