%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: 
%
%
%   Needed software: - CasADi 3.6 
%					 - CaSoS
%
%
% ------------------------------------------------------------------------



clc
clear
close all

import casos.toolboxes.sosopt.*

x  = casos.Indeterminates('x',2);
Q = [0.8,1;0,0.5]*1000;
xc1 = [0;0.15];
xc2 = [-0;-0.15];

xc3 = [-0.5;0.5];
xc4 = [0.7;-0.5];
xc5 = [0.5;-0.2];
% rate constraints, keep-out cone and restriction to sublevel set
g = [
    % x(1)^2+3*x(2)^2-1;
     1-(1*x-xc1)'*Q*(1*x-xc1);
     1-(1*x-xc2)'*Q*(1*x-xc2)];
     % 1-(1*x-xc3)'*Q*(1*x-xc3);
     % 1-(1*x-xc4)'*Q*(1*x-xc4);
     % 1-(1*x-xc5)'*Q*(1*x-xc5);
     % ];


% SOS multiplier
if ~isempty(g) 
    s = casos.PS.sym('s',monomials(x,0:2),[length(g) 1],'gram');
else
    s = [];
end

% multiplier for monotonicty growth
s0 = casos.PS.sym('s0',monomials(x,0:2),'gram');

% allowable set function
h = casos.PS.sym('h',monomials(x,2:6));
h_sym = casos.PS.sym('h_sym',sparsity(h));


% initial guess for Coordinate-descent
h_star = x'*eye(2)*10000*x;
b = 0.1;


%% Compute inner-estimate

disp('=========================================================')
disp('Build solver...')
tic

% define SOS feasibility to compute multiplier
sos = struct('x',s, ...
             'g',s*(h_sym-b) - g, ...
             'p',h_sym);

% states + constraint are SOS cones
opts.Kx.sos = length(g); 
opts.Kx.lin = 0; 
opts.Kc.sos = length(g);

% ignore infeasibility
opts.error_on_fail = false;

% solve by relaxation to SDP
S1 = casos.sossol('S1','mosek',sos,opts);

if ~isempty(g)
    s_sym = casos.PS.sym('s_sym',sparsity(s(1)),[length(g) 1]);
else
    s_sym = [];
end

% define SOS feasibility
sos2 = struct('x',[h;s0], ...
              'g',[s_sym*(h-b) - g;
                  s0*(h_sym-b) + b - h], ...	% "growth" constraint
              'p',[s_sym;h_sym]);

% states + constraint are SOS cones
opts.Kx.lin = 1; 
opts.Kx.sos = 1;
opts.Kc.sos = 1+length(g);

% ignore infeasibility
opts.error_on_fail = false;

% solve by relaxation to SDP
S2 = casos.sossol('S2','mosek',sos2,opts);
tbuild = toc;


disp('Finished building solver!')
disp('=========================================================')
disp('Start iteration...')

itermax = 100;

for iter = 1:itermax

   %% solve s-step
   sol1 = S1('p',h_star);
    
   % check solution
    switch (S1.stats.UNIFIED_RETURN_STATUS)
        case 'SOLVER_RET_SUCCESS'
              disp(['s step feasible in ' num2str(iter) '/' num2str(itermax) ] )
     
        otherwise
            disp(['s step infeasible in ' num2str(iter) '/' num2str(itermax) ] )
            break
    end
    
    %% solve h-step
    sol2 = S2('p',[sol1.x;h_star]);
    
    % check solution
    switch (S2.stats.UNIFIED_RETURN_STATUS)
        case 'SOLVER_RET_SUCCESS'
             h_star = sol2.x(1);
          
              disp(['h step feasible in ' num2str(iter) '/' num2str(itermax) ] )
        case {'SOLVER_RET_INFEASIBLE' 'SOLVER_RET_NAN'}
            disp(['h step infeasible in ' num2str(iter) '/' num2str(itermax) ] )
            break
        otherwise
            disp(['h step infeasible in ' num2str(iter) '/' num2str(itermax) ] )
            break
    end
end
tIter = toc-tbuild;

disp('=========================================================')
disp('Finished iteration')
disp(['Build time: ' num2str(tbuild) ' s' ])
disp(['Iteration time: ' num2str(tIter) ' s' ])
disp('_________________________________________________________')
disp(['Total time: ' num2str(tIter+tbuild) ' s' ])

% re-scale/scale input; remove very small term
h_res = cleanpoly(subs(h_star,x,x),1e-5)-b;

%% plotting

import casos.toolboxes.sosopt.*
figure(100)
pcontour(h_res,0,[-1 1 -1 1],'k--')
hold on
for k = 1:length(g)
    if k == 1
        pcontour(g(k),0,[-1 1 -1 1],'g-')
    else
        pcontour(-g(k),0,[-1 1 -1 1],'r-')
    end
end