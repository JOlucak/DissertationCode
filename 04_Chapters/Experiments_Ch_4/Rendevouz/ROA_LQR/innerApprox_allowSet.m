%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Compute an allowable set in six indeterminate (rates and 
%                      Modified Rodrigues Parameter (MRP)) for one keep-ot cone, 
%					   rate constraints and a constraint to bound the length 
%					   of the MRP. A sampling based approach is used in the end to 
%                      check the solution quality on the unit-sphere. 
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
x =casos.PS('x',6,1);



import casos.toolboxes.sosopt.*


n = 2;
g0 = (x(1)^2/1000^2)^(n/2) + (x(2)^2/1000^2)^(n/2) + (x(3)^2/1000^2)^(n/2)+ ...
     (x(4)^2/10^2)^(n/2) + (x(5)^2/10^2)^(n/2) + (x(6)^2/10^2)^(n/2)-1 ;

Dx   = diag([1/(1000),1/(1000),1/(1000),1/10,1/10,1/10]);


% rate constraints, keep-out cone and restriction to sublevel set
g = g0;
     
% rescale
g = subs(g,x,Dx\x);

% SOS multiplier
if ~isempty(g) 
    s = casos.PS.sym('s',monomials(x,0:1),[length(g) 1],'gram');
else
    s = [];
end

% multiplier for monotonicty growth
s0 = casos.PS.sym('s0',monomials(x,0:1),'gram');

% allowable set function
h = casos.PS.sym('h',monomials(x,2));
h_sym = casos.PS.sym('h_sym',sparsity(h));


% initial guess for Coordinate-descent
h_star = x'*eye(6)*10*x;
b = 1;


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
h_res = cleanpoly(subs(h_star,x,Dx*x),1e-10)-b;

%% plotting
import casos.toolboxes.sosopt.*
figure(100)
pcontour(subs(h_res,x(3:6),0),0,[-10 10 -10 10]*100,'k-')
hold on
pcontour(subs(g0,x(3:6),0),0,[-10 10 -10 10]*100,'r--')

