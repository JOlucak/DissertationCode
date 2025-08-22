
import casos.toolboxes.sosopt.*
clc
clear
close all


% system states
x = casos.Indeterminates('x',2,1);


xmax = [3;3];
xmin = -xmax;
box = -(1*x-xmin).*(xmax-1*x); % g1 <= 0;

Q1 = eye(2)*2;
Q2 = Q1;

xc1 = [-2,2]';
xc2 = [2,-2]';

obs1 = -((1*x-xc1)'*Q1*(1*x-xc1)-1);
obs2 = -((1*x-xc2)'*Q2*(1*x-xc2)-1);


g = [box;obs1;obs2];

% g = box;

% SOS multiplier
if ~isempty(g) 
    s = casos.PS.sym('s',monomials(x,0:1),[length(g) 1],'gram');
else
    s = [];
end

% multiplier for monotonicty growth
s0 = casos.PS.sym('s0',monomials(x,0:2),'gram');

% safe set function
maxh = [4;7];

for K = 1:length(maxh)

h = casos.PS.sym('h',monomials(x,0:maxh(K)));
h_sym = casos.PS.sym('h_sym',sparsity(h));


% initial guess for Coordinate-descent
h_star = x'*eye(2)*x-1;

% figure()
% pcontour(h_star,0,[-1 1 -1 1 ])

%% Compute inner-estimate

disp('=========================================================')
disp('Build solver...')
tic

% define SOS feasibility to compute multiplier
sos = struct('x',s, ...
             'g',s*h_sym - g, ...
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
              'g',[s_sym*h - g;
                  s0*h_sym - h ], ...
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

itermax = 80;

for iter = 1:itermax

   %% solve s-step
   sol1 = S1('p',h_star);
    
   % check solution
    switch (S1.stats.UNIFIED_RETURN_STATUS)
        case 'SOLVER_RET_SUCCESS'
              disp(['s step feasible in ' num2str(iter) '/' num2str(itermax) ] )
        case {'SOLVER_RET_INFEASIBLE' 'SOLVER_RET_NAN'} 
            disp(['s step infeasible in ' num2str(iter) '/' num2str(itermax) ] )
             break
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
tIter(K) = toc-tbuild;
ITER(K) = iter;
% disp('=========================================================')
% disp('Finished iteration')
% disp(['Build time: ' num2str(tbuild) ' s' ])
% disp(['Iteration time: ' num2str(tIter) ' s' ])
% disp('_________________________________________________________')
% disp(['Total time: ' num2str(tIter+tbuild) ' s' ])


if K == 1
    color = 'g';
else
    color = 'b';

end

figure(1)
pcontour(h_star,0,[-1 1 -1 1 ]*1.1*max(xmax),color)
hold on
if K ==1
    plotBoxCon([1 2],xmax,xmin)
    [C1,h1]= pcontour(obs1,0,[-1 1 -1 1 ]*1.1*max(xmax),'k');
    set(h1,'Tag','HatchingRegion');
    ax1 = gca;
    hp = findobj(h1,'Tag','HatchingRegion');
    hh = hatchfill2(hp,'cross','HatchAngle',45,'HatchDensity',100);
    
    [C2,h2]= pcontour(obs2,0,[-1 1 -1 1 ]*1.1*max(xmax),'k');
    set(h2,'Tag','HatchingRegion');
    ax2 = gca;
    hp1 = findobj(h2,'Tag','HatchingRegion');
    hh2 = hatchfill2(hp1,'cross','HatchAngle',45,'HatchDensity',100);
end

end
qw{1} = plot(nan, 'g-');
qw{2} = plot(nan, 'b-');
legend([qw{:}], {'deg(h) = 4','deg(h) = 7'}, 'location', 'northoutside')
tIter
cleanfigure()
matlab2tikz('safeSet_aca_ex.tikz','width','\figW','height','\figH');


