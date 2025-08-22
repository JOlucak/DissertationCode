close all
clc
clear

% load some conversion function for attitute parameterization
addpath("helperFunc\")


% load infinitesimal MPC data
load infiniteMPC_comparison.mat

% load full horizon MPC data
load fullHor_MPC_comparison.mat


% put both methods into one cell
x_sol_all = {x_sol_vec_infMPC,x_sol_vec_fullMPC};
u_sol_all = {u_sol_vec_infMPC,u_sol_vec_fullMPC};
iter_all  = {iter_conv_inf,iter_conv_full};


numRuns = 2;

t = linspace(0, simTime, simTime/simStepSize);


%% convergence time
convergenceTime_infinite = t(iter_conv_inf)
convergenceTime_full     = t(iter_conv_full)

rel_error_conv = (t(iter_conv_inf)-t(iter_conv_full))/t(iter_conv_full)*100


%% evaluate performance

% We have to bring back MRP

% Define L as an anonymous function
L = @(x, u)  x'*Q_infinite*x + u' * R_infinite * u;

% Preallocate cost array for each time step
cost_per_step = zeros(1, iter_conv_inf);

% Compute L for each time step individually
for k = 1:iter_conv_inf
    cost_per_step(k) = L(x_sol_vec_infMPC(:,k), u_sol_vec_infMPC(:,k));
end

% Compute total cost using trapezoidal integration
stageCost_inf = trapz(0:0.1:t(iter_conv_inf), cost_per_step)



% Preallocate cost array for each time step
cost_per_step = zeros(1, iter_conv_full);

% Compute L for each time step individually
for k = 1:iter_conv_full-1
    cost_per_step(k) = L(x_sol_vec_fullMPC(:,k), u_sol_vec_fullMPC(:,k));
end

% Compute total cost using trapezoidal integration
tvec = linspace(0,t(iter_conv_full),length(cost_per_step));

stageCost_full = trapz(tvec, cost_per_step)

(stageCost_inf-stageCost_full)/stageCost_full*100

%% Plotting

colors = lines(numRuns);                        % Get a colormap for different runs
Euler_names = {'\phi','\theta','\psi'};
% re-scale rate constraints (real physical constraints)
x_low =  [-Omega_bounds(1)*180/pi -Omega_bounds(2)*180/pi -Omega_bounds(3)*180/pi]';
x_up  =  -x_low;

t_short = linspace(0, simTime, (simTime/simStepSize)-1);

% set up with torques in miliNetwonmeter
ulow = u_low';
uup  = u_up';

% % Plot Rates in Degree/second
figure('Name', 'Rates');
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t(1:iter_all{j}), x_sol_all{j}(i,1:iter_all{j}) * 180/pi, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\omega_%c [°/s]', 'x' + (i-1)));
    grid on;


    % plot gray shadded area and dashed gray line
    xLimits = [0, t(max([iter_all{:}])) ]; 
    yDashed = x_low(i); 
    plot([0 t(max([iter_all{:}]))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    miny =  x_low(i)+0.5* x_low(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max([iter_all{:}]))]; 
    yDashed =  x_up(i);
    plot([0 t(max([iter_all{:}]))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    maxy = x_up(i)+0.5*x_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max([iter_all{:}])) miny maxy])


    if i == 1
        h = zeros(2, 1);
        h(1) = plot(NaN,NaN,'-','Color',colors(1,:));
        h(2) = plot(NaN,NaN,'-','Color',colors(2,:));
        legend(h, 'infinitesimal','full')
    end
end


figure('Name', 'Control Torques')
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t(1:iter_all{j}-1), u_sol_all{j}(i,(1:iter_all{j}-1)) * 1000, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\tau_%c [mNm]', 'x' + (i-1)));
    grid on;

    % plot gray shadded area and dashed gray line
    xLimits = [0, t(max(iter_all{j}))]; 
    yDashed = ulow(i); 
    miny =  ulow(i)+0.5* ulow(i);
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max(iter_all{j}))]; 
    yDashed =  uup(i); 
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5]) 
    maxy = uup(i)+0.5*uup(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max(iter_all{j})) miny maxy])


    if i == 1
        h = zeros(2, 1);
        h(1) = plot(NaN,NaN,'-','Color',colors(1,:));
        h(2) = plot(NaN,NaN,'-','Color',colors(2,:));
        legend(h, 'infinitesimal','full')
    end

end


