clc
clear


results_disc =DiscreteTime;

results_cont =continousTime_MPC;


x_sol_all = [results_disc.x_sol_all_;results_cont.x_sol_all_cont];
u_sol_all = [results_disc.u_sol_all_disc;results_cont.u_sol_all_cont];
tEnd_all = [results_disc.tEnd_all_disc,results_cont.tEnd_all_cont']';

% %% Compute statistics on computation time in miliseconds
% tEnd_all(isnan(tEnd_all)) = [];
% 
% minSolveTime = min(tEnd_all, [], 'all') * 1000;
% maxSolveTime  = max(tEnd_all, [], 'all') * 1000;
% meanSolveTime = mean(tEnd_all, 'all') * 1000;
% 
% fprintf('Minimum solve time: %f ms\n', minSolveTime);
% fprintf('Maximum solve time: %f ms\n', maxSolveTime);
% fprintf('Mean solve time: %f ms\n', meanSolveTime);

%% Plotting
numRuns = 2;
dt0 = 0.1;
simTime     = 1500;
simStepSize = dt0;

t = linspace(0, simTime, simTime/simStepSize);
k = length(t);
colors = lines(numRuns); % Get a colormap for different runs

% bounds for rates (needed later for plotting)
omegaMax1 = 2*pi/180;
omegaMax2 = 1*pi/180;
omegaMax3 = 1*pi/180;


% re-scale rate constraints (real physical constraints)
x_low =  [-omegaMax1*180/pi -omegaMax2*180/pi -omegaMax3*180/pi]';
x_up  =  [ omegaMax1*180/pi  omegaMax2*180/pi  omegaMax3*180/pi]';

% Plot Rates
figure('Name', 'Rates');
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t(1:k), x_sol_all{j}(i,1:k) * 180/pi, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\omega_%c [°/s]', 'x' + (i-1)));
    % title(sprintf('Rate \\omega_%c', 'x' + (i-1)));
    grid on;

    % plot gray shadded area
    xLimits = [0, t(k)]; 
    yDashed = x_low(i); 
    plot([0 t(k)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    miny =  x_low(i)+0.5* x_low(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    % xLimits = [0, simTime]; 
    yDashed =  x_up(i);
    plot([0 t(k)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    maxy = x_up(i)+0.5*x_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(k) miny maxy])
end

% custom legend
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-','Color',[0 0.4470 0.7410]);
h(2) = plot(NaN,NaN,'-','Color',[0.8500 0.3250 0.0980]);
legend(h, 'Discrete-time','Continous-time');

% Plot Control Torques
t_short = linspace(0, simTime, (simTime/simStepSize)-1);
% set up with torques in miliNetwonmeter
u_low = [-1 -1 -1]'*1000;
u_up  = [ 1  1 1]'*1000;

figure('Name', 'Control Torques');
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t_short(1:k-1), u_sol_all{j}(i,1:k-1) * 1000, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\tau_%c [mNm]', 'x' + (i-1)));
    % title(sprintf('Control Torque \\tau_%c', 'x' + (i-1)));
    grid on;

    % plot gray shadded area
    xLimits = [0, t(k-1)]; 
    yDashed = u_low(1); 
    miny =  u_low(i)+0.5* u_low(i);
    plot([0 t(k-1)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(k-1)]; 
    yDashed =  u_up(i); 
    plot([0 t(k-1)],[yDashed yDashed],'--','Color',[0.5 0.5 0.5]) 
    maxy = u_up(i)+0.5*u_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(k-1) miny maxy])

end
% custom legend
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-','Color',[0 0.4470 0.7410]);
h(2) = plot(NaN,NaN,'-','Color',[0.8500 0.3250 0.0980]);
legend(h, 'Discrete-time','Continous-time');


% Plot Solve Time
figure('Name', 'Solve Time');
for j = 1:numRuns
    semilogy(t_short(1:k-1), tEnd_all(j,1:k-1), 'Color', colors(j, :));
    hold on;
end
xlabel('Simulation time [s]');
ylabel('Computation time [s]');
axis([0 t_short(k-1) 1e-6 1])
grid on;

% custom legend
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'-','Color',[0 0.4470 0.7410]);
h(2) = plot(NaN,NaN,'-','Color',[0.8500 0.3250 0.0980]);
legend(h, 'Discrete-time','Continous-time');

tEnd_all(2,:)-tEnd_all(1,:)./tEnd_all(1,:)*100
