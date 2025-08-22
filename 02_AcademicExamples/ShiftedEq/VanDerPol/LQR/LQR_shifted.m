%% ------------------------------------------------------------------------
% LQR Control for Van der Pol Oscillator (Shifted Equilibrium)
% Equilibrium point: x_e = [0.2;0], u_e = 0.2
% The controller is designed in shifted coordinates and applied
% to the *nonlinear* system in simulation.
% ------------------------------------------------------------------------

close all
clear
clc

%% Parameters
simTime = 20;         % seconds
dt = 0.1;            % integration step for simulation
x0_real = [-0.63;0.4];% initial condition (real)

% Cost weights
Q = diag([10,1]);
R = 0.5;

% Equilibrium point
x_e = [0.2;0];
u_e = 0.2;

%% Define Van der Pol dynamics
f_vdp = @(x,u)[ x(2);
                (1-x(1)^2)*x(2) - x(1) + u ];

%% Compute Jacobians at equilibrium (linearization)
syms x1 x2 u
x_sym = [x1;x2];
f_sym = [ x2;
          (1-x1^2)*x2 - x1 + u ];

A_sym = jacobian(f_sym,x_sym);
B_sym = jacobian(f_sym,u);

A = double(subs(A_sym,{x1,x2,u},{x_e(1),x_e(2),u_e}));
B = double(subs(B_sym,{x1,x2,u},{x_e(1),x_e(2),u_e}));

%% Design LQR Gain
K = lqr(A,B,Q,R);

%% Simulation Loop
Nsteps = round(simTime/dt);
x_sim = zeros(2,Nsteps);
u_sim = zeros(1,Nsteps-1);
t = linspace(0,simTime,Nsteps);

x_sim(:,1) = x0_real;

for k = 1:Nsteps-1
    % Compute deviation in shifted coordinates
    x_shift = x_sim(:,k) - x_e;

    % LQR control law (increment)
    u_shift = -K*x_shift;

    % Convert back to real control
    u_real = u_e + u_shift;
    u_sim(k) = u_real;


    x_sim(:,k+1) = sim_mex(x_sim(:,k), u_real);
end

%% Plot results
figure;
subplot(3,1,1)
plot(t,x_sim(1,:),'LineWidth',1.5); hold on
yline(x_e(1),'--r'); ylabel('x_1 (state)');
title('Van der Pol with Shifted LQR Control');
subplot(3,1,2)
plot(t,x_sim(2,:),'LineWidth',1.5); hold on
yline(x_e(2),'--r'); ylabel('x_2 (state)');
subplot(3,1,3)
plot(t(1:end-1),u_sim,'LineWidth',1.5); hold on
yline(u_e,'--r'); ylabel('u (control)');
xlabel('Time [s]');
