clear; clc; close all

%% Parameters
J = diag([5, 6, 7]);          % Inertia matrix
Jinv = inv(J);
omega_r = [0.1; 0; 0];  % Desired angular rate (rad/s)

% Compute trim torque: u_r = omega_r x J*omega_r
u_r = cross(omega_r, J*omega_r);

%% Symbolic linearization
syms w1 w2 w3 u1 u2 u3 real
omega = [w1; w2; w3];
u = [u1; u2; u3];

omega_dot = Jinv * (u - cross(omega, J*omega));
A_sym = jacobian(omega_dot, omega);
B_sym = jacobian(omega_dot, u);

% Evaluate at omega_r, u_r
A = double(subs(A_sym, [w1, w2, w3, u1, u2, u3], [omega_r', u_r']));
B = double(subs(B_sym, [w1, w2, w3, u1, u2, u3], [omega_r', u_r']));

%% LQR controller for shifted system
Q = eye(3);
R = 0.1 * eye(3);
[K, P, ~] = lqr(A, B, Q, R);  % Terminal ingredients

%% Simulation parameters
dt = 0.01;
T = 10;
N = round(T/dt);
omega_sim = zeros(3, N+1);
torque_sim = zeros(3, N);
V_sim = zeros(1, N);

omega0 = [0; 0; 0];  % Initial state
omega_sim(:,1) = omega0;

%% Simulate nonlinear system
for k = 1:N
    omega_k = omega_sim(:,k);
    delta_omega = omega_k - omega_r;
    delta_u = -K * delta_omega;
    u = u_r + delta_u;
    
    % Nonlinear dynamics
    omega_dot = Jinv * (u - cross(omega_k, J*omega_k));
    omega_sim(:,k+1) = omega_k + dt * omega_dot;

    torque_sim(:,k) = u;
    V_sim(k) = delta_omega' * P * delta_omega;
end

time = 0:dt:T;

%% Plot angular rates
figure;
plot(time, omega_sim');
xlabel('Time (s)');
ylabel('Angular Rate (rad/s)');
legend('\omega_1', '\omega_2', '\omega_3');
title('Satellite Angular Rate Tracking');

%% Plot control torques
figure;
plot(time(1:end-1), torque_sim');
xlabel('Time (s)');
ylabel('Torque (Nm)');
legend('u_1', 'u_2', 'u_3');
title('Control Torque Inputs');

%% Plot Lyapunov function evolution
figure;
plot(time(1:end-1), V_sim, 'k');
xlabel('Time (s)');
ylabel('V(\delta \omega)');
title('Lyapunov Function Evolution');
