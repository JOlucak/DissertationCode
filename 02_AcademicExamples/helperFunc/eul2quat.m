function q = eul2quat(phi, theta, psi)
    % Convert Euler angles (phi, theta, psi) to quaternion
    % Input: phi, theta, psi (Euler angles in radians)
    % Output: q (4x1 quaternion as a column vector)

    % Compute half-angles
    half_phi = phi / 2;
    half_theta = theta / 2;
    half_psi = psi / 2;

    % Compute trigonometric terms
    c_phi = cos(half_phi);
    s_phi = sin(half_phi);
    c_theta = cos(half_theta);
    s_theta = sin(half_theta);
    c_psi = cos(half_psi);
    s_psi = sin(half_psi);

    % Compute quaternion elements
    beta_0 = c_phi * c_theta * c_psi - s_phi * c_theta * s_psi;
    beta_1 = c_phi * s_theta * c_psi + s_phi * s_theta * s_psi;
    beta_2 = c_phi * s_theta * s_psi - s_phi * s_theta * c_psi;
    beta_3 = s_phi * c_theta * c_psi + c_phi * c_theta * s_psi;

    % Output quaternion as a column vector
    q = [beta_0; beta_1; beta_2; beta_3];
end
