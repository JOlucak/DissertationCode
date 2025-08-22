function mrp = quat2mrp(q)
    % Convert quaternion to Modified Rodrigues Parameters (MRP)
    % Input: q (4x1 quaternion as a column vector)
    % Output: mrp (3x1 MRP vector as a column vector)

    % Extract quaternion components
    beta_0 = q(1);
    beta = q(2:4); % Vector part (beta_1, beta_2, beta_3)

    % Compute MRP using the formula
    mrp = beta / (1 + beta_0);
end
