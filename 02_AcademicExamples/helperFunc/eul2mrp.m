function mrp = eul2mrp(phi, theta, psi)

% transform Euler angles to quaternion
q = eul2quat(phi, theta, psi);

% quaternion to MRP
mrp = quat2mrp(q);

end