function [phi,theta,psi] = mrp2eul(sigma)

% transform MRP to quaternion; scalar part first (see book Junkins p.117)
q =  mrp2quat(sigma);

% Euler sequence 1,2,3

q0 = q(1);
q1 = q(2);
q2 = q(3);
q3 = q(4);


phi = atan2(-2*(q2*q3-q0*q1),q0*q0-q1*q1-q2*q2+q3*q3);
theta = asin(2*(q1*q3 + q0*q2));
psi= atan2(-2*(q1*q2-q0*q3),q0*q0+q1*q1-q2*q2-q3*q3);


end