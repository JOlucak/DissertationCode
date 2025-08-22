function q =  mrp2quat(sigma)
    
q0 = (1-sigma'*sigma)/(1+sigma'*sigma);


q1 = (2*sigma(1))/(1+sigma'*sigma);
q2 = (2*sigma(2))/(1+sigma'*sigma);
q3 = (2*sigma(3))/(1+sigma'*sigma);

% scalar part first
q = [q0;q1;q2;q3];

end