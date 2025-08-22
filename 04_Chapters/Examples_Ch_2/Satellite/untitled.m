import casadi.*

x1 = SX.sym('x1'); 
x2 = SX.sym('x2');
qp = struct('x',[x1;x2], 'f',x1^2+x2^2, 'g',x1+x2-2);


qp = {}
qp['h'] = H.sparsity()
qp['a'] = A.sparsity()
S = conic('S','qpoases',qp)
print(S)


S = qpsol('S', 'qpoases', qp);
disp(S)
S([0;0])