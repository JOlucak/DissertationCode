function [z0] = straightLine(N,x0,xf,nx,u0,uf,nu)

grid         = linspace(0,1,N);
state_0_grid = reshape(x0 + grid.*(xf - x0),nx*N,1);

% state_0_grid = zeros(size(state_0_grid));
state_0_grid(1:nx) = x0';
state_0_grid(isinf(state_0_grid)) = 0;

if nu ~=0
control_0_grid = reshape(u0 + grid.*(uf - u0),nu*N,1);
else
   control_0_grid = []; 
end

z0 = [state_0_grid; control_0_grid];