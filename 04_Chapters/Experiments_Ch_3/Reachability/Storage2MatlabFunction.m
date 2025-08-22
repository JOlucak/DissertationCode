% /////////////////////////////////////////////////////////////////////////
%
% Short description: Generate an external matlab function of pre-computed
% storage function with vector input of states.
%
% 
% Author: Jan Olucak, Institue for Flight Mechanics and Control, University
% of Stuttgart, 2023
%
% ////////////////////////////////////////////////////////////////////////
clc
clear

% load only precomputed storage function
load AlphaEstimation.mat


syms sigma1 sigma2 sigma3 omega1 omega2 omega3 real
X = sym('x_',[6 1],'real')   ;
syms t real


Vval_sc  = cleanpoly(Vval_sc,1e-1);

% poly2symbolic
V = p2s(Vval_sc);

% substitute states as "function" to have vectorial input later
syms x(k)
V = subs(V,X',([x(1),x(2),x(3),x(4),x(5),x(6)]));

% go to directory for MPC and store external storage function there
% cd MPC\Matlab\
syms x
V = matlabFunction(V,...                       % function
                  'Vars',[t,x],...             % input variables
                  'File','StorageFunMRP',...   % name of external file
                  'Outputs',{'v'});            % output name

% a warning occurs; this not a bigger problem but we still clear the
% command window
clc

% simple check with external function; checking 
v = StorageFunMRP(0,zeros(6,1));

% go to back to original directory for comparison
% cd ..\..
% polynomial storage function
pvar x_1 x_2 x_3 x_4 x_5 x_6 t

v_poly = double(subs(Vval_sc,[x_1 x_2 x_3 x_4 x_5 x_6 t]',zeros(7,1)));

delta_v = v-v_poly;

disp(['Difference between external function and poly. function: ' num2str(delta_v)])
clear