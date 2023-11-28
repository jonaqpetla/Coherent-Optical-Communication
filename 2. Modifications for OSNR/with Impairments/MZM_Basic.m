%The function which takes the inputs as MZM_function(E_IN,V_PI,V)
%where
% E IN : Real or complex electric field vector of size 1×N with units in in V/m.
%V PI : Vπ of the MZM given in V. Size: 1×1
% V : V (t) , which is the input modulation signal to MZM in V. Size: 1×N
% The output E_OUT is the output electric field.


function [E_OUT] = MZM_Basic(E_IN,V_PI,V)



%The equation is implemented below.
phase_val = 1i*pi*V/(2*V_PI);
E_OUT = E_IN.*(exp(phase_val)+exp(-1*phase_val))/2;
end

