function [E_out_1, E_out_2] = f_directional_coupler_5050(E_in_1, E_in_2)
% some help lines

% make zero vectors in case a scalar zero came in
if (E_in_1 == 0); E_in_1 = zeros(length(E_in_2), 1); end
if (E_in_2 == 0); E_in_2 = zeros(length(E_in_1), 1); end

% output column vector = transfer matrix X input column vector
E_in_stack = [transpose(E_in_1); transpose(E_in_2)];
transfer_function = 1/sqrt(2)*[1 1j; 1j 1];
E_out_stack = transfer_function*E_in_stack;
E_out_1 = transpose(E_out_stack(1, :));
E_out_2 = transpose(E_out_stack(2, :));
