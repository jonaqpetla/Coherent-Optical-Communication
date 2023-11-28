function output_current = f_balanced_photodetector(E_1, E_2, ...
    responsivity, temperature, load_resistance, bandwidth)
% f_balanced_photodetector Help to be added
% f_photodetector used internally to reduce code duplication

I_1 = f_photodetector(E_1, responsivity, temperature, load_resistance, bandwidth);
I_2 = f_photodetector(E_2, responsivity, temperature, load_resistance, bandwidth);
output_current = I_1 - I_2;