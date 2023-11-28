function optical_out_ideal = ...
    f_mach_zehnder_modulator(laser_field, driving_voltage, v_pi_V, extinction_ratio_dB)
% f_mach_zehnder_modulator Help to go here. Currently an ideal modulator.

phase_val = 1j*pi*driving_voltage/(2*v_pi_V);
optical_out_ideal = laser_field.*(exp(phase_val)+exp(-1*phase_val))/2;