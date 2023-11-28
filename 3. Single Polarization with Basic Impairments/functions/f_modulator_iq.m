function optical_out = f_modulator_iq(laser_field, v_drive_I,  v_drive_Q, ...
    v_pi_V, extinction_ratio_dB, phase_shifter_phase)
% f_modulator_iq Help to be added here

optical_out = ...
    f_mach_zehnder_modulator(laser_field, v_drive_I, v_pi_V, extinction_ratio_dB) +...
    f_mach_zehnder_modulator(laser_field, v_drive_Q, v_pi_V, extinction_ratio_dB) .* ...
    exp(1j*phase_shifter_phase);