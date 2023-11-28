function [I, Q] = f_coherent_receiver(optical_signal, local_oscillator_field, ...
    photodetector_responsivity, photodetector_temperature, photodetector_load_resistance, photodetector_bandwidth)
% f_coherent_receiver Implements the basic coherent receiver
% uses sub-functions: f_directional_coupler, f_balanced_detector
% transconductor and phase shifter are hard coded

[E1, E2] = f_directional_coupler_5050(optical_signal, 0);
[E3, E4] = f_directional_coupler_5050(local_oscillator_field, 0);

phase_shift = pi/2;
E4_phase_shifted = exp(1j*phase_shift)*E4;

[E5, E6] = f_directional_coupler_5050(E1, E3);
[E7, E8] = f_directional_coupler_5050(E2, E4_phase_shifted);

Q = f_balanced_photodetector(E5, E6, ...
    photodetector_responsivity, photodetector_temperature, ...
    photodetector_load_resistance, photodetector_bandwidth);
I = f_balanced_photodetector(E8, E7, ...
    photodetector_responsivity, photodetector_temperature, ...
    photodetector_load_resistance, photodetector_bandwidth);