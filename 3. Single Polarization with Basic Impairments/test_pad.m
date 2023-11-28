%% MZM transfer function
% v_pi = 5;
% v = -2*v_pi:0.01:2*v_pi;
% 
% E = 1;
% 
% E_out = f_mach_zehnder_modulator(E, v, v_pi, 50);
% figure; plot(v, E_out);

%% qammod, qamdemod defaut symbol order
% qam_order = 64;
% transmit_bits = transpose(prbs(17, 2^12*log2(qam_order)));
% modded = qammod(transmit_bits, qam_order, 'PlotConstellation',true);
% modded = modded + 0.5*(randn(length(modded), 1) + 1j*randn(length(modded), 1));
% demodded = qamdemod(modded, qam_order, 'PlotConstellation',true);

%% Debugging the coherent receiver
%% Originally written for: 
% Advanced Optical Communication Testbed, IIT Madras
% Authors : Jonaq N. Sarma, Sameer A. Mir, Dr. Deepa Venkitesh
% Dated : 23 Nov, 2023

% Distributed under MIT license
% Free to copy and distribute, with acknowledgements

%% Base code for single-polarization coherent optical communication
% with impairments. Equalization is performed digitally.
% '...' is used to break long statements

clc; clear; close all;

%% 1. Basic simulation parameters
no_symbols = 2^12;      
symbol_rate_Bd = 32e9;
qam_order = 16;
oversampling_factor = 8;
reference_frequency_Hz = 3e8/1550e-9;

total_no_samples = no_symbols*oversampling_factor;
sample_rate_Hz = symbol_rate_Bd*oversampling_factor;

%% 2. Generate random symbols
bits_per_symbol = log2(qam_order);
transmit_bits = transpose(prbs(17, no_symbols*bits_per_symbol));
transmit_symbols = qammod(transmit_bits, qam_order, 'InputType','bit');
% scatterplot(transmit_symbols); title("transmit symbol constellation");

%% 3. Electrical pulse shaping
transmit_symbols_oversampled = repelem(transmit_symbols, oversampling_factor);  % rectangular pulse

%% 4. Laser source - Linewidth, Power
laser_power_W = 1e-3;
laser_linewidth_Hz = 1;
laser_center_frequency_Hz = 3e8/1550e-9;
% reference_frequency = 3e8/1550e-9; % defined already in simulation settings
laser_source_field = f_laser(total_no_samples, sample_rate_Hz, laser_power_W,...
    laser_linewidth_Hz, laser_center_frequency_Hz, reference_frequency_Hz);

%% 5. Optical IQ modulator
v_pi_V = 5;
extinction_ratio_dB = 30;
modulator_phase_shifter = pi/2; % scalar or size(signal waveform)

v_bias = v_pi_V;
v_drive_I = v_bias - 0.2*real(transmit_symbols_oversampled);
v_drive_Q = v_bias - 0.2*imag(transmit_symbols_oversampled);

optically_modulated_signal = ...
    f_modulator_iq(laser_source_field, v_drive_I,  v_drive_Q,...
    v_pi_V, extinction_ratio_dB, modulator_phase_shifter);
scatterplot(optically_modulated_signal);

optically_modulated_signal = optically_modulated_signal/rms(optically_modulated_signal);
scatterplot(optically_modulated_signal);
hacked_bits = qamdemod(optically_modulated_signal(1:oversampling_factor:end), qam_order, 'OutputType', 'bit', 'UnitAveragePower', true);
hacked_ber = sum(transmit_bits ~= hacked_bits)/(no_symbols*bits_per_symbol); return;

%% 6. Fiber channel
fiber_dispersion_parameter = 17;  % ps/km-nm
fiber_attenuation_dB = 0.2; % dB/km
fiber_length_km = 0.1;      % km

received_optical_signal = f_fiber_basic(optically_modulated_signal, ...
    fiber_dispersion_parameter, fiber_attenuation_dB, fiber_length_km);

%% 7. Add ASE noise to set OSNR. 
% Equivalent of an EDFA, but OSNR is set directly instread of setting gain
osnr_dB = 30;
signal_power_W = mean(abs(received_optical_signal).^2);
ase_noise_power_W = signal_power_W / 10^(osnr_dB*0.1);
received_noisy_optical_signal = ...
    received_optical_signal + sqrt(ase_noise_power_W/2)*...
    (randn(total_no_samples, 1) + 1j*(randn(total_no_samples, 1)));

%% 7. Alternate: Add EDFA, consider ase noise

%% 8. Coherent receiver - Make LO, mix, take output from PD, filter output
local_oscillator_power_W = 1e-3;
local_oscillator_linewidth_Hz = 1;
local_oscillator_center_frequency_Hz = 3e8/1550e-9;

lo_laser_field = f_laser(total_no_samples, sample_rate_Hz, ...
    local_oscillator_power_W, local_oscillator_linewidth_Hz, ...
    local_oscillator_center_frequency_Hz, reference_frequency_Hz);

photodetector_responsivity_AperW = 1;           % A/W
photodetector_temperature_K = 300;              % Kelvin
photodetector_load_resistance_ohm = 50;         % ohm
photodetector_bandwidth_Hz = symbol_rate_Bd;    % Hz

[received_I_unfiltered, received_Q_unfiltered] = ...
    f_coherent_receiver(received_noisy_optical_signal, lo_laser_field, ...
    photodetector_responsivity_AperW, photodetector_temperature_K, ...
    photodetector_load_resistance_ohm, photodetector_bandwidth_Hz);

%% 9. Filter received signal to keep the main lobe
ideal_filter_bandwidth_Hz = 0.99*sample_rate_Hz;    % no filtering

received_I = f_ideal_filter(received_I_unfiltered, ...
    ideal_filter_bandwidth_Hz, sample_rate_Hz);
received_Q = f_ideal_filter(received_Q_unfiltered, ...
    ideal_filter_bandwidth_Hz, sample_rate_Hz);
received_symbols_uncorrected = received_I + 1j*received_Q;

scatterplot(received_symbols_uncorrected);
scatterplot(received_I_unfiltered + 1j*received_Q_unfiltered);
received_symbols_uncorrected = received_symbols_uncorrected / rms(received_symbols_uncorrected);
recovered_bits = qamdemod(received_symbols_uncorrected, qam_order, 'OutputType','bit');
bits_in_error = sum(recovered_bits(1:oversampling_factor:end) ~= transmit_bits);
return