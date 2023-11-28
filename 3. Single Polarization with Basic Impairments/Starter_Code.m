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
addpath('.\functions');     % location of helper functions

%% 1. Basic simulation parameters
no_symbols = 2^20;      
symbol_rate_Bd = 32e9;
qam_order = 16;
oversampling_factor = 8;
reference_frequency_Hz = 3e8/1550e-9;

total_no_samples = no_symbols*oversampling_factor;
sample_rate_Hz = symbol_rate_Bd*oversampling_factor;

%% 2. Generate random symbols
bits_per_symbol = log2(qam_order);
transmit_bits = transpose(prbs(17, no_symbols*bits_per_symbol));
transmit_symbols = qammod(transmit_bits, qam_order, ...
    'InputType','bit', 'UnitAveragePower',true);
% scatterplot(transmit_symbols); title("transmit symbol constellation");

%% 3. Electrical pulse shaping
transmit_symbols_oversampled = repelem(transmit_symbols, oversampling_factor);  % rectangular pulse

%% 4. Laser source - Linewidth, Power
laser_power_W = 1e-3;
laser_linewidth_Hz = 100e3;
laser_center_frequency_Hz = 3e8/1550e-9;
% reference_frequency = 3e8/1550e-9; % defined already in simulation settings
laser_source_field = f_laser(total_no_samples, sample_rate_Hz, laser_power_W,...
    laser_linewidth_Hz, laser_center_frequency_Hz, reference_frequency_Hz);

%% 5. Optical IQ modulator
v_pi_V = 5;
extinction_ratio_dB = 30;
modulator_phase_shifter = pi/2; % scalar or size(signal waveform)

% f_precompensator(v_pi_V, transmit_symbols_oversampled) possible
v_bias = v_pi_V;
v_drive_I = v_bias - 1.5*real(transmit_symbols_oversampled);
v_drive_Q = v_bias - 1.5*imag(transmit_symbols_oversampled);

optically_modulated_signal = ...
    f_modulator_iq(laser_source_field, v_drive_I,  v_drive_Q,...
    v_pi_V, extinction_ratio_dB, modulator_phase_shifter);

%% 6. Fiber channel
fiber_dispersion_parameter = 17;  % ps/km-nm
fiber_attenuation_dB = 0.2; % dB/km
fiber_length_km = 1;      % km

received_optical_signal = f_fiber_basic(optically_modulated_signal, ...
    fiber_dispersion_parameter, fiber_attenuation_dB, fiber_length_km, ...
    reference_frequency_Hz, sample_rate_Hz);

%% 7. Add ASE noise to set OSNR. 
% Equivalent of an EDFA, but OSNR is set directly instread of setting gain
osnr_dB = 25;
osnr_linear = 10^(osnr_dB*0.1);
signal_power_W = mean(abs(received_optical_signal).^2);
ase_noise_power_W = signal_power_W / osnr_linear;
received_noisy_optical_signal = ...
    received_optical_signal + sqrt(ase_noise_power_W/2)*...
    (randn(total_no_samples, 1) + 1j*(randn(total_no_samples, 1)));

%% 7. Alternate: Add EDFA, consider ase noise

%% 8. Coherent receiver - Make LO, mix, take output from PD, filter output
local_oscillator_power_W = 1e-3;
local_oscillator_linewidth_Hz = 100e3;
local_oscillator_center_frequency_Hz = 3e8/1550e-9 + 2e9;

lo_laser_field = f_laser(total_no_samples, sample_rate_Hz, ...
    local_oscillator_power_W, local_oscillator_linewidth_Hz, ...
    local_oscillator_center_frequency_Hz, reference_frequency_Hz);

photodetector_responsivity_AperW = 1;           % A/W
photodetector_temperature_K = 300;              % Kelvin
photodetector_load_resistance_ohm = 1e6;         % ohm
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
received_symbols_uncorrected = (received_I + 1j*received_Q);

%% 10. DSP to correct for impairments
%% 10. (a) Chromatic Dispersion Compensation
received_symbols_CD_corrected = ...
    f_DSP_cd_compensation_freq_domain(received_symbols_uncorrected, ...
    fiber_dispersion_parameter, fiber_length_km, ...
    reference_frequency_Hz, sample_rate_Hz);

%% 10. (b) Frequency Offset Compensation
[received_symbols_CD_FO_corrected, frequency_offset_estimate] = ...
    f_DSP_fo_correction_freq_domain(received_symbols_CD_corrected, sample_rate_Hz);

%% 10. (c) Phase Noise Correction
learning_rate = 20e-3;
received_symbols_CD_FO_PN_corrected = ...
    f_DSP_pn_correction_lms(received_symbols_CD_FO_corrected, ...
    learning_rate, qam_order);

%% Downsample at the largest eye opening, demodulate
received_symbols_corrected = f_downsample_at_best_sample(...
    received_symbols_CD_FO_PN_corrected, oversampling_factor);
received_symbols_corrected = ...
    received_symbols_corrected/rms(received_symbols_corrected);
% scatterplot(received_symbols); title("received symbol constellation");
received_bits = qamdemod(received_symbols_corrected, qam_order, ...
    'OutputType','bit', 'UnitAveragePower', true);

bits_in_error = sum(received_bits ~= transmit_bits)