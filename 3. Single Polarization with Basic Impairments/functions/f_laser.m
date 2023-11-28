function field = f_laser(no_samples, sample_rate, power_W,...
    linewidth_Hz, center_frequency_Hz, reference_frequency_Hz)
% f_laser Simulates a laser field by considering its phase noise
% outputs a vector containing electric fiel amplitudes for each point of time

% power
field_amplitudes = sqrt(power_W)*ones(no_samples, 1);

% phase noise as a random walk
delta_phase_noise_variance = 2*pi*linewidth_Hz/sample_rate;
delta_phase_noise = sqrt(delta_phase_noise_variance)*randn(no_samples, 1);
phase_noise = cumsum(delta_phase_noise);

% frequency offset from reference
dt = 1/sample_rate;
t = transpose( 0 : dt : (no_samples-1)*dt );

% total field = product of elementwise amplitude X phase noise X freq offset
field = field_amplitudes.*exp(1j*phase_noise).*...
    exp(1j*2*pi*(center_frequency_Hz - reference_frequency_Hz)*t);