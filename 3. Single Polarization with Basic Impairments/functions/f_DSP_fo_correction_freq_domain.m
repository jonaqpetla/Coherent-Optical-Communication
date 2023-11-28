function [compensated_output, offset_estimate] = ...
    f_DSP_fo_correction_freq_domain(input, sample_rate_Hz)
% there is room for improvement here. The estimate precision needs to go up

input_4th_power = input.^4;
input_4th_power_frequency_domain = fftshift(fft(input_4th_power));
[peak_value, peak_location] = max(abs(input_4th_power_frequency_domain));

delta_f = sample_rate_Hz/length(input);
frequency_axis_reference = [linspace(-sample_rate_Hz/2, -delta_f, length(input)/2), ... 
    linspace(0, sample_rate_Hz/2 - delta_f, length(input)/2)];
offset_estimate = frequency_axis_reference(peak_location)/4;

% figure; plot(20*log10(abs((input_4th_power_frequency_domain))));

% multiply in time domain
delta_t = 1/sample_rate_Hz;
time_axis = transpose( 0 : delta_t : (length(input) - 1)*delta_t );
compensated_output = input .* exp(-1j*2*pi* offset_estimate *time_axis);

% equivalently, rotate 1st power in frequency domain