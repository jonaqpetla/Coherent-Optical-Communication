function compensated_output = f_DSP_cd_compensation_freq_domain(input, ...
    fiber_dispersion_parameter_psperkmnm, fiber_length_km, ...
    reference_frequency_Hz, sample_rate_Hz)
% f_fiber_basic model using attenuation, alpha, and dispersion, beta_2
wavelength = 3e8/reference_frequency_Hz;
beta_2 = -fiber_dispersion_parameter_psperkmnm*1e-6 * wavelength^2/(2*pi*3e8);
frequency_axis = linspace(-sample_rate_Hz/2, sample_rate_Hz/2, length(input));
angular_frequency_axis = 2*pi*transpose(frequency_axis);
transfer_function_frequency_domain = exp(1j*0.5*beta_2*fiber_length_km*1e3*...
    angular_frequency_axis.^2);

input_frequency_domain = fftshift(fft(input));
output_frequency_domain = transfer_function_frequency_domain .* input_frequency_domain;
compensated_output = ifft(ifftshift(output_frequency_domain));