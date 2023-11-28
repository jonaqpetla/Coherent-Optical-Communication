function photocurrent = f_photodetector(electric_field, responsivity, temperature, ...
    load_resistance, bandwidth)
% f_photodetector simulates the effect of a photodetector. Considers thermal noise
% and shot noise, converts an electric field into photocurrent
photocurrent_noiseless = responsivity * abs(electric_field).^2;
boltzmann_constant = 1.38e-23;

thermal_noise_variance = 4*boltzmann_constant*temperature/load_resistance*bandwidth;
thermal_noise_current = sqrt(thermal_noise_variance)*randn(length(electric_field), 1);

photocurrent = photocurrent_noiseless + thermal_noise_current;