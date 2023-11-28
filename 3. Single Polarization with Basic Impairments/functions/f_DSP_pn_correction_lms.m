function compensated_output = f_DSP_pn_correction_lms(input, learning_rate, qam_order)

ideal_constellation_points = qammod( transpose([0:qam_order-1]),...
    qam_order, 'UnitAveragePower',true );
input = input / rms(input);
compensated_output = zeros(length(input), 1);

filter_tap_value = 1;

for iteration = 1:length(input)
    current_estimate = filter_tap_value * input(iteration);
    [~, closest_match] = min(abs(current_estimate - ideal_constellation_points));
    
    current_error = ideal_constellation_points(closest_match) - current_estimate;

    filter_tap_value = filter_tap_value + ...
        learning_rate * current_error .* conj(input(iteration))/abs(input(iteration)).^2;
    compensated_output(iteration) = current_estimate;
end