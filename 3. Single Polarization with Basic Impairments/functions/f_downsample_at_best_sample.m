function downsampled_output = f_downsample_at_best_sample(input, oversampling_factor)
output_size = length(input)/oversampling_factor;
% downsampled_candidates = zeros(output_size, oversampling_factor);
downsampled_candidates = transpose(reshape(input, oversampling_factor, output_size));

% largest eye opening will show the largest variance
candidate_variance = var(downsampled_candidates);   % variance of each column
[best_variance, best_samples] = max(candidate_variance);
downsampled_output = downsampled_candidates(:, best_samples);