function filtered_output = f_ideal_filter(signal, filter_bandwidth, sampling_rate)
% a low pass brick wall ideal filter implemented in frequency domain
input_size = length(signal);
indexes_to_keep = zeros(input_size, 1);
index_cut_off = round( filter_bandwidth/sampling_rate * input_size );

indexes_to_keep(1:index_cut_off) = 1;
indexes_to_keep(end-index_cut_off:end) = 1;

filtered_output = ifft(indexes_to_keep .* fft(signal));
