clc; clear; close all;

no_symbols = 2^14;
symbol_rate = 56e9;
qam_order = 64;

no_bits = no_symbols*log2(qam_order);
transmitted_bits = transpose(prbs(13, no_bits));
transmitted_symbols = qammod(transmitted_bits, qam_order, 'InputType','bit');

osnr_vector_dB = 5:2:35;
reference_bandwidth = 12.5e9;
ber_vector = zeros(1, length(osnr_vector_dB));
for osnr_iter = 1:length(osnr_vector_dB)
    % % problem bit:
    % h = 6.62607015e-34; c = 3e8; lambda = 1550e-9; G = 10;
    % signal_power = mean(abs(transmitted_symbols).^2);
    % noise_figure_dB = 10*log10(signal_power) + 57.955 ...
    % - 10*log10(G-1) - osnr_vector_dB(osnr_iter);
    % noise_figure_linear = 10.^(noise_figure_dB./10);
    % noise_power = (noise_figure_linear)*(h*c/lambda)*(G-1)*2*symbol_rate;
    
    % hack:
    signal_bandwidth = symbol_rate;
    snr_actual = osnr_vector_dB(osnr_iter)...
        + 10*log10(reference_bandwidth) - 10*log10(signal_bandwidth);

    % Basically awgn()
    snr_actual_linear = 10^(snr_actual/10);
    noise_power = 1/snr_actual_linear*mean(abs(transmitted_symbols).^2);
    received_symbols = transmitted_symbols ...
       + sqrt(noise_power/2)*randn(length(transmitted_symbols), 1)...
       + 1j*sqrt(noise_power/2)*randn(length(transmitted_symbols), 1);
    % received_symbols = awgn(transmitted_symbols, snr_actual, "measured");

    scatterplot(received_symbols);
    received_bits = qamdemod(received_symbols, qam_order, 'OutputType','bit');
    no_bits_in_error = sum(received_bits ~= transmitted_bits);
    ber_vector(osnr_iter) = no_bits_in_error/no_bits;
end

osnr_vector_smooth = osnr_vector_dB(1):0.1:osnr_vector_dB(end);
snr_vector = 10.^(0.1*osnr_vector_smooth) * reference_bandwidth/symbol_rate;
ber_theoretical = 2/log2(qam_order)*(1 - 1/(sqrt(qam_order)))...
    *erfc(sqrt(3*snr_vector/(2*(qam_order - 1))));

figure; 
loglog(osnr_vector_smooth, ber_theoretical, 'b-', 'linewidth', 1.4); 
hold on; 
loglog(osnr_vector_dB, ber_vector, 'ko', 'linewidth', 1.4); 
xlabel("OSNR (dB)"); ylabel("BER"); 
legend(["Theoretical", "Simulation"]);  
grid minor; 
axis([5, 35, 1e-4, 1]); 
title("BER vs OSNR, " + num2str(symbol_rate/1e9) + " GBaud " + num2str(qam_order) + "-QAM");
% title("BER vs OSNR, " + num2str(symbol_rate/1e9) + " GBaud " + "QPSK");
set(gca, 'FontSize', 14);