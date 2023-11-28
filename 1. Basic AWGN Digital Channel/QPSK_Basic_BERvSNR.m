clc; clear; close all;
% all signal vectors are column vectors
% How to handle bandwidth/symbol rate? Any bandwidth limitation/channel behaviour?

%% transmit symbols
no_symbols = 2^18;
for bits_per_symbol = [2]
    transmitted_bits = transpose(prbs(11, bits_per_symbol*no_symbols));
    transmitted_symbols = qammod(transmitted_bits, 2^bits_per_symbol, 'InputType','bit');

    oversampling_factor = 1;
    transmitted_waveform = repelem(transmitted_symbols, oversampling_factor);
    % scatterplot(transmitted_symbols);

    %% SNR effects, loop
    SNR_dB = 3:1:15;
    BER = zeros(1, length(SNR_dB));

    %figure;
    for SNR_iter = 1:length(SNR_dB)
        noise_power_linear = 10.^(-SNR_dB(SNR_iter)/10);
        received_waveform = awgn(transmitted_waveform, SNR_dB(SNR_iter), "measured");
        % received_waveform = transmitted_waveform + 0.5*sqrt(noise_power_linear)*(mean(abs(transmitted_waveform).^2))*(randn(length(transmitted_waveform), 1) + 1j*randn(length(transmitted_waveform), 1));
        received_symbols = received_waveform;
        received_bits = qamdemod(received_symbols,2^bits_per_symbol, 'OutputType','bit');
        bits_in_error = sum(received_bits ~= transmitted_bits);
        BER(SNR_iter) = bits_in_error/(no_symbols*bits_per_symbol);
    end
    % figure;
    loglog(SNR_dB, BER, 'ko', 'linewidth', 1.2);
    hold on;
    
    SNR_dB_smoother = SNR_dB(1):0.1:SNR_dB(length(SNR_dB));
    SNR_linear = 10.^(SNR_dB_smoother/10);
    BER_theoretical = 2/bits_per_symbol*(1 - 1/sqrt(2^bits_per_symbol))*...
        erfc(sqrt(3*SNR_linear/(2*(2^bits_per_symbol - 1))));
    loglog(SNR_dB_smoother, BER_theoretical, 'b-', 'linewidth', 1.2);
end
xlabel("SNR (dB)");
ylabel("BER");
xticks([0, 5, 10, 15]);
title("QPSK");
axis([0, 15, 0.000001, 0.5]);
legend(["QPSK Simulation", "QPSK Theoretical"]);
set(gca, 'fontsize', 14);