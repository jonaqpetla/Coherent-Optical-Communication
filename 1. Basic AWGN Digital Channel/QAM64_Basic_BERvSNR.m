clc; clear; close all;
% all signal vectors are column vectors
% How to handle bandwidth/symbol rate? Any bandwidth limitation/channel behaviour?

%% transmit symbols
no_symbols = 2^20;
for bits_per_symbol = [6]
    transmitted_bits = transpose(prbs(11, bits_per_symbol*no_symbols));
    transmitted_symbols = qammod(transmitted_bits, 2^bits_per_symbol, 'InputType','bit');

    oversampling_factor = 1;
    transmitted_waveform = repelem(transmitted_symbols, oversampling_factor);
    % scatterplot(transmitted_symbols);

    %% SNR effects, loop
    SNR_dB = 5:1:25;
    BER = zeros(1, length(SNR_dB));

    %figure;
    for SNR_iter = 1:length(SNR_dB)
        noise_power_linear = 10.^(-SNR_dB(SNR_iter)/10);
        received_waveform = awgn(transmitted_waveform, SNR_dB(SNR_iter), "measured");
        %received_symbols = resample();
        received_symbols = received_waveform;
        %scatterplot(received_symbols);
        received_bits = qamdemod(received_symbols,2^bits_per_symbol, 'OutputType','bit');
        bits_in_error = sum(received_bits ~= transmitted_bits);
        BER(SNR_iter) = bits_in_error/(no_symbols*bits_per_symbol);
    end
    % experimental
    % figure;
    loglog(SNR_dB, BER, 'Ko', 'linewidth', 1.2);
    hold on;
    
    % theoretical: Q(x) = 0.5 erfc(x/sqrt(2))
    % P_e_symbol = (M-1) Q(sqrt(d_min^2/2N_0)) = (2^m-1) [0.5*erfc( sqrt(d_min^2/4N_0) )]
    SNR_dB_smoother = SNR_dB(1):0.1:SNR_dB(length(SNR_dB));
    SNR_linear = 10.^(SNR_dB_smoother/10);
    BER_theoretical = 2/bits_per_symbol*(1 - 1/sqrt(2^bits_per_symbol))*...
        erfc(sqrt(3*SNR_linear/(2*(2^bits_per_symbol - 1))));
    % if bits_per_symbol == 2
    %      BER_theoretical = erfc(sqrt(SNR_linear/2));
    % end
    % if bits_per_symbol == 4
    %      BER_theoretical = 1*erfc(sqrt(1.5*SNR_linear/(2^bits_per_symbol - 1)));
    % end
    % if bits_per_symbol == 6
    %     BER_theoretical = (2^bits_per_symbol - 1 )*0.5*erfc(0.5*sqrt(SNR_dB_linear/4));
    % end

    loglog(SNR_dB_smoother, BER_theoretical, 'B-', 'linewidth', 1.2);
end
xlabel("SNR (dB)");
ylabel("BER");
xticks([0, 5, 10, 15, 25]);
title("BER vs SNR curves");
axis([0, 25, 0.0001, 0.5]);
legend(["64-QAM Simulation", "64-QAM Theoretical"]);
set(gca, 'fontsize', 14);