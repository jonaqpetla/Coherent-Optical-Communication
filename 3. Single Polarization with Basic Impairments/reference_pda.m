

%% PM-QPSK Transmitter
clc;
clear all;
close all;
%% Generate QPSK data


% QPSK Data Parameters
prbs_order_X = 12;% The order of the sequence as we need PRBS11
Total_no_symbols_X = 2^20;% Total number of symbols required is 2^16  % 5%
Total_no_symbols_Y = Total_no_symbols_X; %Total number of symbols required is 2^14
Symbol_Rate = 32e9; % The BaudRate is defined as 32 GBaud
Symbol_repeat_X = 8;% The number of samples per symbol
quam_mode_X = 4; % Since QPSK


%% Generate the bit/data sequence
Total_N_bits = 2*Total_no_symbols_X;  % QPSK - 2 bits per symbol- Total No of bits = Total number of symbols*2
Sequence = prbs(prbs_order_X,Total_N_bits); % The entire sequence of length = Total No of bits  %[0 1 0 0 1 1 1 0 1 1];%
dt_symbol = 1/Symbol_Rate; %symbol duration
Bit_Rate = 2*Symbol_Rate; % As the data rate given is 32GBaud per sec, Bit rate = 64GBits per sec
T  = Total_no_symbols_X*dt_symbol;%Total duration of bit sequence
dt = 1/(Symbol_repeat_X*Symbol_Rate);
t=0:dt:T-dt;



%% Bit sequence for X and Y polarized data
Sequence_X = Sequence;
section = 1000;% The number of samples to be put at the end to create the different data(Y polarized)
Sequence_Y =  [Sequence_X((section/4)+1:end)  Sequence_X(1:section/4)]; % This is created to compare the Recived Y polarized data with Y sequence

%% Ceate the QPSK Data for X polarization

%The bit sequence are to be grouped (2 bits ) and treat as a symbol(Bit to symbol mapping)

col1_X = Sequence(1:2:end).';% Taking the odd part and taking Transpose.
col2_X = Sequence(2:2:end).';% Taking the even part and taking Transpos.
append_col_X = [col1_X col2_X];% append the two to get the symbol requence.Each row has one symbol.

% Till now we have grouped the 2 bits as one symbol
symbols_X = bi2de(append_col_X,'left-msb'); % Binary to decimal conversion
symb_map = [0 1 2 3]; % Specifying the order of mapping (default it will take gray coding)
qam_signal_X = qammod(symbols_X,quam_mode_X,symb_map);% Compute the QPSK symbols . Convert the decimal numbers to complex numbers.

% Now the complex symbols are assigned to each 'group of 2 bits' in the sequence

Nt = Symbol_repeat_X;    % Number of samples per symbol
QPSK_Sequence_X = repelem(qam_signal_X,Symbol_repeat_X); % Repat each symbol Nt times.
Data_X = QPSK_Sequence_X; % X polarized Data
Data_Y = [QPSK_Sequence_X(section+1:end) ; QPSK_Sequence_X(1:section)]; % Y polarized Data
qam_signal_Y = [Data_Y(1:Nt:end)];


In_phase_X = real(Data_X); % In phase component of X polarized data
Quad_phase_X  = imag(Data_X); % Quadrature phase component of X polarized data
In_phase_Y = real(Data_Y); % In phase component of Y polarized data
Quad_phase_Y  = imag(Data_Y); % Quadrtaure phase component of Y polarized data

%% Scatterplot for X and Y data
% scatterplot(Data_X);% Scatterplot for XData
% title("Scatterplot for X Data");
% scatterplot(Data_Y);% Scatterplot for YData
% title("Scatterplot for Y Data");

%%  Generate the Laser Source at Transmitter Side

length_of_laser_source  = Total_no_symbols_X*Symbol_repeat_X; % The laser source also has the same length as that of the number of bits
symbol_duration = (1/Symbol_Rate);
delta_f = 1*10*1e3;
laserpower = 1e-3; % Laser power is 1mW
E_IN = sqrt(laserpower)*ones(1,length_of_laser_source); % Define the laser source sequnce with the given power
phase_seq = (randn(round(length_of_laser_source),1))';% To make it integer value used the round command
phase_noise_variance  = 2*pi*delta_f*symbol_duration; % Compute the phase noise variance
corrected_phase_seq = sqrt(phase_noise_variance)*(phase_seq)/sqrt(var(phase_seq)); % Define the phase nois with the computed variance
corrected_phase_seq(1) = 0; % Define the first element in phase noise sequence as zero
phi_of_k = cumsum(corrected_phase_seq); % Compute the phase noise as cumulative sum
E_IN_Laser_source = E_IN.*exp(1i*phi_of_k); % Define the laser source with the computed phase noise and power

E_IN_X = E_IN_Laser_source/sqrt(2); % Splitting the laser source for X and Y polarization
E_IN_Y = E_IN_X;


%% Generating modulated signal using IQ-Modulator

V_PI = 3; % Vpi voltage

% Bias the data to -Vpi voltage . So the swing is from -2Vpi to 0

% X Polarized
Biased_In_Phase_Data_X = V_PI*In_phase_X-V_PI;% Bias the In phase component of X polarized data
Biased_Q_Phase_Data_X = V_PI*Quad_phase_X-V_PI;% Bias the Quadrature phase component of X polarized data
% Y Polarized
Biased_In_Phase_Data_Y = V_PI*In_phase_Y-V_PI;% Bias the In phase component of Y polarized data
Biased_Q_Phase_Data_Y = V_PI*Quad_phase_Y-V_PI;% Bias the In Quadrature component of Y polarized data

% Modulating X Polarized Data
E_OUT_In_X = MZM_Basic(E_IN_X',V_PI,Biased_In_Phase_Data_X);
E_OUT_Quad_X = MZM_Basic(E_IN_X',V_PI,Biased_Q_Phase_Data_X);
% Modulating Y Polarized Data
E_OUT_In_Y =  MZM_Basic(E_IN_Y',V_PI,Biased_In_Phase_Data_Y);
E_OUT_Quad_Y = MZM_Basic(E_IN_Y',V_PI,Biased_Q_Phase_Data_Y);

E_out_X = E_OUT_In_X + 1i*E_OUT_Quad_X;     % Output of the  X Polarized IQ Modulator
E_out_Y = E_OUT_In_Y + 1i*E_OUT_Quad_Y;     % Output of the  Y Polarized IQ Modulator

%% Scatterplot
% scatterplot(E_out_X);% Scatterplot for XData
% title("Scatterplot for X Polarized IQ Modulator Output");
% scatterplot(E_out_Y);% Scatterplot for YData
% title("Scatterplot for Y Polarized IQ Modulator Output");



norm_power_X = E_out_X./sqrt(mean(abs(E_out_X).^2)); % Normalize the power for X polarization
norm_power_Y = E_out_Y./sqrt(mean(abs(E_out_Y).^2)); % Normalize the power for Y polarization
% %
% %



%% Local Oscillator at the Reciever Side

length_of_laser_source_LO  = Total_no_symbols_X*Symbol_repeat_X;
symbol_duration_LO = 1/Symbol_Rate;
linewidth_of_LO = 1*100*1e3;%100*1e3;
laserpower_LO_dBm = 0;
laserpower_LO = 10^(0.1*laserpower_LO_dBm)*1e-3;
E_LO = sqrt(laserpower_LO)*ones(length_of_laser_source,1);
phase_seq_LO = (randn(round(length_of_laser_source),1));
phase_noise_variance_LO  = 2*pi*linewidth_of_LO*symbol_duration;
corrected_phase_seq_LO = sqrt(phase_noise_variance_LO)*(phase_seq_LO)/sqrt(var(phase_seq_LO));
corrected_phase_seq_LO(1) = 0;
phi_of_k_LO = cumsum(corrected_phase_seq_LO);
E_LO_Laser_source = E_LO.*exp(1i*phi_of_k_LO);% This is the Local oscillator at the Rceiver Side




%% Frequency Offset start
nfft = length_of_laser_source_LO;
detunig_freq = 0.2*500e6;
fmin  =  Symbol_Rate/ Total_no_symbols_X;
delta_f_LO = floor(detunig_freq/fmin);
LO_detuning = 2*pi*delta_f_LO/nfft*(1:nfft);
E_LO_Laser_source = E_LO.*exp(1i.*LO_detuning');
%% Frequency Offset end




E_LO_Laser_source_X = E_LO_Laser_source/sqrt(2); % Local oscillator signal to be used with X polarized dat
%E_LO_Laser_source_X = E_LO_Laser_source;
E_LO_Laser_source_Y = E_LO_Laser_source_X; % Local oscillator signal to be used with Y polarized dat








%% Adding ASE noise
c = 3e8;% Velocity of light
h = 6.62607015e-34; % Plank's constant
G_dB = 12;% Gain in dB
G_lin = 10^(.1*G_dB); % Convert the gain in dB to linear scale
lamda = 1550*1e-9; % Wavelength is 1550 nm
P_sig_X = bandpower(norm_power_X); %Computing  the average power of the X polarized input of the EDFA
P_sig_Y = bandpower(norm_power_Y); %Computing  the average power of the Y polarized input of the EDFA
OSNR_Array_dB = 5:1:25;% %10:1:18; %20;%0:1:12; % The OSNR value is defined here
%OSNR_Array_dB = 30;
R_d = 1;   % Responsivity is defined as 1

for m = 1:length(OSNR_Array_dB)   %-10*log10((h*c/lamda)*B1_ref)-10*log10(1000)

    %% Computation of ASE noise variance for X polarization
    NF_indB= 10*log10(P_sig_X)+57.955-10*log10(G_lin-1)-OSNR_Array_dB(m) ;% noise figure is 2*ita_sp
    Noisefig_lin = 10.^(NF_indB./10);
    P_ASE_lin = (Noisefig_lin./4)*(h*c/lamda)*(G_lin-1)*Nt*Symbol_Rate; %

    ASE_noise_I_X  = randn(length(E_out_X),1);
    ASE_noise_Q_X  = randn(length(E_out_X),1);
    ASE_noise_corrected_I_X = sqrt(P_ASE_lin)*ASE_noise_I_X;%/sqrt(var(ASE_noise_I));
    ASE_noise_corrected_Q_X = sqrt(P_ASE_lin)*ASE_noise_Q_X;%/sqrt(var(ASE_noise_Q));
    ASE_noise_X = ASE_noise_corrected_I_X+1i*ASE_noise_corrected_Q_X;

    % Amplifying the attenuated signal
    E_OUT_trans_amplified_X = E_out_X;

    % Adding the ASE noise to the amplified signal
    E_OUT_transmitted_X = E_OUT_trans_amplified_X+ASE_noise_X; %Adding the ASE noise  This is the X polarized transmitted data


    %% Computation of ASE noise variance for Y polarization
    NF_indB= 10*log10(P_sig_Y)+57.955-10*log10(G_lin-1)-OSNR_Array_dB(m) ;% noise figure is 2*ita_sp
    Noisefig_lin = 10.^(NF_indB./10);
    P_ASE_lin_Y = (Noisefig_lin./4)*(h*c/lamda)*(G_lin-1)*Nt*Symbol_Rate; %

    ASE_noise_I_Y  = randn(length(E_out_Y),1);
    ASE_noise_Q_Y  = randn(length(E_out_Y),1);
    ASE_noise_corrected_I_Y = sqrt(P_ASE_lin_Y)*ASE_noise_I_Y;%/sqrt(var(ASE_noise_I));
    ASE_noise_corrected_Q_Y = sqrt(P_ASE_lin_Y)*ASE_noise_Q_Y;%/sqrt(var(ASE_noise_Q));
    ASE_noise_Y = ASE_noise_corrected_I_Y+1i*ASE_noise_corrected_Q_Y;

    % Amplifying the attenuated signal
    E_OUT_trans_amplified_Y = E_out_Y;

    % Adding the ASE noise to the amplified signal
    E_OUT_transmitted_Y = E_OUT_trans_amplified_Y+ASE_noise_Y; %  Adding the ASE noise This is the Y polarized transmitted data

    %end

    %% Scatterplot
    %% scatterplot(E_OUT_transmitted_X);
    % title("Scatterplot for X Polarized IQ Modulator with OSNR 20 dB");
    %scatterplot(E_OUT_transmitted_Y);
    % title("Scatterplot for Y Polarized IQ Modulator with OSNR 20 dB");






    %% Polarization mixing

    alpha_pol_mix  = .6; % Set the value for split ratio
    delta_pol_mix  = pi/6; % Set the value for phase delay between X and Y polarization.

    %% Define the Jones matrix

    pxx = sqrt(alpha_pol_mix)*exp(1i*delta_pol_mix); % Set pxx value
    pxy = sqrt(1 - alpha_pol_mix); % Set pxy value
    pyx = -1*pxy; % Set pyx value
    pyy = sqrt(alpha_pol_mix)*exp(-1*1i*delta_pol_mix); % Set pyy value


    E_OUT_X_at_fiberend= pxx*E_OUT_transmitted_X+pxy*E_OUT_transmitted_Y;
    E_OUT_Y_at_fiberend= pyx*E_OUT_transmitted_X+pyy*E_OUT_transmitted_Y;

    E_Mixing_OUT_X_at_fiberend = E_OUT_X_at_fiberend;
    E_Mixing_OUT_Y_at_fiberend = E_OUT_Y_at_fiberend;
    %CHK2 = E_OUT_X_at_fiberend;
    % %



    %% Chromatic Dispersion for X

    % scatterplot(E_OUT_X_at_fiberend);
    % title('Before CD Intro');
    E_out_CD_in_X = E_Mixing_OUT_X_at_fiberend;

    %  Fiber Optic Channel
    % Fiber Transerfunction.

    Fs_data_CD_X = Nt*Symbol_Rate;
    f_var_CD_X =linspace(-Fs_data_CD_X/2,+Fs_data_CD_X/2, length(E_out_CD_in_X));
    FiberLength_X = 1*80e3;

    c_CD_X = 3*1e8; % Speed of light in m/s
    D_CD_X= 17*1*1e-6; % Dispersion has taken for wavelength 1550nm in ps/(km-nm) = 10*1e-6(in s/(m-m))
    lambda_CD_X = 1550*1e-9; % Wavelength in m
    omega_2_X = (2*pi*f_var_CD_X).^2;

    z_CD_X= FiberLength_X ; % length of optical fiber in m*
    fib_TF_CD_X = exp(-1i*D_CD_X*lambda_CD_X^2*omega_2_X.*z_CD_X/(4*pi*c_CD_X));% Define the chromatic dispersion transfer function

    %Multiply signal spectrum and fiber trnasfer function
    E_OUT_transmitted_fspectrum_CD_X = fftshift(fft(E_out_CD_in_X))/length(E_out_CD_in_X);

    % This will give the spectrum of the signal at the output of fiber

    E_OUT_received_fspectrum_CD_X  = fib_TF_CD_X.'.*E_OUT_transmitted_fspectrum_CD_X;

    E_OUT_received_timedom_CD_X = ifft(ifftshift(E_OUT_received_fspectrum_CD_X))*length(E_OUT_received_fspectrum_CD_X);
    E_out_CD_X =E_OUT_received_timedom_CD_X;
    %  scatterplot(E_out_CD_X);
    %  title('With CD ');
    E_CD_OUT_X_at_fiberend = E_out_CD_X;

    %%  %% CD Ends for X

    %% Chromatic Dispersion for Y

    % scatterplot(E_OUT_X_at_fiberend);
    % title('Before CD Intro');
    E_out_CD_in_Y = E_Mixing_OUT_Y_at_fiberend;

    %  Fiber Optic Channel
    % Fiber Transerfunction.

    Fs_data_CD_Y = 2*Symbol_Rate;
    f_var_CD_Y =linspace(-Fs_data_CD_Y/2,+Fs_data_CD_Y/2, length(E_out_CD_in_Y));
    FiberLength_Y = 1*80e3;

    c_CD_Y = 3*1e8; % Speed of light in m/s
    D_CD_Y= 17*1*1e-6; % Dispersion has taken for wavelength 1550nm in ps/(km-nm) = 10*1e-6(in s/(m-m))
    lambda_CD_Y = 1550*1e-9; % Wavelength in m
    omega_2_Y = (2*pi*f_var_CD_Y).^2;

    z_CD_Y= FiberLength_Y ; % length of optical fiber in m*
    fib_TF_CD_Y = exp(-1i*D_CD_Y*lambda_CD_Y^2*omega_2_Y.*z_CD_Y/(4*pi*c_CD_Y));% Define the chromatic dispersion transfer function

    %Multiply signal spectrum and fiber trnasfer function
    E_OUT_transmitted_fspectrum_CD_Y = fftshift(fft(E_out_CD_in_Y))/length(E_out_CD_in_Y);

    % This will give the spectrum of the signal at the output of fiber

    E_OUT_received_fspectrum_CD_Y  = fib_TF_CD_Y.'.*E_OUT_transmitted_fspectrum_CD_Y;

    E_OUT_received_timedom_CD_Y = ifft(ifftshift(E_OUT_received_fspectrum_CD_Y))*length(E_OUT_received_fspectrum_CD_Y);
    E_out_CD_Y =E_OUT_received_timedom_CD_Y;
    %  scatterplot(E_out_CD_X);
    %  title('With CD ');
    E_CD_OUT_Y_at_fiberend = E_out_CD_Y;

    %%  %% CD Ends for Y





    %E_OUT_X_at_fiberend = E_OUT_transmitted_X;
    %%scatterplot(E_OUT_X_at_fiberend);


    %% X and Y polarized data Receieved at the receiver end
    E_OUT_received_X = E_CD_OUT_X_at_fiberend;%E_OUT_transmitted_X;  % X polarized Data
    E_OUT_received_Y = E_CD_OUT_Y_at_fiberend;%E_OUT_transmitted_Y;  % Y polarized Data
    %
    % % % To bypass CD
    % E_OUT_received_X = E_Mixing_OUT_X_at_fiberend;%E_OUT_transmitted_X;  % X polarized Data
    % E_OUT_received_Y = E_Mixing_OUT_Y_at_fiberend;%E_OUT_transmitted_Y;  % Y polarized Data


    %% 2X4 90 degree  optical  hybrids for X polarized data
    E1_X = (1 *E_OUT_received_X + 1i*E_LO_Laser_source_X)/(2); % Input to PD1 for X polarization
    E2_X = (1i*E_OUT_received_X + 1*E_LO_Laser_source_X)/(2);  % Input to PD2 for X polarization
    E3_X = (1i*E_OUT_received_X - 1i*E_LO_Laser_source_X)/(2); % Input to PD3 for X polarization
    E4_X = (-1*E_OUT_received_X - 1*E_LO_Laser_source_X)/(2);  % Input to PD4 for X polarization

    %% 2X4 90 degree  optical  hybrids for Y polarized data
    E1_Y = (1 *E_OUT_received_Y + 1i*E_LO_Laser_source_Y)/(2); % Input to PD1 for Y polarization
    E2_Y = (1i*E_OUT_received_Y + 1*E_LO_Laser_source_Y)/(2);  % Input to PD2 for Y polarization
    E3_Y = (1i*E_OUT_received_Y - 1i*E_LO_Laser_source_Y)/(2); % Input to PD3 for Y polarization
    E4_Y = (-1*E_OUT_received_Y - 1*E_LO_Laser_source_Y)/(2);  % Input to PD4 for Y polarization

    %% PhotoDetector Current (Balanced Detection) for X polarization
    Responsivity_X = 1;
    I_Inphase_X   = Responsivity_X*((E4_X.*conj(E4_X)) - (E3_X.*conj(E3_X))); % In phase current of X polarization
    I_Quadphase_X = Responsivity_X*((E1_X.*conj(E1_X)) - (E2_X.*conj(E2_X))); % Quadrature phase current of X polarization
    Received_signal_wo_Rx_noise_X = I_Inphase_X + exp(1i*pi/2)*I_Quadphase_X;

    %% PhotoDetector Current (Balanced Detection) for Y polarization
    Responsivity_Y = 1;
    I_Inphase_Y   = Responsivity_Y*((E4_Y.*conj(E4_Y)) - (E3_Y.*conj(E3_Y))); % In phase current of Y polarization
    I_Quadphase_Y = Responsivity_Y*((E1_Y.*conj(E1_Y)) - (E2_Y.*conj(E2_Y))); % Quadrature phase current of Y polarization
    Received_signal_wo_Rx_noise_Y = I_Inphase_Y + exp(1i*pi/2)*I_Quadphase_Y;

    %% Scatterplot
    %% scatterplot(Received_signal_wo_Rx_noise_X);% Scatterplot for XData
    % title("Scatterplot for X Polarized Received Data with Nt samples/symb");
    % scatterplot(Received_signal_wo_Rx_noise_Y);% Scatterplot for YData
    % title("Scatterplot for Y Polarized Received Data with Nt samples/symb");

    %% Filtering the X polarized data
    %I_out_signal_noise = Received_signal_wo_Rx_noise_X;
    %% Filter the signal to get the main lobe
    samp_freq = Symbol_Rate*Nt; % Sampling frequency = Symbol rate * Samples per symbol
    band_width = 1.2*Symbol_Rate; % Bandwidth is defined here
    init_lizing = zeros(length(Received_signal_wo_Rx_noise_X ),1);
    f = linspace(-samp_freq/2,samp_freq/2,length(Received_signal_wo_Rx_noise_X ));% Define the frequency axis
    init_lizing(abs(f)<band_width) = 1;% Initialize the value as 1 over the bandwidth
    freq_domain_signal = fftshift(fft(Received_signal_wo_Rx_noise_X )); % Signal in frequency domina
    filter_signal = freq_domain_signal.*init_lizing;% Multiply the signal in frequency domain with filter in frequency domain
    time_domain_signal = ifft(ifftshift(filter_signal)); % Convert back in to time domain.
    I_out_final_X = time_domain_signal;


    %% Filtering the Y polarized data
    %I_out_signal_noise = Received_signal_wo_Rx_noise_Y;
    %% Filter the signal to get the main lobe
    samp_freq = Symbol_Rate*Nt; % Sampling frequency = Symbol rate * Samples per symbol
    band_width = 1.2*Symbol_Rate; % Bandwidth is defined here
    init_lizing = zeros(length(Received_signal_wo_Rx_noise_Y ),1);
    f = linspace(-samp_freq/2,samp_freq/2,length(Received_signal_wo_Rx_noise_Y ));% Define the frequency axis
    init_lizing(abs(f)<band_width) = 1;% Initialize the value as 1 over the bandwidth
    freq_domain_signal = fftshift(fft(Received_signal_wo_Rx_noise_Y )); % Signal in frequency domina
    filter_signal = freq_domain_signal.*init_lizing;% Multiply the signal in frequency domain with filter in frequency domain
    time_domain_signal = ifft(ifftshift(filter_signal)); % Convert back in to time domain.
    I_out_final_Y = time_domain_signal;


    % % %% Sactterplot
    % % scatterplot(I_out_final_X);% Scatterplot for XData
    % % title("X Polarized Before Comp ="+num2str(OSNR_Array_dB(m))+"dB");
    % % scatterplot(I_out_final_Y);% Scatterplot for YData
    % % %title("Scatterplot for Y Polarized Received Data");
    % % title("Y Polarized Before comp ="+num2str(OSNR_Array_dB(m))+"dB");





    %% Chormatic Dispersion Compensation for X

    %I_out_final = I_out_final_X;
    %%
    fib_TF_CD_Compen_X = exp(1i*D_CD_X*lambda_CD_X^2*omega_2_X.*z_CD_X/(4*pi*c_CD_X));% Define the chromatic dispersion transfer function

    %Multiply signal spectrum and fiber trnasfer function
    %Received_signal_fspectrum_CD_Compen = fftshift(fft(Received_signal))/length(Received_signal);
    Received_signal_fspectrum_CD_Compen_X = fftshift(fft(I_out_final_X))/length(I_out_final_X);

    % This will give the spectrum of the signal at the output of fiber

    E_OUT_received_fspectrum_CDCompen_X  = fib_TF_CD_Compen_X.'.*Received_signal_fspectrum_CD_Compen_X;

    E_OUT_received_timedom_CD_Compen_X = ifft(ifftshift(E_OUT_received_fspectrum_CDCompen_X))*length(E_OUT_received_fspectrum_CDCompen_X);
    Received_signal_X = E_OUT_received_timedom_CD_Compen_X;


    % scatterplot(Received_signal);
    % title("After Compensation with OSNR ="+num2str(OSNR_Array_dB(m))+"dB");
    I_CD_Comp_out_final_X_2 = Received_signal_X;

    %% CD Compensation Ends for X



    %% Chormatic Dispersion Compensation for X

    %I_out_final = I_out_final_Y;
    %%
    fib_TF_CD_Compen_Y = exp(1i*D_CD_Y*lambda_CD_Y^2*omega_2_Y.*z_CD_Y/(4*pi*c_CD_Y));% Define the chromatic dispersion transfer function

    %Multiply signal spectrum and fiber trnasfer function
    %Received_signal_fspectrum_CD_Compen = fftshift(fft(Received_signal))/length(Received_signal);
    Received_signal_fspectrum_CD_Compen_Y = fftshift(fft(I_out_final_Y))/length(I_out_final_Y);

    % This will give the spectrum of the signal at the output of fiber

    E_OUT_received_fspectrum_CDCompen_Y  = fib_TF_CD_Compen_Y.'.*Received_signal_fspectrum_CD_Compen_Y;

    E_OUT_received_timedom_CD_Compen_Y = ifft(ifftshift(E_OUT_received_fspectrum_CDCompen_Y))*length(E_OUT_received_fspectrum_CDCompen_Y);
    Received_signal_Y = E_OUT_received_timedom_CD_Compen_Y;


    % scatterplot(Received_signal);
    % title("After Compensation with OSNR ="+num2str(OSNR_Array_dB(m))+"dB");
    I_CD_Comp_out_final_Y_2 = Received_signal_Y;

    %% CD Compensation Ends for Y




    %
    % % To Bypass CD
    % I_CD_Comp_out_final_X_2 = I_out_final_X;
    % I_CD_Comp_out_final_Y_2 = I_out_final_Y;


    %% CMA Start
    % Polarization demultiplexing

    len_pol_data = length(I_CD_Comp_out_final_X_2); % Length of the input data to CMA Algorithm
    mu_CMA = 1e-4; % Define the step size

    %Initialize the tap weights
    pxx_demux = 1;
    pxy_demux = 0;
    pyx_demux = 0;
    pyy_demux = 1;
    M=2;
    S=qammod(0:1:2^M-1,2^M);
    % scatterplot(S)
    Ideal_rms=rms(S);


    % Normalize the X and Y polarized input data
    Input_X = I_CD_Comp_out_final_X_2.*Ideal_rms/rms(I_CD_Comp_out_final_X_2);%sqrt(1/(sum(abs(I_out_final_X).^2)/len_pol_data));
    Input_Y = I_CD_Comp_out_final_Y_2.*Ideal_rms/rms(I_CD_Comp_out_final_Y_2);%sqrt(1/(sum(abs(I_out_final_Y).^2)/len_pol_data));
    % [xpol_samples,ypol_samples]=poldemux_rd_multitap_cma_qam_M_ary( Input_X, Input_Y,2,S,1,1e-3);
    % scatterplot(xpol_samples(1:4:end))
    % return
    R = 1; % Reference radius

    %  DMUX_F1 = zeros(1,len_pol_data);
    %  DMUX_F2 = zeros(1,len_pol_data);
    %  DMUX_F3 = zeros(1,len_pol_data);
    %  DMUX_F4 = zeros(1,len_pol_data);

    %% To plot filter coefficients
    col_ind=0;
    C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]} ;% Cell array of colros.
    % figure;
    % title("PLot for different step sizes ");
    %for mu_CMA = [1e-4 4e-4 8e-4 10e-4]
    for mu_CMA = [1e-4 ]
        col_ind = col_ind+1;
        for arr_ind = 1:len_pol_data

            R1=Input_X(arr_ind);
            R2=Input_Y(arr_ind);


            x_cap(arr_ind)=pxx_demux.'*R1+pxy_demux.'*R2;
            y_cap(arr_ind)=pyx_demux.'*R1+pyy_demux.'*R2;

            r2_x=abs(x_cap(arr_ind))^2;
            r2_y=abs(y_cap(arr_ind))^2;


            e_x = x_cap(arr_ind)*(R-r2_x);
            e_y = y_cap(arr_ind)*(R-r2_y);


            pxx_demux = pxx_demux+mu_CMA*e_x*conj(R1);
            pxy_demux = pxy_demux+mu_CMA*e_x*conj(R2);
            pyy_demux = conj(pxx_demux);
            pyx_demux = -conj(pxy_demux);
            DMUX_F1(arr_ind) = pxx_demux;
            DMUX_F2(arr_ind) = pxy_demux;
            DMUX_F3(arr_ind) = pyx_demux;
            DMUX_F4(arr_ind) = pyy_demux;


        end
        % plot(10*log10((abs(DMUX_F2)).^2),C{col_ind});
        % hold on;
        %Enable the following 4 lines also to plot filtercoefficients
        % pxx_demux = 1;
        % pxy_demux = 0;
        % pyx_demux = 0;
        % pyy_demux = 1;


    end  %% To plot for diffrent stpsiz
    % hold off;

    %  X_Rx_demux = pxx_demux*Input_X + pxy_demux*Input_Y;
    %  Y_Rx_demux = pyx_demux*Input_X + pyy_demux*Input_Y;


    % X-pol
    X_Rx_demux= conv(Input_X,pxx_demux,'same')+conv(Input_Y,pxy_demux,'same');
    %scatterplot(X_Rx_demux);
    % Y-pol
    Y_Rx_demux=conv(Input_X,pyx_demux,'same')+conv(Input_Y,pyy_demux,'same');

    X_Rx_demux_OUT = X_Rx_demux;
    Y_Rx_demux_OUT = Y_Rx_demux;

    %% CMA End
    %
    %
    %% Frequency Offset Compensation for X
    Rec_Samples = X_Rx_demux_OUT; % Take Rec_Samples and give it to downconvrt it will show circl for freq offset
    Fs = Symbol_Rate*Symbol_repeat_X;
    est = zeros(1,20);
    Power_factor  = 4;%Power_factor_array;

    Nsymb=Total_no_symbols_X;
    fval = linspace(-Fs/2,Fs/2,length(Rec_Samples));%fftshift( -Nt/2:1/Nsymb:(Nt/2)-(1/Nsymb) );
    Data_len = length(Rec_Samples);
    spectrum = (fftshift(abs(fft(Rec_Samples.^Power_factor-mean(Rec_Samples.^Power_factor)))));
    [max_val, max_loc] = max(spectrum);
    FO_Error = (1/Power_factor)*(max_loc-1-Data_len/2)/Data_len;
    FO_estimate_inHZ = (FO_Error)*Symbol_Rate*Nt;
    Offset_Error(m) = abs(detunig_freq - abs(FO_estimate_inHZ));
    Rec_Samples_1 = Rec_Samples.*(exp(-1i*2*pi*FO_estimate_inHZ*t)).';
    FO_compensated = Rec_Samples_1;
    I_out_final_FOcomp_X = FO_compensated;

    %X_Rx_demux = I_out_final_FOcomp_X;
    I_out_final_X = I_out_final_FOcomp_X;
    %% Frequency Offset Compensation for X Ends



    %% Frequency Offset Compensation for Y
    Rec_Samples_Y = Y_Rx_demux_OUT; % Take Rec_Samples and give it to downconvrt it will show circl for freq offset
    Fs = Symbol_Rate*Symbol_repeat_X;
    est = zeros(1,20);
    Power_factor  = 4;%Power_factor_array;

    Nsymb=Total_no_symbols_Y;
    fval = linspace(-Fs/2,Fs/2,length(Rec_Samples_Y));%fftshift( -Nt/2:1/Nsymb:(Nt/2)-(1/Nsymb) );
    Data_len = length(Rec_Samples_Y);
    spectrum = (fftshift(abs(fft(Rec_Samples_Y.^Power_factor-mean(Rec_Samples_Y.^Power_factor)))));
    [max_val, max_loc] = max(spectrum);
    FO_Error = (1/Power_factor)*(max_loc-1-Data_len/2)/Data_len;
    FO_estimate_inHZ = (FO_Error)*Symbol_Rate*Nt;
    Offset_Error(m) = abs(detunig_freq - abs(FO_estimate_inHZ));
    Rec_Samples_1 = Rec_Samples_Y.*(exp(-1i*2*pi*FO_estimate_inHZ*t)).';
    FO_compensated = Rec_Samples_1;
    I_out_final_FOcomp_Y = FO_compensated;

    %X_Rx_demux = I_out_final_FOcomp_X;
    I_out_final_Y = I_out_final_FOcomp_Y;
    %% Frequency Offset Compensation  for Y Ends
    X_OUT_FO = I_out_final_X;
    Y_OUT_FO = I_out_final_Y;
    X_Rx_demux = X_OUT_FO;
    Y_Rx_demux = Y_OUT_FO;
    %


    %  %% Sactterplot
    % scatterplot(X_Rx_demux);% Scatterplot for XData
    % title("X Polarized after Comp ="+num2str(OSNR_Array_dB(m))+"dB");
    % scatterplot(Y_Rx_demux);% Scatterplot for YData
    % %title("Scatterplot for Y Polarized Received Data");
    % title("Y Polarized after comp ="+num2str(OSNR_Array_dB(m))+"dB");



    %X_Rx_demux = Received_signal_X;
    %% Take one sample per symbol and create such Nt arrays
    Recieved_signal_Array_X = zeros(Total_no_symbols_X,Nt);% Initalize the array to store Nt sets of data for X polarization
    Recieved_signal_Array_Y = zeros(Total_no_symbols_X,Nt);% Initalize the array to store Nt sets of data for Y polarization
    for count=1:1:Nt
        %     Recieved_signal_Array_X(:,i) = I_out_final_X(i:Nt:end);% Take each Nt sample and create X polarized Data
        %     Recieved_signal_Array_Y(:,i) = I_out_final_Y(i:Nt:end);% Take each Nt sample and create Y polarized Data

        Recieved_signal_Array_X(:,count) = X_Rx_demux(count:Nt:end);% Take each Nt sample and create X polarized Data
        Recieved_signal_Array_Y(:,count) = Y_Rx_demux(count:Nt:end);% Take each Nt sample and create Y polarized Data

    end

    %% Compute the Standard deviation for each data set(in each column)
    std_dev_X = std(Recieved_signal_Array_X);
    std_dev_Y = std(Recieved_signal_Array_Y);

    %% Find index of the best data set
    index_X = find(max(std_dev_X) == std_dev_X);
    index_Y = find(max(std_dev_Y) == std_dev_Y);

    %% Find the best data set from the array
    Received_signal_X = Recieved_signal_Array_X(:,index_X(1)); % The best dataset (out of Nt data set) for X polarization
    Received_signal_Y = Recieved_signal_Array_Y(:,index_Y(1)); % The best dataset (out of Nt data set) for Y polarization


    % %% Sactterplot
    % scatterplot(Received_signal_X);% Scatterplot for XData
    %  title("X Polarized Received DataOSNR ="+num2str(OSNR_Array_dB(m))+"dB");
    %  scatterplot(Received_signal_Y);% Scatterplot for YData
    %  title("Y Polarized Received DataOSNR ="+num2str(OSNR_Array_dB(m))+"dB");



    %% Correction for Rotation of X polarized data

    div_result_X = Received_signal_X(1:100)./qam_signal_X(1:100);
    angle_values_X = atan(imag(div_result_X)./real(div_result_X));
    mean_ang_val_X = mean(angle_values_X);

    Received_signal_X_corrected = Received_signal_X.*exp(-1i*mean_ang_val_X);
    %
    % scatterplot(Received_signal_X_corrected);
    % title("After rotational correction for X");


    %% Correction for Rotation of Y polarized data

    div_result_Y = Received_signal_Y(1:100)./qam_signal_Y(1:100);
    angle_values_Y = atan(imag(div_result_Y)./real(div_result_Y));
    mean_ang_val_Y = mean(angle_values_Y);

    Received_signal_Y_corrected = Received_signal_Y.*exp(-1i*mean_ang_val_Y);
    %
    % scatterplot(Received_signal_Y_corrected);
    % title("After rotational correction for Y");
    % %



    %% Phase compensation for X polarized data
    % Ideal_symb = Sequence_X; %[1+1i 1-1i  -1-1i -1+1i ];
    M=2;
    S=qammod(0:2^M-1,2^M);

    %% Phase noise Compensation Single tapped adaptive filter
    mu = 0.01;           % step size
    %mu = 0.001;           % step size
    w = 1;               %Initializing the filter tap
    L = length(Received_signal_X_corrected);
    for ind = 1:L
        input = Received_signal_X_corrected(ind);
        y = w.*input;                    % Filter output
        [~, dec_index] = min(abs(S-y));
        output =  S(dec_index);
        e_n = output-y;                       % error estimation
        w = w + mu.*e_n.*conj(input)./(abs(input).^2); % Filter weight updation
        I_out_signal_final_X(ind) = y;
    end
    Received_signal_final_X = I_out_signal_final_X.';

    Received_signal_X_corrected = Received_signal_final_X;


    % Phase noise compensation ends for X


    %% Phase compensation for Y polarized data
    % Ideal_symb = Sequence_X; %[1+1i 1-1i  -1-1i -1+1i ];
    M=2;
    S=qammod(0:2^M-1,2^M);

    %% Phase noise Compensation Single tapped adaptive filter
    mu = 0.01;           % step size
    %mu = 0.001;           % step size
    w = 1;               %Initializing the filter tap
    L = length(Received_signal_Y_corrected);
    for ind = 1:L
        input = Received_signal_Y_corrected(ind);
        y = w.*input;                    % Filter output
        [~, dec_index] = min(abs(S-y));
        output =  S(dec_index);
        e_n = output-y;                       % error estimation
        w = w + mu.*e_n.*conj(input)./(abs(input).^2); % Filter weight updation
        I_out_signal_final_Y(ind) = y;
    end
    Received_signal_final_Y = I_out_signal_final_Y.';

    Received_signal_Y_corrected = Received_signal_final_Y;


    % Phase noise compensation ends






    %% Demodulation of X polarization
    Demodulated_Received_signal_X = qamdemod(Received_signal_X_corrected,4,symb_map);% Demodulate the best matched received signal
    Received_signal_in_Binarysymbols_X = de2bi(Demodulated_Received_signal_X,'left-msb');% Convert the above sequence from decimal to binary
    Received_signal_final_X = zeros(2*Total_no_symbols_X,1);%Initialize array to store the received bits
    Received_signal_final_X(1:2:end) = Received_signal_in_Binarysymbols_X(:,1);%Create bit sequence from the symbols(QPSK has 2 bits per symbol)
    Received_signal_final_X(2:2:end) = Received_signal_in_Binarysymbols_X(:,2);





    %% Computation of BER for X polarized data

    Number_of_errorbits_X  =  sum(Received_signal_final_X.' ~= Sequence_X);
    No_of_transmitted_bits_X = length(Sequence_X);
    BER_X(m) = Number_of_errorbits_X/No_of_transmitted_bits_X;

    %% Demodulation of Y polarization
    Demodulated_Received_signal_Y = qamdemod(Received_signal_Y_corrected,4,symb_map);% Demodulate the best matched received signal
    Received_signal_in_Binarysymbols_Y = de2bi(Demodulated_Received_signal_Y,'left-msb');% Convert the above sequence from decimal to binary
    %Create bit sequence from the symbols(QPSK has 2 bits per symbol)
    Received_signal_final_Y = zeros(2*Total_no_symbols_Y,1);%Initialize array to store the received bits
    Received_signal_final_Y(1:2:end) = Received_signal_in_Binarysymbols_Y(:,1); % Store the first column elements to odd positions in the array
    Received_signal_final_Y(2:2:end) = Received_signal_in_Binarysymbols_Y(:,2); % Store the Second column elements to even positions in the array
    % %



    %% Computation of BER for Y polarized data
    Number_of_errorbits_Y  =  sum(Received_signal_final_Y.' ~= Sequence_Y);
    No_of_transmitted_bits_Y = length(Sequence_Y);
    BER_Y(m) = Number_of_errorbits_Y/No_of_transmitted_bits_Y;

end


%% PLot the graph between OSNR vs BER for X polarized data
figure();
semilogy(OSNR_Array_dB,BER_X,'-or','lineWidth',4);
grid minor;
xlabel('OSNR in dB');% Label the X axis as time in seconds
ylabel('BER');% Label the y axis as Amplitude
title("OSNR vs BER for X polarized data");% Give the title for figure
set(gca,'fontsize',12,'fontweight','bold');% Define the font settings

hold on;

%% Theoretical Calculation of BER and plot BER vs OSNR
OSNR_linear  = 10.^(.1*OSNR_Array_dB);
bitspersymbol = 2;
k = bitspersymbol;
B_ref = 12.5e9;
p = 2;
Rs = Symbol_Rate;
Eb_N0 = OSNR_linear*2*B_ref/(p*Rs*k);
M_ary = 4;
BER_theoretical = (1-(1/sqrt(M_ary)))*erfc(sqrt(3*bitspersymbol*Eb_N0/((M_ary-1)*2)));
semilogy(OSNR_Array_dB,BER_theoretical,'--k','lineWidth',4);
%%

legend("Simulation Curve for BER","Theoretical Curve for BER");
hold off;



%% PLot the graph between OSNR vs BER for Y polarized data
figure();
semilogy(OSNR_Array_dB,BER_Y,'-or','lineWidth',4);
grid minor;
xlabel('OSNR in dB');% Label the X axis as time in seconds
ylabel('BER');% Label the y axis as Amplitude
title("OSNR vs BER for Y polarized data ");% Give the title for figure
set(gca,'fontsize',12,'fontweight','bold');% Define the font settings

hold on;

%% Theoretical Calculation of BER and plot BER vs OSNR
OSNR_linear  = 10.^(.1*OSNR_Array_dB);
bitspersymbol = 2;
k = bitspersymbol;
B_ref = 12.5e9;
p = 2;
Rs = Symbol_Rate;
Eb_N0 = OSNR_linear*2*B_ref/(p*Rs*k);
M_ary = 4;
BER_theoretical = (1-(1/sqrt(M_ary)))*erfc(sqrt(3*bitspersymbol*Eb_N0/((M_ary-1)*2)));
semilogy(OSNR_Array_dB,BER_theoretical,'--k','lineWidth',4);
%%

legend("Simulation Curve for BER ","Theoretical Curve for BER");
hold off;

%PLot filter coefficients
%
% figure
% plot(10*log10((abs(DMUX_F1)).^2));
% figure
% plot(10*log10((abs(DMUX_F2)).^2))
% figure
% plot(10*log10((abs(DMUX_F3)).^2));
% figure
% plot(10*log10((abs(DMUX_F4)).^2));