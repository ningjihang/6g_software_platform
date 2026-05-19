clear; clc;

base_dir = 'C:\Users\27147\Documents\xwechat_files\wxid_yr6ld27a33lm22_1ffb\msg\file\2026-05';
repo_dir = 'C:\Users\27147\Desktop\mix_precode';
addpath(base_dir);

channelH = [ ...
    1.0 + 0.0j,  0.2 + 0.1j, -0.1 + 0.0j,  0.0 + 0.0j; ...
    0.0 + 0.0j,  0.9 + 0.0j,  0.2 + 0.0j, -0.1 + 0.0j; ...
    0.0 + 0.0j,  0.0 + 0.0j,  0.8 + 0.0j,  0.1 + 0.0j; ...
    0.0 + 0.0j,  0.0 + 0.0j,  0.0 + 0.0j,  0.7 + 0.0j ...
];

Nr = 4;
Qm = 6;
snr_per_stream = 10.0;
alpha = 1.0 / snr_per_stream;

akk = [ ...
    1;1;1;1;1;1; ...
    1;1;1;0;0;0; ...
    1;1;0;1;0;0; ...
    1;1;0;0;1;1 ...
];

[U,S,V] = svd(channelH, "econ");
[P,W,snrOut] = ucd(U,S,V,Nr,alpha,0);

% Manual Fudan 64QAM mapping, avoiding deprecated comm.RectangularQAMModulator
a = 1/sqrt(42);
axis_bits = [
    1 1 1;
    1 1 0;
    1 0 0;
    1 0 1;
    0 0 1;
    0 0 0;
    0 1 0;
    0 1 1
];
axis_levels = [
   -7;
   -5;
   -3;
   -1;
    1;
    3;
    5;
    7
] * a;

num_sym = length(akk) / Qm;
bit_mat = reshape(akk, Qm, num_sym).';
symbol = zeros(num_sym,1);
for k = 1:num_sym
    real_bits = bit_mat(k,1:3);
    imag_bits = bit_mat(k,4:6);
    real_idx = find(ismember(axis_bits, real_bits, 'rows'));
    imag_idx = find(ismember(axis_bits, imag_bits, 'rows'));
    symbol(k) = axis_levels(real_idx) + 1j * axis_levels(imag_idx);
end

xk = reshape(symbol, Nr, []);
[dpCodData,B,M,d] = dp_transmit(channelH,P,W,xk,Qm,Nr);
noise = zeros(Nr, size(xk,2));
y_UCD = channelH * P * dpCodData + noise;
yEqu = W' * y_UCD;
y1 = diag(diag(B)) \ yEqu;
dpRxData = real(y1) - floor((real(y1) + M*d/2)/(M*d))*M*d + ...
           1j*(imag(y1) - floor((imag(y1) + M*d/2)/(M*d))*M*d);
symbolHat_UCDDP = reshape(dpRxData, [], 1);
soft_bits = softDemod_dp_v2(Qm, symbolHat_UCDDP);
LLR = reshape(-soft_bits, Nr*Qm,[]).*snrOut;
akkn = reshape(1 - 2*akk, Nr*Qm, []);
bit_losses = log2(1 + exp(akkn .* LLR));
MI = Qm*Nr - sum(mean(bit_losses, 2));

save(fullfile(repo_dir, 'tools', 'ucd_single_trace_from_matlab.mat'), ...
    'akk', 'symbol', 'P', 'W', 'snrOut', 'B', 'dpCodData', ...
    'y_UCD', 'yEqu', 'y1', 'dpRxData', 'soft_bits', 'LLR', ...
    'akkn', 'bit_losses', 'MI');

disp('Saved MATLAB trace to:');
disp(fullfile(repo_dir, 'tools', 'ucd_single_trace_from_matlab.mat'));
