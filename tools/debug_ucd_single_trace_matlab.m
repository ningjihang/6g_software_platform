clear; clc;

base_dir = 'C:\Users\27147\Documents\xwechat_files\wxid_yr6ld27a33lm22_1ffb\msg\file\2026-05';
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

symbol = NR_modulation(akk,Qm);
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

disp('akk =');
disp(akk.');
disp('symbol =');
disp(symbol.');
disp('P =');
disp(P);
disp('W =');
disp(W);
disp('snrOut =');
disp(snrOut);
disp('B =');
disp(B);
disp('B1 =');
disp(diag(diag(B)) \ B);
disp('dpCodData =');
disp(dpCodData);
disp('y_UCD =');
disp(y_UCD);
disp('yEqu =');
disp(yEqu);
disp('y1 =');
disp(y1);
disp('dpRxData =');
disp(dpRxData);
disp('soft_bits =');
disp(reshape(soft_bits, Qm, []));
disp('LLR =');
disp(LLR);
disp('akkn =');
disp(akkn);
disp('bit_losses =');
disp(bit_losses);
disp('MI =');
disp(MI);
