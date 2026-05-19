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

L = 4;
snr_per_stream = 10.0;
alpha = 1.0 / snr_per_stream;

[U,S,V] = svd(channelH, "econ");
[P,W,snrOut] = ucd(U,S,V,L,alpha,0);

K = size(V,2);
ldpow = ones(L,1);
Sigm = zeros(L,1);
Sigm(1:K) = sqrt(ldpow(1:K)).*diag(S);
SigmAlpha = sqrt(Sigm(1:K).^2+alpha);
[Wraw,R,Praw] = gmd( ...
    [U(:,1:K)*diag(Sigm(1:K)./SigmAlpha(1:K)),zeros(size(U,1),L-K)], ...
    diag(SigmAlpha), ...
    [V*diag(sqrt(ldpow(1:K))),zeros(size(V,1),L-K)] ...
);
Wnorm = Wraw/diag(diag(R));
B = Wnorm' * channelH * Praw;

save(fullfile(repo_dir, 'tools', 'ucd_design_trace_from_matlab.mat'), ...
    'channelH', 'U', 'S', 'V', 'P', 'W', 'snrOut', 'Wraw', 'R', 'Praw', 'Wnorm', 'B', 'alpha');

disp('Saved MATLAB design trace to:');
disp(fullfile(repo_dir, 'tools', 'ucd_design_trace_from_matlab.mat'));
