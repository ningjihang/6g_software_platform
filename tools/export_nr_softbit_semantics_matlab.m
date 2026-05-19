clear; clc;

repo_dir = 'C:\Users\27147\Desktop\mix_precode';
out_dir = fullfile(repo_dir, 'tools', 'nr_mapping_tables');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

mapping_mat = fullfile(repo_dir, 'tools', 'nr_modulation_mappings_from_matlab.mat');
if ~exist(mapping_mat, 'file')
    error('Mapping MAT not found. Run tools/export_nr_modulation_mappings_matlab.m first.');
end

load(mapping_mat, ...
    'bits_qpsk', 'symbols_qpsk', ...
    'bits_16qam', 'symbols_16qam', ...
    'bits_64qam', 'symbols_64qam', ...
    'bits_256qam', 'symbols_256qam');

% QPSK: export hard semantic truth directly from the protocol mapping.
bits_in = bits_qpsk;
symbols_in = symbols_qpsk(:);
label = 'QPSK_soft_semantics';
hard_bits_from_soft = bits_in;

save(fullfile(out_dir, sprintf('%s.mat', label)), ...
    'bits_in', 'symbols_in', 'hard_bits_from_soft');

bits_str = strings(size(bits_in,1),1);
hard_bits_str = strings(size(bits_in,1),1);
for row_idx = 1:size(bits_in,1)
    bits_str(row_idx) = string(sprintf('%d', bits_in(row_idx, :)));
    hard_bits_str(row_idx) = string(sprintf('%d', hard_bits_from_soft(row_idx, :)));
end
T = table( ...
    (0:size(bits_in,1)-1).', ...
    bits_str, ...
    real(symbols_in), ...
    imag(symbols_in), ...
    hard_bits_str, ...
    'VariableNames', {'row_index', 'bits', 'real_part', 'imag_part', 'hard_bits_from_soft'} ...
);
writetable(T, fullfile(out_dir, sprintf('%s.csv', label)));

% 16QAM / 64QAM: use the existing Fudan soft demodulator if available.
for Qm = [4, 6]
    switch Qm
        case 4
            bits_in = bits_16qam;
            symbols_in = symbols_16qam(:);
            label = '16QAM_soft_semantics';
        case 6
            bits_in = bits_64qam;
            symbols_in = symbols_64qam(:);
            label = '64QAM_soft_semantics';
    end

    soft_bits = softDemod_dp_v2(Qm, symbols_in);
    soft_bits = reshape(soft_bits, Qm, []).';
    hard_bits_from_soft = (soft_bits < 0);

    save(fullfile(out_dir, sprintf('%s.mat', label)), ...
        'bits_in', 'symbols_in', 'soft_bits', 'hard_bits_from_soft');

    bits_str = strings(size(bits_in,1),1);
    hard_bits_str = strings(size(bits_in,1),1);
    for row_idx = 1:size(bits_in,1)
        bits_str(row_idx) = string(sprintf('%d', bits_in(row_idx, :)));
        hard_bits_str(row_idx) = string(sprintf('%d', hard_bits_from_soft(row_idx, :)));
    end
    T = table( ...
        (0:size(bits_in,1)-1).', ...
        bits_str, ...
        real(symbols_in), ...
        imag(symbols_in), ...
        hard_bits_str, ...
        'VariableNames', {'row_index', 'bits', 'real_part', 'imag_part', 'hard_bits_from_soft'} ...
    );
    writetable(T, fullfile(out_dir, sprintf('%s.csv', label)));
end

% 256QAM: export mapping truth only. No Fudan soft demodulator is provided in codebase.
bits_in = bits_256qam;
symbols_in = symbols_256qam(:);
label = '256QAM_soft_semantics';
hard_bits_from_soft = bits_in;

save(fullfile(out_dir, sprintf('%s.mat', label)), ...
    'bits_in', 'symbols_in', 'hard_bits_from_soft');

bits_str = strings(size(bits_in,1),1);
hard_bits_str = strings(size(bits_in,1),1);
for row_idx = 1:size(bits_in,1)
    bits_str(row_idx) = string(sprintf('%d', bits_in(row_idx, :)));
    hard_bits_str(row_idx) = string(sprintf('%d', hard_bits_from_soft(row_idx, :)));
end
T = table( ...
    (0:size(bits_in,1)-1).', ...
    bits_str, ...
    real(symbols_in), ...
    imag(symbols_in), ...
    hard_bits_str, ...
    'VariableNames', {'row_index', 'bits', 'real_part', 'imag_part', 'hard_bits_from_soft'} ...
);
writetable(T, fullfile(out_dir, sprintf('%s.csv', label)));

fprintf('\nSaved soft-bit semantic tables to:\n%s\n', out_dir);
