clear; clc;

repo_dir = 'C:\Users\27147\Desktop\mix_precode';
out_dir = fullfile(repo_dir, 'tools', 'nr_mapping_tables');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

qam_bits_list = [2, 4, 6, 8]; % QPSK, 16QAM, 64QAM, 256QAM

for idx_cfg = 1:numel(qam_bits_list)
    Qm = qam_bits_list(idx_cfg);
    M = 2^Qm;
    bits_in = dec2bin(0:M-1, Qm) - '0';   % M x Qm
    bit_index = bits_in * (2.^(Qm-1:-1:0)).';

    switch Qm
        case 2
            custom_map = [0 2 3 1];
            mapped_index = custom_map(bit_index + 1).';
            symbols_out = pskmod(mapped_index, 4, pi/4);
            label = 'QPSK_custom';
        case 4
            custom_map = [11 10 14 15 9 8 12 13 1 0 4 5 3 2 6 7];
            mapped_index = custom_map(bit_index + 1).';
            symbols_out = qammod(mapped_index, 16, 'gray', ...
                'InputType', 'integer', 'UnitAveragePower', true);
            label = '16QAM_custom';

        case 6
            custom_map = [ ...
                47 46 42 43 59 58 62 63 45 44 40 41 57 56 60 61 ...
                37 36 32 33 49 48 52 53 39 38 34 35 51 50 54 55 ...
                 7  6  2  3 19 18 22 23  5  4  0  1 17 16 20 21 ...
                13 12  8  9 25 24 28 29 15 14 10 11 27 26 30 31];
            mapped_index = custom_map(bit_index + 1).';
            symbols_out = qammod(mapped_index, 64, 'gray', ...
                'InputType', 'integer', 'UnitAveragePower', true);
            label = '64QAM_custom';

        case 8
            bit_stream = reshape(bits_in.', [], 1);
            symbols_out = qammod(bit_stream, 256, 'gray', ...
                'InputType', 'bit', 'UnitAveragePower', true);
            mapped_index = zeros(M, 1);
            label = '256QAM_gray_default';

        otherwise
            error('Unsupported Qm = %d', Qm);
    end

    fprintf('\n================ %s ================\n', label);
    fprintf(' row | bits | map_idx | symbol\n');
    fprintf('-----------------------------------------------\n');
    for k = 1:M
        bit_str = sprintf('%d', bits_in(k, :));
        fprintf('%4d | %s | %7d | %+0.6f %+0.6fj\n', ...
            k-1, bit_str, mapped_index(k), real(symbols_out(k)), imag(symbols_out(k)));
    end

    T = table( ...
        (0:M-1).', ...
        string(bits_in), ...
        mapped_index, ...
        real(symbols_out(:)), ...
        imag(symbols_out(:)), ...
        'VariableNames', {'row_index', 'bits', 'mapped_index', 'real_part', 'imag_part'} ...
    );
    out_csv = fullfile(out_dir, sprintf('%s.csv', label));
    writetable(T, out_csv);

    save(fullfile(out_dir, sprintf('%s.mat', label)), ...
        'bits_in', 'bit_index', 'mapped_index', 'symbols_out');
end

fprintf('\nSaved mapping tables to:\n%s\n', out_dir);
