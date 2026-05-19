clear; clc;

repo_dir = 'C:\Users\27147\Desktop\mix_precode';

output_path = fullfile(repo_dir, 'tools', 'nr_modulation_mappings_from_matlab.mat');

qam_orders = [2, 4, 6, 8]; % bits per symbol: QPSK, 16QAM, 64QAM, 256QAM

for idx = 1:numel(qam_orders)
    Qm = qam_orders(idx);
    M = 2^Qm;

    bits_in = dec2bin(0:M-1, Qm) - '0';     % M x Qm
    idx_in = bits_in * (2.^(Qm-1:-1:0)).';

    switch Qm
        case 2
            custom_map = [0 2 3 1];
            symbol_out = pskmod(custom_map(idx_in + 1), 4, pi/4);
        case 4
            custom_map = [11 10 14 15 9 8 12 13 1 0 4 5 3 2 6 7];
            symbol_out = qammod(custom_map(idx_in + 1), 16, 'gray', ...
                'InputType', 'integer', 'UnitAveragePower', true);
        case 6
            custom_map = [ ...
                47 46 42 43 59 58 62 63 45 44 40 41 57 56 60 61 ...
                37 36 32 33 49 48 52 53 39 38 34 35 51 50 54 55 ...
                 7  6  2  3 19 18 22 23  5  4  0  1 17 16 20 21 ...
                13 12  8  9 25 24 28 29 15 14 10 11 27 26 30 31];
            symbol_out = qammod(custom_map(idx_in + 1), 64, 'gray', ...
                'InputType', 'integer', 'UnitAveragePower', true);
        case 8
            bit_stream = reshape(bits_in.', [], 1);
            symbol_out = qammod(bit_stream, 256, 'gray', ...
                'InputType', 'bit', 'UnitAveragePower', true);
        otherwise
            error('Unsupported Qm=%d', Qm);
    end

    switch Qm
        case 2
            bits_qpsk = bits_in;
            symbols_qpsk = symbol_out(:);
        case 4
            bits_16qam = bits_in;
            symbols_16qam = symbol_out(:);
        case 6
            bits_64qam = bits_in;
            symbols_64qam = symbol_out(:);
        case 8
            bits_256qam = bits_in;
            symbols_256qam = symbol_out(:);
    end
end

save(output_path, ...
    'bits_qpsk', 'symbols_qpsk', ...
    'bits_16qam', 'symbols_16qam', ...
    'bits_64qam', 'symbols_64qam', ...
    'bits_256qam', 'symbols_256qam');

disp('Saved MATLAB NR_modulation mappings to:');
disp(output_path);
