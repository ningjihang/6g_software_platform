import numpy as np
from scipy.linalg import svd
from full_digital_mu import FullyDigitalMuMimoBicmEnvironment

print("Creating FD environment...")
env = FullyDigitalMuMimoBicmEnvironment(
    num_users=4,
    num_tx_antennas=16,
    num_rx_antennas=4,
    num_streams_per_user=4,
    channel_type='cdl-a',
)

print("Generating channels...")
channels = env.generate_user_channels()
print(f'Channels shape: {channels.shape}')

snr_db = 10.0
snr_linear = 10 ** (snr_db / 10)
print(f'SNR: {snr_db} dB (linear: {snr_linear})')

print('\n=== Testing GMD ===')
print("Building structured precoder with GMD...")
f_bb, effective_channels, bd_null_bases = env.build_structured_precoder(
    user_channels=channels,
    snr_per_stream=snr_linear,
    strategy='gmd'
)
print(f'f_bb shape: {f_bb.shape}')
print(f'Effective channels count: {len(effective_channels)}')
print(f'BD null bases count: {len(bd_null_bases)}')

user_idx = 0
f_rf = np.eye(16)
desired_block = effective_channels[user_idx] @ (bd_null_bases[user_idx] @ f_bb[:, user_idx * 4 : (user_idx + 1) * 4])
print(f'Desired block shape: {desired_block.shape}')

_, r_factor = np.linalg.qr(desired_block, mode='reduced')
r_factor = r_factor[:4, :4]
diagonal = np.diag(r_factor)
print(f'R factor diagonal: {diagonal}')
print(f'Magnitudes: {np.abs(diagonal)}')

stream_snrs_linear = (np.abs(diagonal) ** 2 * snr_linear)
print(f'Stream SNRs (linear): {stream_snrs_linear}')
print(f'Stream SNRs (dB): {10 * np.log10(stream_snrs_linear)}')

print("\n=== Testing SVD ===")
print("Building structured precoder with SVD...")
f_bb_svd, effective_channels_svd, bd_null_bases_svd = env.build_structured_precoder(
    user_channels=channels,
    snr_per_stream=snr_linear,
    strategy='svd'
)

desired_block_svd = effective_channels_svd[user_idx] @ (bd_null_bases_svd[user_idx] @ f_bb_svd[:, user_idx * 4 : (user_idx + 1) * 4])
_, r_factor_svd = np.linalg.qr(desired_block_svd, mode='reduced')
r_factor_svd = r_factor_svd[:4, :4]
diagonal_svd = np.diag(r_factor_svd)
stream_snrs_linear_svd = (np.abs(diagonal_svd) ** 2 * snr_linear)
print(f'SVD Stream SNRs (dB): {10 * np.log10(stream_snrs_linear_svd)}')

print("\n=== Comparing GMD vs SVD ===")
print(f"GMD SNRs are uniform: {np.std(10 * np.log10(stream_snrs_linear)) < 0.1}")
print(f"SVD SNRs vary: {np.std(10 * np.log10(stream_snrs_linear_svd)) > 1.0}")
