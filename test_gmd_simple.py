import numpy as np
from scipy.linalg import svd
import math

print("=== Testing GMD Logic ===")
print()

np.random.seed(42)
H_k = np.random.randn(4, 16) + 1j * np.random.randn(4, 16)
_, s, _ = svd(H_k, full_matrices=False)
s = s[:4]

print(f"Channel singular values: {s}")
print()

snr_db = 10.0
snr_linear = 10 ** (snr_db / 10)
print(f"System SNR: {snr_db} dB")
print()

print("=== SVD Precoding ===")
stream_snrs_linear_svd = [(si**2) * snr_linear for si in s]
stream_snrs_db_svd = [10 * math.log10(snr) for snr in stream_snrs_linear_svd]
for i, (sv, snr) in enumerate(zip(s, stream_snrs_db_svd)):
    print(f"  Stream {i+1}: σ={sv:.3f}, SNR={snr:.1f} dB")

print()
print("=== GMD Precoding ===")
sigma_bar = np.prod(s) ** (1.0 / len(s))
print(f"  σ_bar = {sigma_bar:.3f}")
uniform_snr_linear = (sigma_bar ** 2) * snr_linear
uniform_snr_db = 10 * math.log10(uniform_snr_linear)
print(f"  All streams SNR = {uniform_snr_db:.1f} dB (uniform)")

print()
print("=== Comparison ===")
print(f"  SVD max SNR: {max(stream_snrs_db_svd):.1f} dB")
print(f"  SVD min SNR: {min(stream_snrs_db_svd):.1f} dB")
print(f"  SVD avg SNR: {np.mean(stream_snrs_db_svd):.1f} dB")
print(f"  GMD uniform SNR: {uniform_snr_db:.1f} dB")
print()
print(f"  GMD achieves fairness by making all streams have the same SNR")
print(f"  This is the geometric mean of all singular values")
