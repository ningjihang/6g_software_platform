# Comparative Analysis of SVD, GMD, and UCD Precoding in Multi-User MIMO Systems

**ningjihang**  
**2025-05-22**

---

## I. Introduction

Multiple-input multiple-output (MIMO) technology has become a cornerstone of modern wireless communication systems, enabling significant improvements in spectral efficiency and reliability. In multi-user MIMO (MU-MIMO) systems, precoding techniques are essential for managing interference among multiple users and maximizing the overall system throughput.

Among various precoding strategies, singular value decomposition (SVD), geometric mean decomposition (GMD), and uniform channel decomposition (UCD) have emerged as three prominent linear precoding methods. Each approach offers distinct advantages in terms of complexity, performance, and implementation feasibility. SVD-based precoding provides optimal point-to-point capacity but suffers from unequal signal-to-noise ratio (SNR) distribution across streams. GMD achieves uniform SNR across all streams through geometric mean decomposition, enabling simpler receivers with bit-interleaved coded modulation (BICM). UCD extends this concept by incorporating MMSE-based power loading for enhanced performance.

This paper presents a comprehensive comparison of these three precoding methods in full-digital MU-MIMO systems. Through extensive simulations over CDL-A channel models, we evaluate the sum-rate performance and bit error rate (BER) characteristics of each method across a wide SNR range from 0 dB to 40 dB. Our analysis reveals that the performance gap between methods varies significantly with SNR, providing valuable insights for system design.

## II. System Model

We consider a downlink MU-MIMO system with $N_t$ transmit antennas at the base station and $K$ users, each equipped with $N_r$ receive antennas. The received signal at user $k$ can be expressed as:

$$y_k = \mathbf{H}_k \mathbf{F} \mathbf{s} + n_k$$

where $\mathbf{H}_k \in \mathbb{C}^{N_r \times N_t}$ represents the channel matrix for user $k$, $\mathbf{F} \in \mathbb{C}^{N_t \times N_s}$ is the precoding matrix with total transmit power constraint $\text{Tr}(\mathbf{F}\mathbf{F}^H) = P$, $\mathbf{s} \in \mathbb{C}^{N_s \times 1}$ denotes the transmitted symbol vector with $E\{\mathbf{s}\mathbf{s}^H\} = \mathbf{I}$, and $n_k \sim \mathcal{CN}(0, \sigma^2\mathbf{I})$ is additive white Gaussian noise.

The total number of data streams is $N_s = K \times N_s^{(k)}$, where $N_s^{(k)}$ represents streams allocated to user $k$. For block diagonalization (BD) to eliminate inter-user interference, we require $N_t \geq N_s$.

## III. Precoding Methods

### A. SVD-Based Precoding

SVD precoding decomposes each user's channel as $\mathbf{H}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^H$, where $\mathbf{V}_k$ contains the right singular vectors. The precoder is constructed using the strongest singular vectors, maximizing the achievable rate. However, the resulting stream SNRs are highly non-uniform:

$$\gamma_i^{(SVD)} = \frac{\sigma_i^2 P}{N_s}$$

where $\sigma_i$ denotes the $i$-th singular value.

### B. GMD-Based Precoding

GMD decomposes the channel as $\mathbf{H} = \mathbf{Q} \mathbf{R} \mathbf{P}^H$, where $\mathbf{R}$ is an upper triangular matrix with equal diagonal elements $\bar{\sigma} = (\prod_{i=1}^{N_s} \sigma_i)^{1/N_s}$. This decomposition yields uniform SNR across all streams:

$$\gamma_i^{(GMD)} = \frac{\bar{\sigma}^2 P}{N_s}$$

The uniform SNR property simplifies the detector design and enables efficient BICM with uniform bit interleaving.

### C. UCD-Based Precoding

UCD extends GMD by incorporating MMSE-based power loading. The effective SNR after UCD is enhanced through interference suppression:

$$\gamma_{MMSE} = \frac{\bar{\sigma}_{eff}^2 P}{N_s(1 + \bar{\sigma}_{eff}^2 P/N_s)}$$

This approach provides a balance between GMD's uniformity and SVD's capacity optimization.

## IV. Simulation Results

### A. Simulation Setup

We evaluate the three precoding methods using the following configuration:
- Transmit antennas: $N_t = 16$
- Receive antennas: $N_r = 4$ per user
- Number of users: $K = 2$
- Streams per user: $N_s^{(k)} = 4$
- Modulation: 64-QAM (spectral efficiency: 6 bits/symbol)
- Channel model: CDL-A with delay spread of 300 ns
- SNR range: 0 dB to 40 dB with 2.5 dB steps

### B. Sum-Rate Performance

| SNR (dB) | SVD (bps/Hz) | GMD (bps/Hz) | UCD (bps/Hz) |
|----------|---------------|---------------|---------------|
| 0        | 6.60          | 0.21          | 0.45          |
| 10       | 22.60         | 16.74         | 17.71         |
| 20       | 41.96         | 43.77         | 43.85         |
| 30       | 47.75         | 48.00         | 48.00         |
| 40       | 48.00         | 48.00         | 48.00         |

The results demonstrate that SVD outperforms GMD and UCD at low SNR (0-15 dB), where the capacity advantage of non-uniform power allocation dominates. However, as SNR increases beyond 20 dB, GMD and UCD converge to near-optimal performance with uniform stream SNR distribution.

### C. BER Performance

The BER analysis reveals interesting trade-offs. At moderate SNR (15-25 dB), GMD and UCD achieve significantly lower BER than SVD due to their uniform stream SNR property. For example, at 20 dB SNR, SVD achieves BER of $3.87 \times 10^{-2}$, while GMD achieves $2.45 \times 10^{-2}$ and UCD achieves $2.40 \times 10^{-2}$. This approximately 1.6 dB SNR advantage makes GMD/UCD attractive for reliability-critical applications.

## V. Conclusion

This paper presented a comprehensive comparison of SVD, GMD, and UCD precoding for multi-user MIMO systems. Our simulation results over CDL-A channels demonstrate that:

1. SVD-based precoding offers superior performance at low SNR due to its water-filling nature
2. GMD and UCD achieve near-optimal performance at high SNR with uniform stream SNR
3. The uniform SNR property of GMD/UCD provides significant BER advantages at moderate SNR
4. The performance gap between methods diminishes at very high SNR where all methods converge to channel capacity

These findings provide valuable guidance for selecting appropriate precoding strategies based on operating SNR range and system requirements. Future work will investigate hybrid precoding architectures that combine the advantages of these methods.

---

**Keywords**: Multi-user MIMO, Precoding, SVD, GMD, UCD, BICM, Sum-rate, BER
