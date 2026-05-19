from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

def qr_with_positive_diagonal(channel_block: np.ndarray) -> np.ndarray:
    """?? qr with positive diagonal ???"""
    num_streams = channel_block.shape[1]
    r_factor = np.linalg.qr(channel_block)[1][:num_streams, :num_streams]
    diagonal = np.diag(r_factor)
    phase = np.exp(-1j * np.angle(diagonal))
    return np.diag(phase) @ r_factor


def build_fd_svd_precoder(channel: np.ndarray, num_streams: int) -> np.ndarray:
    """?????fd svd precoder?"""
    _, _, vh_channel = svd(channel, full_matrices=False)
    return vh_channel.conj().T[:, :num_streams]


@dataclass(frozen=True)
class UCDTransceiverBlock:
    p_ucd: np.ndarray
    w_ucd: np.ndarray
    r_aug_ucd: np.ndarray
    rho_target: np.ndarray
    power_loading: np.ndarray


@dataclass(frozen=True)
class GMDTransceiverBlock:
    p_gmd: np.ndarray
    q_gmd: np.ndarray
    r_gmd: np.ndarray


class DigitalStructuredPrecoder:
    """
    Structured hybrid digital precoder on the per-user reduced channel.

    Compared with fully-digital equations, here we only replace:
        H  ->  H_eff = H @ V_RF
    """

    def __init__(
        self,
        num_rf_chains: int,
        num_streams: int,
        ucd_waterfill: bool = False,
        ucd_min_power_loading: float = 0.0,
    ):
        """????????????"""
        self.num_rf_chains = num_rf_chains
        self.num_streams = num_streams
        self.ucd_waterfill = bool(ucd_waterfill)
        self.ucd_min_power_loading = float(ucd_min_power_loading)

    def design_precoder(
        self,
        h_eff: np.ndarray,
        snr_per_stream: float,
        strategy: str = "gmd",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """?? design precoder ???"""
        _, singular_values, vh_eff = svd(h_eff, full_matrices=False)
        singular_values = singular_values[: self.num_streams]
        v_eff = vh_eff.conj().T[:, : self.num_streams]
        v_d, rho_target = self.design_from_svd(
            v_eff=v_eff,
            singular_values=singular_values,
            snr_per_stream=snr_per_stream,
            strategy=strategy,
        )
        rho_realized = self.compute_stream_snr(h_eff, v_d, snr_per_stream)
        return v_d, rho_realized, rho_target

    def design_from_svd(
        self,
        v_eff: np.ndarray,
        singular_values: np.ndarray,
        snr_per_stream: float,
        strategy: str = "gmd",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Design the right-unitary digital block from a reduced-channel SVD."""

        strategy = strategy.lower()
        singular_values = np.asarray(singular_values, dtype=float)
        mu = (singular_values ** 2) * snr_per_stream

        if strategy == "svd":
            return v_eff, mu.copy()
        if strategy == "ucd":
            return self._design_strict_ucd(
                v_eff=v_eff,
                singular_values=singular_values,
                snr_per_stream=snr_per_stream,
            )

        rho_target = self.build_target_rho(mu, strategy)
        v_d = self.apply_gtd(v_eff, singular_values, rho_target, snr_per_stream)
        return v_d, rho_target

    def apply_gtd(
        self,
        v_eff: np.ndarray,
        singular_values: np.ndarray,
        rho_target: np.ndarray,
        snr_per_stream: float,
    ) -> np.ndarray:
        """?? apply gtd ???"""
        num_streams = len(singular_values)
        r_target = np.sqrt(np.maximum(rho_target, 1e-12) / max(snr_per_stream, 1e-12))

        if self._is_uniform_target(r_target):
            return self._apply_strict_gmd(
                v_eff=v_eff,
                singular_values=singular_values,
                target_diagonal=float(r_target[0]),
            )

        r_mat = np.diag(singular_values).astype(float)
        omega = np.eye(num_streams, dtype=complex)

        for i in range(num_streams - 1):
            found = False
            pivot = num_streams - 1
            for p in range(i + 1, num_streams):
                if r_mat[p, p] <= r_target[i] <= r_mat[i, i] + 1e-12:
                    pivot = p
                    found = True
                    break
            if not found:
                pivot = num_streams - 1

            if pivot != i + 1:
                r_mat[[i + 1, pivot], [i + 1, pivot]] = r_mat[[pivot, i + 1], [pivot, i + 1]]
                omega[:, [i + 1, pivot]] = omega[:, [pivot, i + 1]]

            numerator = r_target[i] ** 2 - r_mat[i + 1, i + 1] ** 2
            denominator = r_mat[i, i] ** 2 - r_mat[i + 1, i + 1] ** 2 + 1e-12
            val = float(np.clip(numerator / denominator, 0.0, 1.0))
            c = np.sqrt(val)
            s = np.sqrt(max(0.0, 1.0 - c**2))
            g_r = np.array([[c, -s], [s, c]], dtype=float)

            r_mat[:, [i, i + 1]] = r_mat[:, [i, i + 1]] @ g_r
            omega[:, [i, i + 1]] = omega[:, [i, i + 1]] @ g_r

            x = float(r_mat[i, i])
            y = float(r_mat[i + 1, i])
            norm_xy = np.sqrt(x**2 + y**2)
            if norm_xy > 1e-12:
                c_l = x / norm_xy
                s_l = -y / norm_xy
                g_l = np.array([[c_l, -s_l], [s_l, c_l]], dtype=float)
                r_mat[[i, i + 1], :] = g_l @ r_mat[[i, i + 1], :]

            r_mat[i, i] = r_target[i]
            r_mat[i + 1, i] = 0.0

        r_mat[num_streams - 1, num_streams - 1] = r_target[num_streams - 1]

        # Keep GMD as a pure right-unitary rotation of the SVD basis.
        return v_eff @ omega

    def _design_strict_ucd(
        self,
        v_eff: np.ndarray,
        singular_values: np.ndarray,
        snr_per_stream: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct UCD using MMSE augmentation, waterfilling, and GMD equalization."""

        singular_values = np.asarray(singular_values, dtype=float)
        if snr_per_stream < 0.0:
            raise ValueError(
                f"Strict UCD requires non-negative snr_per_stream, got {snr_per_stream}."
            )

        num_streams = len(singular_values)
        alpha = 1.0 / max(float(snr_per_stream), 1e-12)
        power_loading = self._build_ucd_power_loading(
            singular_values=singular_values,
            alpha=alpha,
            num_streams=num_streams,
        )
        effective_power_loading = np.maximum(power_loading, self.ucd_min_power_loading)
        sigma_loaded = np.sqrt(effective_power_loading) * singular_values
        augmented_singular_values = np.sqrt(np.maximum(sigma_loaded**2 + alpha, 1e-12))
        target_diagonal = float(
            np.exp(np.mean(np.log(np.maximum(augmented_singular_values, 1e-12))))
        )
        omega, upper_b = self._apply_strict_gmd(
            v_eff=np.eye(num_streams, dtype=complex),
            singular_values=augmented_singular_values,
            target_diagonal=target_diagonal,
            return_upper=True,
        )
        v_d = v_eff @ np.diag(np.sqrt(effective_power_loading)) @ omega
        rho_equal = max(target_diagonal**2 / alpha - 1.0, 0.0)
        rho_target = np.full(num_streams, rho_equal, dtype=float)
        return v_d, rho_target

    def design_gmd_transceiver_from_svd(
        self,
        u_eff: np.ndarray,
        v_eff: np.ndarray,
        singular_values: np.ndarray,
    ) -> GMDTransceiverBlock:
        """Return the GMD transmit/receive factors on the reduced SVD channel."""

        singular_values = np.asarray(singular_values, dtype=float)
        num_streams = len(singular_values)
        q_gmd, r_gmd, p_gmd = self._apply_original_gmd_factors(
            left_basis=u_eff[:, :num_streams],
            right_basis=v_eff[:, :num_streams],
            singular_values=singular_values,
        )
        return GMDTransceiverBlock(
            p_gmd=np.asarray(p_gmd, dtype=complex),
            q_gmd=np.asarray(q_gmd, dtype=complex),
            r_gmd=np.asarray(r_gmd, dtype=complex),
        )

    def design_ucd_transceiver_from_svd(
        self,
        u_eff: np.ndarray,
        v_eff: np.ndarray,
        singular_values: np.ndarray,
        snr_per_stream: float,
        tx_scale: float = 1.0,
    ) -> UCDTransceiverBlock:
        """Return the UCD transmit/receive factors following the original UCD construction."""

        singular_values = np.asarray(singular_values, dtype=float)
        if snr_per_stream < 0.0:
            raise ValueError(
                f"UCD requires non-negative snr_per_stream, got {snr_per_stream}."
            )

        num_streams = len(singular_values)
        alpha = 1.0 / max(float(snr_per_stream), 1e-12)
        power_loading = self._build_ucd_power_loading(
            singular_values=singular_values,
            alpha=alpha,
            num_streams=num_streams,
        )
        effective_power_loading = np.maximum(power_loading, self.ucd_min_power_loading)
        tx_scale = float(tx_scale)
        sigma_loaded = tx_scale * np.sqrt(effective_power_loading) * singular_values
        augmented_singular_values = np.sqrt(np.maximum(sigma_loaded**2 + alpha, 1e-12))
        target_diagonal = float(
            np.exp(np.mean(np.log(np.maximum(augmented_singular_values, 1e-12))))
        )
        left_scale = sigma_loaded / np.maximum(augmented_singular_values, 1e-12)
        left_basis = u_eff[:, :num_streams] @ np.diag(left_scale)
        right_scale = tx_scale * np.sqrt(effective_power_loading)
        right_basis = v_eff[:, :num_streams] @ np.diag(right_scale)
        q_ucd, r_ucd, p_ucd = self._apply_strict_gmd_factors(
            left_basis=left_basis,
            right_basis=right_basis,
            singular_values=augmented_singular_values,
            target_diagonal=target_diagonal,
        )
        rho_equal = max(target_diagonal**2 / alpha - 1.0, 0.0)
        rho_target = np.full(num_streams, rho_equal, dtype=float)
        diagonal = np.diag(r_ucd)
        safe_diagonal = np.where(np.abs(diagonal) > 1e-12, diagonal, 1e-12 + 0.0j)
        w_ucd = q_ucd @ np.diag(1.0 / safe_diagonal)
        return UCDTransceiverBlock(
            p_ucd=np.asarray(p_ucd, dtype=complex),
            w_ucd=np.asarray(w_ucd, dtype=complex),
            r_aug_ucd=np.asarray(r_ucd, dtype=complex),
            rho_target=rho_target,
            power_loading=effective_power_loading,
        )

    def _build_ucd_power_loading(
        self,
        singular_values: np.ndarray,
        alpha: float,
        num_streams: int,
    ) -> np.ndarray:
        """Replicate the original UCD waterfilling-style power loading."""

        if not self.ucd_waterfill:
            return np.full(
                num_streams,
                max(1.0, self.ucd_min_power_loading),
                dtype=float,
            )

        singular_power = np.maximum(np.asarray(singular_values, dtype=float) ** 2, 1e-12)
        water_level = float(num_streams)
        active_streams = num_streams

        for stream_count in range(num_streams, 0, -1):
            inverse_snr_sum = np.sum(alpha / singular_power[:stream_count])
            water_level = (num_streams + inverse_snr_sum) / stream_count
            threshold = alpha / singular_power[stream_count - 1]
            if water_level > threshold:
                active_streams = stream_count
                break

        power_loading = np.zeros(num_streams, dtype=float)
        power_loading[:active_streams] = np.maximum(
            water_level - alpha / singular_power[:active_streams],
            0.0,
        )
        return np.maximum(power_loading, self.ucd_min_power_loading)

    def _apply_strict_gmd(
        self,
        v_eff: np.ndarray,
        singular_values: np.ndarray,
        target_diagonal: float,
        return_upper: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Apply the original Hager/Jiang GMD update on the right singular basis."""

        q_mat, r_mat, p_mat = self._apply_original_gmd_factors(
            left_basis=np.eye(len(singular_values), dtype=complex),
            right_basis=v_eff[:, : len(singular_values)],
            singular_values=singular_values,
        )
        del q_mat
        if return_upper:
            return p_mat, r_mat
        return p_mat

    def _apply_strict_gmd_factors(
        self,
        left_basis: np.ndarray,
        right_basis: np.ndarray,
        singular_values: np.ndarray,
        target_diagonal: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the original Hager/Jiang GMD update while carrying both factors."""

        del target_diagonal
        return self._apply_original_gmd_factors(
            left_basis=left_basis,
            right_basis=right_basis,
            singular_values=singular_values,
        )

    def _apply_original_gmd_factors(
        self,
        left_basis: np.ndarray,
        right_basis: np.ndarray,
        singular_values: np.ndarray,
        tol: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact port of the original Hager/Jiang MATLAB `gmd.m` update."""

        singular_values = np.asarray(singular_values, dtype=float)
        num_streams = len(singular_values)
        q_mat = np.asarray(left_basis[:, :num_streams], dtype=complex).copy()
        p_mat = np.asarray(right_basis[:, :num_streams], dtype=complex).copy()
        r_mat = np.zeros((num_streams, num_streams), dtype=float)
        d_full = np.zeros(num_streams + 1, dtype=float)
        d_full[1:] = singular_values

        positive_count = 0
        for idx in range(num_streams, 0, -1):
            if d_full[idx] >= tol:
                positive_count = idx
                break
        if positive_count < 1:
            return q_mat, r_mat.astype(complex), p_mat
        if positive_count < 2:
            r_mat[0, 0] = d_full[1]
            return q_mat, r_mat.astype(complex), p_mat

        z_full = np.zeros(positive_count, dtype=float)
        large = 2
        small = positive_count
        perm = np.arange(positive_count + 1, dtype=int)
        invperm = np.arange(positive_count + 1, dtype=int)
        sigma_bar = float(np.prod(d_full[1 : positive_count + 1]) ** (1.0 / positive_count))

        for stream_idx_1b in range(1, positive_count):
            flag = False
            if d_full[stream_idx_1b] >= sigma_bar:
                pivot_1b = int(perm[small])
                small -= 1
                if d_full[pivot_1b] >= sigma_bar:
                    flag = True
            else:
                pivot_1b = int(perm[large])
                large += 1
                if d_full[pivot_1b] <= sigma_bar:
                    flag = True

            next_idx_1b = stream_idx_1b + 1
            if pivot_1b != next_idx_1b:
                tmp = d_full[next_idx_1b]
                d_full[next_idx_1b] = d_full[pivot_1b]
                d_full[pivot_1b] = tmp

                perm_slot = int(invperm[next_idx_1b])
                perm[perm_slot] = pivot_1b
                invperm[pivot_1b] = perm_slot

                cols = [next_idx_1b - 1, pivot_1b - 1]
                q_mat[:, cols] = q_mat[:, cols[::-1]]
                p_mat[:, cols] = p_mat[:, cols[::-1]]

            delta_1 = float(d_full[stream_idx_1b])
            delta_2 = float(d_full[next_idx_1b])
            total_delta = delta_1 + delta_2
            if flag:
                cosine = 1.0
                sine = 0.0
            else:
                denominator = delta_1 - delta_2
                if abs(denominator) <= tol:
                    denominator = np.copysign(tol, denominator if denominator != 0.0 else 1.0)
                frac = (delta_1 - sigma_bar) / denominator
                sine_arg = frac * (delta_1 + sigma_bar) / total_delta
                cosine_arg = (1.0 - frac) * (delta_2 + sigma_bar) / total_delta
                sine = float(np.sqrt(max(sine_arg, 0.0)))
                cosine = float(np.sqrt(max(cosine_arg, 0.0)))

            d_full[next_idx_1b] = delta_1 * delta_2 / sigma_bar
            z_full[stream_idx_1b] = (
                sine
                * cosine
                * (delta_2 - delta_1)
                * total_delta
                / sigma_bar
            )
            r_mat[stream_idx_1b - 1, stream_idx_1b - 1] = sigma_bar
            if stream_idx_1b > 1:
                r_mat[: stream_idx_1b - 1, stream_idx_1b - 1] = z_full[1:stream_idx_1b] * cosine
                z_full[1:stream_idx_1b] = -z_full[1:stream_idx_1b] * sine

            g1 = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
            pair = [stream_idx_1b - 1, next_idx_1b - 1]
            p_mat[:, pair] = p_mat[:, pair] @ g1

            g2 = (1.0 / sigma_bar) * np.array(
                [
                    [cosine * delta_1, -sine * delta_2],
                    [sine * delta_2, cosine * delta_1],
                ],
                dtype=float,
            )
            q_mat[:, pair] = q_mat[:, pair] @ g2

        r_mat[positive_count - 1, positive_count - 1] = sigma_bar
        r_mat[: positive_count - 1, positive_count - 1] = z_full[1:positive_count]
        return q_mat, r_mat.astype(complex), p_mat

    def compute_stream_snr(self, h_eff: np.ndarray, v_d: np.ndarray, snr_per_stream: float) -> np.ndarray:
        """?????stream snr?"""
        r = qr_with_positive_diagonal(h_eff @ v_d)
        diag_r = np.abs(np.diag(r)[: self.num_streams]) ** 2
        return diag_r * snr_per_stream

    def build_target_rho(self, mu: np.ndarray, strategy: str) -> np.ndarray:
        """?????target rho?"""
        strategy = strategy.lower()
        if strategy == "svd":
            return mu.copy()
        if strategy == "gmd":
            return self._uniform_target(mu)
        if strategy == "ucd":
            raise ValueError(
                "Strict UCD target generation depends on the MMSE augmented spectrum and must be built via "
                "design_from_svd(..., strategy='ucd')."
            )
        raise ValueError(f"Unknown strategy: {strategy}")

    def _find_gmd_pivot(
        self,
        diagonal: np.ndarray,
        target_diagonal: float,
        start_index: int,
    ) -> int | None:
        """?? find gmd pivot ???"""
        lead = float(diagonal[start_index])
        if lead >= target_diagonal:
            for pivot in range(start_index + 1, len(diagonal)):
                if float(diagonal[pivot]) <= target_diagonal + 1e-12:
                    return pivot
        else:
            for pivot in range(start_index + 1, len(diagonal)):
                if float(diagonal[pivot]) >= target_diagonal - 1e-12:
                    return pivot

        if np.allclose(diagonal[start_index + 1 :], target_diagonal, atol=1e-10):
            return start_index + 1
        return None

    def _uniform_target(self, mu: np.ndarray) -> np.ndarray:
        """?? uniform target ???"""
        clipped = np.maximum(np.asarray(mu, dtype=float), 1e-12)
        return np.exp(np.mean(np.log(clipped))) * np.ones_like(clipped)

    def _is_uniform_target(self, x: np.ndarray, atol: float = 1e-10) -> bool:
        """?? is uniform target ???"""
        return bool(np.allclose(x, x[0], atol=atol, rtol=0.0))
