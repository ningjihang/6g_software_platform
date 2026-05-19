from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bicm_metrics import get_constellation
from channel_estimation import estimate_user_channels_with_gaussian_error
from channel_estimation_mmse import (
    estimate_channel_covariance_from_model,
    estimate_user_channels_with_mmse_pilots_full_covariance,
)
from multiuser_simulation_environment import MultiUserSimulationEnvironment
from sic_sample_average import build_multiuser_sample_average
from thp_precoding import centered_modulo_complex


@dataclass(frozen=True)
class CompareConfig:
    mode: str
    bits_per_symbol: int
    snr_values_db: np.ndarray
    num_channels: int
    train_num_samples: int
    train_num_repeats: int
    base_seed: int
    out_dir: Path
    output_tag: str
    pilot_length: int | None
    pilot_snr_db: float | None
    reuse_pilot_noise_across_snr: bool
    covariance_num_samples: int | None
    covariance_diagonal_loading: float | None
    csi_nmse_db: float | None
    labeling: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid MU comparison for SVD, GMD, and UCD under their default receiver chains.",
    )
    parser.add_argument("--num-users", type=int, default=2)
    parser.add_argument("--num-tx-antennas", type=int, default=16)
    parser.add_argument("--num-rx-antennas", type=int, default=4)
    parser.add_argument("--num-rf-chains", type=int, default=8)
    parser.add_argument("--num-streams-per-user", type=int, default=4)
    parser.add_argument("--digital-power-constraint", type=float, default=None)
    parser.add_argument("--bits-per-symbol", type=int, default=6)
    parser.add_argument("--channel-type", type=str, default="cdl-a")
    parser.add_argument("--mode", choices=("perfect", "gaussian", "mmse_fullcov"), default="mmse_fullcov")
    parser.add_argument("--csi-nmse-db", type=float, default=-20.0)
    parser.add_argument("--pilot-length", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--reuse-pilot-noise-across-snr", action="store_true")
    parser.add_argument("--snr-start-db", type=float, default=0.0)
    parser.add_argument("--snr-stop-db", type=float, default=40.0)
    parser.add_argument("--snr-step-db", type=float, default=2.5)
    parser.add_argument("--num-channels", type=int, default=10)
    parser.add_argument("--train-num-samples", type=int, default=128)
    parser.add_argument("--train-num-repeats", type=int, default=2)
    parser.add_argument("--covariance-num-samples", type=int, default=256)
    parser.add_argument("--covariance-diagonal-loading", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--labeling", choices=("gray_standard", "nr_like"), default="gray_standard")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    return parser.parse_args()


def build_snr_values(start_db: float, stop_db: float, step_db: float) -> np.ndarray:
    if step_db <= 0.0:
        raise ValueError(f"snr_step_db must be positive, got {step_db}.")
    return np.arange(start_db, stop_db + 0.5 * step_db, step_db, dtype=float)


def _format_tag_value(value: float) -> str:
    numeric_value = float(value)
    if np.isclose(numeric_value, round(numeric_value)):
        text = str(int(round(numeric_value)))
    else:
        text = f"{numeric_value:g}"
    return text.replace("-", "m").replace(".", "p")


def resolve_digital_power_constraint(args: argparse.Namespace) -> float:
    if args.digital_power_constraint is None:
        return float(args.num_users * args.num_streams_per_user)
    resolved = float(args.digital_power_constraint)
    if resolved <= 0.0:
        raise ValueError("digital_power_constraint must be positive.")
    return resolved


def build_output_tag(args: argparse.Namespace) -> str:
    resolved_power = resolve_digital_power_constraint(args)
    mode_map = {
        "perfect": "理想信道",
        "gaussian": "高斯信道误差",
        "mmse_fullcov": "全协方差MMSE",
    }
    parts = [
        "多用户混合SVD_GMD_UCD对比",
        mode_map.get(args.mode, str(args.mode)),
        str(args.channel_type).strip().upper(),
        f"K{args.num_users}",
        f"Nt{args.num_tx_antennas}",
        f"Nr{args.num_rx_antennas}",
        f"Nrf{args.num_rf_chains}",
        f"Ns{args.num_streams_per_user}",
        f"{2 ** args.bits_per_symbol}QAM",
        (
            f"SNR{_format_tag_value(args.snr_start_db)}"
            f"到{_format_tag_value(args.snr_stop_db)}"
            f"步长{_format_tag_value(args.snr_step_db)}dB"
        ),
        f"功率{_format_tag_value(resolved_power)}",
    ]
    if args.mode == "mmse_fullcov":
        parts.append(f"导频{_format_tag_value(args.pilot_length)}")
    return "_".join(parts)


def build_config(args: argparse.Namespace) -> CompareConfig:
    return CompareConfig(
        mode=args.mode,
        bits_per_symbol=args.bits_per_symbol,
        snr_values_db=build_snr_values(args.snr_start_db, args.snr_stop_db, args.snr_step_db),
        num_channels=args.num_channels,
        train_num_samples=args.train_num_samples,
        train_num_repeats=args.train_num_repeats,
        base_seed=args.seed,
        out_dir=Path(args.out_dir),
        output_tag=build_output_tag(args),
        pilot_length=args.pilot_length if args.mode == "mmse_fullcov" else None,
        pilot_snr_db=args.pilot_snr_db if args.mode == "mmse_fullcov" else None,
        reuse_pilot_noise_across_snr=bool(args.reuse_pilot_noise_across_snr),
        covariance_num_samples=args.covariance_num_samples if args.mode == "mmse_fullcov" else None,
        covariance_diagonal_loading=args.covariance_diagonal_loading if args.mode == "mmse_fullcov" else None,
        csi_nmse_db=args.csi_nmse_db if args.mode == "gaussian" else None,
        labeling=args.labeling,
    )


def _resolve_pilot_snr_db(config: CompareConfig, snr_db: float) -> float:
    if config.pilot_snr_db is not None:
        return float(config.pilot_snr_db)
    return float(snr_db)


def _resolve_mmse_estimate_seed(config: CompareConfig, snr_index: int, chan_index: int) -> int:
    if config.reuse_pilot_noise_across_snr:
        return int(config.base_seed + chan_index)
    return int(config.base_seed + snr_index * config.num_channels + chan_index)


def _gaussian_sum_rate_from_user_rho(user_rho: list[np.ndarray]) -> float:
    total_rate = 0.0
    for rho_values in user_rho:
        rho_array = np.maximum(np.asarray(rho_values, dtype=float), 0.0)
        total_rate += float(np.sum(np.log2(1.0 + rho_array)))
    return total_rate


def run_compare(env: MultiUserSimulationEnvironment, config: CompareConfig) -> tuple[Path, Path, Path]:
    np.random.seed(config.base_seed)

    print(f"[Stage 1/4] Generating {config.num_channels} true channel realizations ...", flush=True)
    true_base_channels = []
    for chan_idx in range(config.num_channels):
        true_base_channels.append(env.generate_user_channels())
        print(f"  true channel {chan_idx + 1}/{config.num_channels} ready", flush=True)

    fixed_design_base_channels = []
    sampled_covariance = None
    if config.mode == "mmse_fullcov":
        print(
            f"[Stage 2/4] Estimating full covariance for MMSE (samples={int(config.covariance_num_samples)}) ...",
            flush=True,
        )
        sampled_covariance = estimate_channel_covariance_from_model(
            channel_model=env.channel_model,
            num_samples=int(config.covariance_num_samples),
            seed=config.base_seed + 400000,
            diagonal_loading=float(config.covariance_diagonal_loading),
        )
        print("[Stage 2/4] Full covariance estimation done", flush=True)

    for chan_idx, true_channels in enumerate(true_base_channels):
        if config.mode == "perfect":
            fixed_design_base_channels.append(np.asarray(true_channels, dtype=complex))
        elif config.mode == "gaussian":
            estimate = estimate_user_channels_with_gaussian_error(
                true_user_channels=true_channels,
                nmse_db=float(config.csi_nmse_db),
                seed=config.base_seed + 200000 + chan_idx,
            )
            fixed_design_base_channels.append(estimate.estimated_channels)

    print(
        f"[Stage 3/4] Building shared Monte Carlo batches (samples={config.train_num_samples}, repeats={config.train_num_repeats}) ...",
        flush=True,
    )
    shared_sample_averages = []
    for chan_idx in range(config.num_channels):
        shared_sample_averages.append(
            build_multiuser_sample_average(
                env=env,
                bits_per_symbol=config.bits_per_symbol,
                num_samples=config.train_num_samples,
                num_repeats=config.train_num_repeats,
                base_seed=config.base_seed + 1000 * chan_idx,
                labeling=config.labeling,
            )
        )
        print(f"  Monte Carlo batch {chan_idx + 1}/{config.num_channels} ready", flush=True)
    print("[Stage 3/4] Shared Monte Carlo batches done", flush=True)

    print(
        f"Config: {env.num_users} users, Nt={env.num_tx_antennas}, Nr={env.num_rx_antennas}/user, "
        f"Nrf={env.num_rf_chains}, {env.num_streams_per_user} streams/user, "
        f"Pdig={env.digital_power_constraint:g}, {2 ** config.bits_per_symbol}-QAM, "
        f"channel={env.channel_model.channel_type}",
        flush=True,
    )
    print("Comparison: SVD vs GMD vs UCD", flush=True)
    print("Metric: SVD uses the current generic recursive chain; GMD/UCD use fixed design-side recursive chains", flush=True)
    print("", flush=True)
    print(" SNR(dB) | svd | gmd | ucd | gain_gmd_vs_svd | gain_ucd_vs_svd | gain_ucd_vs_gmd", flush=True)
    print("-------------------------------------------------------------------------------------", flush=True)

    rows: list[dict[str, float]] = []
    curve_svd = []
    curve_gmd = []
    curve_ucd = []

    config.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = config.out_dir / f"{config.output_tag}.csv"
    out_png = config.out_dir / f"{config.output_tag}.png"
    out_ber_png = config.out_dir / f"{config.output_tag}_ber.png"

    print("[Stage 4/4] Running SNR sweep ...", flush=True)
    total_snr_points = len(config.snr_values_db)
    for snr_idx, snr_db in enumerate(config.snr_values_db):
        print(f"[SNR {snr_idx + 1}/{total_snr_points}] start {snr_db:.1f} dB", flush=True)
        snr_linear = 10 ** (float(snr_db) / 10.0)
        snr_per_stream = snr_linear / env.total_streams

        svd_sum = 0.0
        gmd_sum = 0.0
        ucd_sum = 0.0
        svd_ber_sum = 0.0
        gmd_ber_sum = 0.0
        ucd_ber_sum = 0.0
        svd_leakage_sum = 0.0
        gmd_leakage_sum = 0.0
        ucd_leakage_sum = 0.0

        for chan_idx, true_user_channels in enumerate(true_base_channels):
            if config.mode == "perfect":
                design_user_channels = np.asarray(true_user_channels, dtype=complex)
            elif config.mode == "gaussian":
                design_user_channels = fixed_design_base_channels[chan_idx]
            elif config.mode == "mmse_fullcov":
                design_user_channels = estimate_user_channels_with_mmse_pilots_full_covariance(
                    true_user_channels=true_user_channels,
                    channel_covariance=sampled_covariance.covariance,
                    channel_mean=sampled_covariance.mean,
                    pilot_length=int(config.pilot_length),
                    pilot_snr_db=_resolve_pilot_snr_db(config, float(snr_db)),
                    seed=_resolve_mmse_estimate_seed(config, snr_idx, chan_idx),
                ).estimated_channels
            else:
                raise ValueError(f"Unsupported mode: {config.mode}")

            train_average = shared_sample_averages[chan_idx]
            f_rf = env.build_analog_precoder(design_user_channels)

            svd_chain = env.build_structured_digital_chain(
                user_channels=design_user_channels,
                f_rf=f_rf,
                snr_per_stream=snr_per_stream,
                strategy="svd",
            )
            gmd_chain = env.build_structured_digital_chain(
                user_channels=design_user_channels,
                f_rf=f_rf,
                snr_per_stream=snr_per_stream,
                strategy="gmd",
            )
            ucd_chain = env.build_structured_digital_chain(
                user_channels=design_user_channels,
                f_rf=f_rf,
                snr_per_stream=snr_per_stream,
                strategy="ucd",
            )

            svd_eval = env.evaluate_precoder_current_receiver_average_fixed_chain(
                user_channels=true_user_channels,
                f_rf=f_rf,
                f_bb=svd_chain.f_bb,
                r_chains=svd_chain.r_chains,
                q_chains=svd_chain.q_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=config.bits_per_symbol,
                sample_average=train_average,
                labeling=config.labeling,
            )
            gmd_eval = env.evaluate_precoder_current_receiver_average_fixed_chain(
                user_channels=true_user_channels,
                f_rf=f_rf,
                f_bb=gmd_chain.f_bb,
                r_chains=gmd_chain.r_chains,
                q_chains=gmd_chain.q_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=config.bits_per_symbol,
                sample_average=train_average,
                labeling=config.labeling,
            )
            ucd_eval = env.evaluate_ucd_precoder_current_receiver_average_b_chain(
                user_channels=true_user_channels,
                f_rf=f_rf,
                f_bb=ucd_chain.f_bb,
                q_chains=ucd_chain.q_chains,
                r_chains=ucd_chain.r_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=config.bits_per_symbol,
                sample_average=train_average,
                labeling=config.labeling,
            )

            svd_sum += svd_eval.sum_rate
            gmd_sum += gmd_eval.sum_rate
            ucd_sum += ucd_eval.sum_rate
            svd_ber_sum += svd_eval.bit_error_rate
            gmd_ber_sum += gmd_eval.bit_error_rate
            ucd_ber_sum += ucd_eval.bit_error_rate
            svd_leakage_sum += svd_eval.offdiag_to_desired
            gmd_leakage_sum += gmd_eval.offdiag_to_desired
            ucd_leakage_sum += ucd_eval.offdiag_to_desired
            print(f"  channel {chan_idx + 1}/{config.num_channels} done at {snr_db:.1f} dB", flush=True)

        avg_svd = svd_sum / config.num_channels
        avg_gmd = gmd_sum / config.num_channels
        avg_ucd = ucd_sum / config.num_channels
        avg_svd_ber = svd_ber_sum / config.num_channels
        avg_gmd_ber = gmd_ber_sum / config.num_channels
        avg_ucd_ber = ucd_ber_sum / config.num_channels
        avg_svd_leakage = svd_leakage_sum / config.num_channels
        avg_gmd_leakage = gmd_leakage_sum / config.num_channels
        avg_ucd_leakage = ucd_leakage_sum / config.num_channels

        curve_svd.append(avg_svd)
        curve_gmd.append(avg_gmd)
        curve_ucd.append(avg_ucd)
        rows.append(
            {
                "snr_db": float(snr_db),
                "svd": avg_svd,
                "gmd": avg_gmd,
                "ucd": avg_ucd,
                "gain_gmd_vs_svd": avg_gmd - avg_svd,
                "gain_ucd_vs_svd": avg_ucd - avg_svd,
                "gain_ucd_vs_gmd": avg_ucd - avg_gmd,
                "svd_ber": avg_svd_ber,
                "gmd_ber": avg_gmd_ber,
                "ucd_ber": avg_ucd_ber,
                "svd_leakage": avg_svd_leakage,
                "gmd_leakage": avg_gmd_leakage,
                "ucd_leakage": avg_ucd_leakage,
            }
        )
        print(
            f"{snr_db:8.1f} | {avg_svd:7.4f} | {avg_gmd:7.4f} | {avg_ucd:7.4f} | "
            f"{avg_gmd - avg_svd:15.4f} | {avg_ucd - avg_svd:15.4f} | {avg_ucd - avg_gmd:15.4f}",
            flush=True,
        )

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(9.0, 5.4))
    plt.plot(config.snr_values_db, curve_svd, marker="d", linewidth=2.0, label="SVD")
    plt.plot(config.snr_values_db, curve_gmd, marker="v", linewidth=2.0, label="GMD")
    plt.plot(config.snr_values_db, curve_ucd, marker="o", linewidth=2.0, label="UCD")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sum-Rate (bits/s/Hz)")
    plt.title("Hybrid SVD vs GMD vs UCD")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    plt.figure(figsize=(9.0, 5.4))
    plt.yscale("log")
    plt.plot(config.snr_values_db, [row["svd_ber"] for row in rows], marker="d", linewidth=2.0, label="SVD BER")
    plt.plot(config.snr_values_db, [row["gmd_ber"] for row in rows], marker="v", linewidth=2.0, label="GMD BER")
    plt.plot(config.snr_values_db, [row["ucd_ber"] for row in rows], marker="o", linewidth=2.0, label="UCD BER")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Hybrid SVD vs GMD vs UCD (BER)")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_ber_png, dpi=180)
    plt.close()

    print(f"\nSaved csv: {out_csv.resolve()}", flush=True)
    print(f"Saved plot: {out_png.resolve()}", flush=True)
    print(f"Saved BER plot: {out_ber_png.resolve()}", flush=True)
    return out_csv.resolve(), out_png.resolve(), out_ber_png.resolve()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    env = MultiUserSimulationEnvironment(
        num_users=args.num_users,
        num_tx_antennas=args.num_tx_antennas,
        num_rx_antennas=args.num_rx_antennas,
        num_rf_chains=args.num_rf_chains,
        num_streams_per_user=args.num_streams_per_user,
        channel_type=args.channel_type,
        digital_power_constraint=args.digital_power_constraint,
    )
    run_compare(env=env, config=config)


if __name__ == "__main__":
    main()
