from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

try:
    from _bootstrap import ensure_classical_on_path
except ModuleNotFoundError:
    from ._bootstrap import ensure_classical_on_path

ensure_classical_on_path()

from bicm_metrics import get_constellation
from fd_mu_environment import FullyDigitalMuMimoBicmEnvironment
from sic_sample_average import build_multiuser_sample_average
@dataclass(frozen=True)
class CompareConfig:
    bits_per_symbol: int
    snr_values_db: np.ndarray
    num_channels: int
    train_num_samples: int
    train_num_repeats: int
    base_seed: int
    out_dir: Path
    output_tag: str
    labeling: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair full-digital comparison for SVD, GMD, and UCD under fixed design-side receiver chains.",
    )
    parser.add_argument("--num-users", type=int, default=2)
    parser.add_argument("--num-tx-antennas", type=int, default=16)
    parser.add_argument("--num-rx-antennas", type=int, default=4)
    parser.add_argument("--num-streams-per-user", type=int, default=4)
    parser.add_argument("--digital-power-constraint", type=float, default=None)
    parser.add_argument("--bits-per-symbol", type=int, default=6)
    parser.add_argument("--channel-type", type=str, default="cdl-a")
    parser.add_argument("--snr-start-db", type=float, default=0.0)
    parser.add_argument("--snr-stop-db", type=float, default=40.0)
    parser.add_argument("--snr-step-db", type=float, default=2.5)
    parser.add_argument("--num-channels", type=int, default=2)
    parser.add_argument("--train-num-samples", type=int, default=128)
    parser.add_argument("--train-num-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--ucd-waterfill", action="store_true")
    parser.add_argument("--ucd-min-power-loading", type=float, default=0.0)
    parser.add_argument("--labeling", choices=("gray_standard", "nr_like"), default="nr_like")
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
    return "_".join(
        [
            "全数字SVD_GMD_UCD公平对比",
            str(args.channel_type).strip().upper(),
            f"K{args.num_users}",
            f"Nt{args.num_tx_antennas}",
            f"Nr{args.num_rx_antennas}",
            f"Ns{args.num_streams_per_user}",
            f"{2 ** args.bits_per_symbol}QAM",
            (
                f"SNR{_format_tag_value(args.snr_start_db)}"
                f"到{_format_tag_value(args.snr_stop_db)}"
                f"步长{_format_tag_value(args.snr_step_db)}dB"
            ),
            f"功率{_format_tag_value(resolved_power)}",
        ]
    )


def build_config(args: argparse.Namespace) -> CompareConfig:
    return CompareConfig(
        bits_per_symbol=args.bits_per_symbol,
        snr_values_db=build_snr_values(args.snr_start_db, args.snr_stop_db, args.snr_step_db),
        num_channels=args.num_channels,
        train_num_samples=args.train_num_samples,
        train_num_repeats=args.train_num_repeats,
        base_seed=args.seed,
        out_dir=Path(args.out_dir),
        output_tag=build_output_tag(args),
        labeling=args.labeling,
    )


def run_compare(env: FullyDigitalMuMimoBicmEnvironment, config: CompareConfig) -> tuple[Path, Path, Path]:
    np.random.seed(config.base_seed)
    base_channels = [env.generate_user_channels() for _ in range(config.num_channels)]
    shared_sample_averages = [
        build_multiuser_sample_average(
            env=env,
            bits_per_symbol=config.bits_per_symbol,
            num_samples=config.train_num_samples,
            num_repeats=config.train_num_repeats,
            base_seed=config.base_seed + 1000 * chan_idx,
            labeling=config.labeling,
        )
        for chan_idx in range(config.num_channels)
    ]

    print(
        f"Config: {env.num_users} users, Nt={env.num_tx_antennas}, "
        f"Nr={env.num_rx_antennas}/user, Nrf={env.num_rf_chains}, "
        f"{env.num_streams_per_user} streams/user, Pdig={env.digital_power_constraint:g}, "
        f"{2 ** config.bits_per_symbol}-QAM, channel={env.channel_type}",
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

    for snr_db in config.snr_values_db:
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

        for chan_idx, user_channels in enumerate(base_channels):
            train_average = shared_sample_averages[chan_idx]

            svd_chain = env.build_structured_chain(
                user_channels=user_channels,
                snr_per_stream=snr_per_stream,
                strategy="svd",
            )
            gmd_chain = env.build_structured_chain(
                user_channels=user_channels,
                snr_per_stream=snr_per_stream,
                strategy="gmd",
            )
            ucd_chain = env.build_structured_chain(
                user_channels=user_channels,
                snr_per_stream=snr_per_stream,
                strategy="ucd",
            )

            svd_eval = env.evaluate_precoder_current_receiver_average_parallel(
                user_channels=user_channels,
                f=svd_chain.f_bb,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=config.bits_per_symbol,
                sample_average=train_average,
                labeling=config.labeling,
            )
            gmd_eval = env.evaluate_precoder_current_receiver_average_thp(
                user_channels=user_channels,
                f=gmd_chain.f_bb,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=config.bits_per_symbol,
                sample_average=train_average,
                labeling=config.labeling,
            )
            ucd_eval = env.evaluate_ucd_precoder_current_receiver_average_b_chain(
                user_channels=user_channels,
                f=ucd_chain.f_bb,
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
    plt.title("Full-Digital SVD vs GMD vs UCD")
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
    plt.title("Full-Digital SVD vs GMD vs UCD (BER)")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_ber_png, dpi=180)
    plt.close()

    print(f"\nSaved CSV: {out_csv}", flush=True)
    print(f"Saved Figure: {out_png}", flush=True)
    print(f"Saved BER Figure: {out_ber_png}", flush=True)
    return out_csv, out_png, out_ber_png


def main() -> None:
    args = parse_args()
    config = build_config(args)
    env = FullyDigitalMuMimoBicmEnvironment(
        num_users=args.num_users,
        num_tx_antennas=args.num_tx_antennas,
        num_rx_antennas=args.num_rx_antennas,
        num_streams_per_user=args.num_streams_per_user,
        channel_type=args.channel_type,
        digital_power_constraint=args.digital_power_constraint,
        ucd_waterfill=args.ucd_waterfill,
        ucd_min_power_loading=args.ucd_min_power_loading,
    )
    run_compare(env=env, config=config)


if __name__ == "__main__":
    main()
