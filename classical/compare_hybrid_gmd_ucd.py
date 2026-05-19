from __future__ import annotations

from compare_hybrid_svd_gmd_ucd import (
    CompareConfig,
    build_config,
    parse_args,
    run_compare as _run_compare_all,
)
from multiuser_simulation_environment import MultiUserSimulationEnvironment
import csv
from pathlib import Path


def run_compare(env: MultiUserSimulationEnvironment, config: CompareConfig):
    csv_path, png_path, ber_path = _run_compare_all(env=env, config=config)
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "snr_db": row["snr_db"],
                    "gmd": row["gmd"],
                    "ucd": row["ucd"],
                    "gain_ucd_vs_gmd": row["gain_ucd_vs_gmd"],
                    "gmd_ber": row["gmd_ber"],
                    "ucd_ber": row["ucd_ber"],
                    "gmd_leakage": row["gmd_leakage"],
                    "ucd_leakage": row["ucd_leakage"],
                }
            )
    slim_csv = Path(csv_path).with_name(Path(csv_path).stem + "_gmd_ucd_only.csv")
    with slim_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved slim CSV: {slim_csv}", flush=True)
    return slim_csv, png_path, ber_path


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
