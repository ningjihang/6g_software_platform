from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CLASSICAL_DIR = ROOT / "classical"
for extra_path in (ROOT, CLASSICAL_DIR):
    text = str(extra_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from learn_svd_ucd_threshold import (
    SpectralSetTransformerThresholdRegressor,
    ThresholdSample,
    build_env_from_config,
    config_from_features,
    evaluate_threshold_predictions,
    load_threshold_samples,
)
from multiuser_simulation_environment import MultiUserSimulationEnvironment
from sic_sample_average import build_multiuser_sample_average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SVD, GMD, UCD, and AI threshold-based SVD/UCD selection.",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-users", type=int, default=2)
    parser.add_argument("--num-tx-antennas", type=int, default=16)
    parser.add_argument("--num-rx-antennas", type=int, default=4)
    parser.add_argument("--num-rf-chains", type=int, default=8)
    parser.add_argument("--num-streams-per-user", type=int, default=4)
    parser.add_argument("--digital-power-constraint", type=float, default=None)
    parser.add_argument("--bits-per-symbol", type=int, default=6)
    parser.add_argument("--channel-type", type=str, default="cdl-a")
    parser.add_argument("--snr-start-db", type=float, default=0.0)
    parser.add_argument("--snr-stop-db", type=float, default=30.0)
    parser.add_argument("--snr-step-db", type=float, default=1.0)
    parser.add_argument("--train-num-samples", type=int, default=128)
    parser.add_argument("--train-num-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20270521)
    parser.add_argument("--labeling", choices=("gray_standard", "nr_like"), default="nr_like")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "results" / "ai_threshold_compare"))
    parser.add_argument("--output-prefix", type=str, default="ai_threshold_compare")
    parser.add_argument("--gmd-cache", type=str, default=None)
    return parser.parse_args()


def build_snr_values(start_db: float, stop_db: float, step_db: float) -> np.ndarray:
    if step_db <= 0.0:
        raise ValueError(f"snr_step_db must be positive, got {step_db}.")
    return np.arange(start_db, stop_db + 0.5 * step_db, step_db, dtype=float)


def predict_thresholds(samples: list[ThresholdSample], model_path: Path, args: argparse.Namespace) -> np.ndarray:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    artifacts = checkpoint["artifacts"]

    tokens = np.stack([sample.tokens for sample in samples], axis=0).astype(np.float32)
    token_masks = np.stack([sample.token_mask for sample in samples], axis=0).astype(np.float32)
    config_features = np.stack([sample.config_features for sample in samples], axis=0).astype(np.float32)
    feature_mean = np.asarray(artifacts["feature_mean"], dtype=np.float32).reshape(1, 1, -1)
    feature_std = np.asarray(artifacts["feature_std"], dtype=np.float32).reshape(1, 1, -1)
    config_mean = np.asarray(artifacts["config_mean"], dtype=np.float32).reshape(1, -1)
    config_std = np.asarray(artifacts["config_std"], dtype=np.float32).reshape(1, -1)
    target_mean = float(artifacts["target_mean"])
    target_std = float(artifacts["target_std"])
    tokens_norm = (tokens - feature_mean) / np.maximum(feature_std, 1e-6)
    config_norm = (config_features - config_mean) / np.maximum(config_std, 1e-6)
    sign_point_indices = np.unique(
        np.clip(
            np.round(np.linspace(0, len(args.snr_values_db) - 1, num=min(7, len(args.snr_values_db)))).astype(int),
            0,
            len(args.snr_values_db) - 1,
        )
    )
    sign_snr_values = torch.from_numpy(np.asarray(args.snr_values_db[sign_point_indices], dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralSetTransformerThresholdRegressor(
        token_dim=tokens.shape[-1],
        config_dim=config_features.shape[-1],
        d_model=int(model_args.get("d_model", 32)),
        num_heads=int(model_args.get("num_heads", 2)),
        num_layers=int(model_args.get("num_layers", 1)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_norm, _ = model(
            torch.from_numpy(tokens_norm.astype(np.float32)).to(device),
            torch.from_numpy(token_masks.astype(np.float32)).to(device),
            torch.from_numpy(config_norm.astype(np.float32)).to(device),
            sign_snr_values=sign_snr_values.to(device),
        )
        pred_norm = pred_norm.cpu().numpy()
    return pred_norm * target_std + target_mean


def regenerate_user_channels(env: MultiUserSimulationEnvironment, seed: int, channel_index: int) -> np.ndarray:
    np.random.seed(int(seed) + 1_000_003 * int(channel_index))
    return env.generate_user_channels()


def compute_gmd_rates(
    samples: list[ThresholdSample],
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    rates = np.zeros((len(samples), len(snr_values_db)), dtype=float)
    for sample_idx, sample in enumerate(samples):
        env = build_env_from_config(config_from_features(sample.config_features), args)
        true_channels = regenerate_user_channels(env, int(args.seed), int(sample.channel_index))
        f_rf = env.build_analog_precoder(true_channels)
        sample_average = build_multiuser_sample_average(
            env=env,
            bits_per_symbol=int(args.bits_per_symbol),
            num_samples=int(args.train_num_samples),
            num_repeats=int(args.train_num_repeats),
            base_seed=int(args.seed) + 1000 * int(sample.channel_index),
            labeling=str(args.labeling),
        )
        for snr_idx, snr_db in enumerate(snr_values_db):
            snr_linear = 10 ** (float(snr_db) / 10.0)
            snr_per_stream = snr_linear / env.total_streams
            gmd_chain = env.build_structured_digital_chain(
                user_channels=true_channels,
                f_rf=f_rf,
                snr_per_stream=snr_per_stream,
                strategy="gmd",
            )
            gmd_eval = env.evaluate_precoder_current_receiver_average_fixed_chain(
                user_channels=true_channels,
                f_rf=f_rf,
                f_bb=gmd_chain.f_bb,
                r_chains=gmd_chain.r_chains,
                q_chains=gmd_chain.q_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=int(args.bits_per_symbol),
                sample_average=sample_average,
                labeling=str(args.labeling),
            )
            rates[sample_idx, snr_idx] = float(gmd_eval.sum_rate)
        print(
            f"gmd channel {sample_idx + 1}/{len(samples)} "
            f"(channel_index={sample.channel_index}) done",
            flush=True,
        )
    return rates


def save_outputs(
    snr_values_db: np.ndarray,
    samples: list[ThresholdSample],
    predicted_thresholds: np.ndarray,
    svd_rates: np.ndarray,
    gmd_rates: np.ndarray,
    ucd_rates: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    curve_csv = out_dir / f"{prefix}_curves.csv"
    threshold_csv = out_dir / f"{prefix}_threshold_predictions.csv"
    metrics_json = out_dir / f"{prefix}_metrics.json"
    curve_png = out_dir / f"{prefix}_curves.png"

    snr_grid = np.asarray(snr_values_db, dtype=float)[None, :]
    ai_rates = np.where(snr_grid >= predicted_thresholds[:, None], ucd_rates, svd_rates)
    exhaustive_rates = np.maximum(svd_rates, ucd_rates)

    curve_rows = []
    for snr_idx, snr_db in enumerate(snr_values_db):
        curve_rows.append(
            {
                "snr_db": float(snr_db),
                "svd": float(np.mean(svd_rates[:, snr_idx])),
                "gmd": float(np.mean(gmd_rates[:, snr_idx])),
                "ucd": float(np.mean(ucd_rates[:, snr_idx])),
                "ai_threshold": float(np.mean(ai_rates[:, snr_idx])),
                "exhaustive_svd_ucd": float(np.mean(exhaustive_rates[:, snr_idx])),
                "oracle_svd_ucd": float(np.mean(exhaustive_rates[:, snr_idx])),
            }
        )

    with curve_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snr_db",
                "svd",
                "gmd",
                "ucd",
                "ai_threshold",
                "exhaustive_svd_ucd",
                "oracle_svd_ucd",
            ],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    with threshold_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_index", "true_threshold_db", "pred_threshold_db", "abs_error_db"])
        for sample, pred_threshold in zip(samples, predicted_thresholds):
            true_threshold = float(sample.threshold_db)
            writer.writerow(
                [
                    sample.channel_index,
                    f"{true_threshold:.8g}",
                    f"{float(pred_threshold):.8g}",
                    f"{abs(float(pred_threshold) - true_threshold):.8g}",
                ]
            )

    threshold_metrics = evaluate_threshold_predictions(
        thresholds_true=np.asarray([sample.threshold_db for sample in samples], dtype=float),
        thresholds_pred=np.asarray(predicted_thresholds, dtype=float),
        svd_rates=svd_rates,
        ucd_rates=ucd_rates,
        snr_values_db=snr_values_db,
    )
    metrics = {
        "num_channels": len(samples),
        "snr_start_db": float(snr_values_db[0]),
        "snr_stop_db": float(snr_values_db[-1]),
        "snr_step_db": float(snr_values_db[1] - snr_values_db[0]) if len(snr_values_db) > 1 else 0.0,
        "avg_svd": float(np.mean(svd_rates)),
        "avg_gmd": float(np.mean(gmd_rates)),
        "avg_ucd": float(np.mean(ucd_rates)),
        "avg_ai_threshold": float(np.mean(ai_rates)),
        "avg_exhaustive_svd_ucd": float(np.mean(exhaustive_rates)),
        "avg_oracle_svd_ucd": float(np.mean(exhaustive_rates)),
        "ai_gain_vs_svd": float(np.mean(ai_rates) - np.mean(svd_rates)),
        "ai_gain_vs_gmd": float(np.mean(ai_rates) - np.mean(gmd_rates)),
        "ai_gain_vs_ucd": float(np.mean(ai_rates) - np.mean(ucd_rates)),
        "ai_gap_to_exhaustive_svd_ucd": float(np.mean(exhaustive_rates) - np.mean(ai_rates)),
        **threshold_metrics,
    }
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(snr_values_db, [row["svd"] for row in curve_rows], marker="d", linewidth=2.0, label="SVD")
    plt.plot(snr_values_db, [row["gmd"] for row in curve_rows], marker="v", linewidth=2.0, label="GMD")
    plt.plot(snr_values_db, [row["ucd"] for row in curve_rows], marker="o", linewidth=2.0, label="UCD")
    plt.plot(
        snr_values_db,
        [row["ai_threshold"] for row in curve_rows],
        marker="s",
        linewidth=2.0,
        label="AI threshold",
    )
    plt.plot(
        snr_values_db,
        [row["exhaustive_svd_ucd"] for row in curve_rows],
        linestyle="--",
        linewidth=1.5,
        label="Exhaustive max(SVD,UCD)",
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average BICM rate")
    plt.title("SVD vs GMD vs UCD vs AI Threshold Selection")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_png, dpi=180)
    plt.close()

    return curve_csv, threshold_csv, metrics_json, curve_png


def main() -> None:
    args = parse_args()
    snr_values_db = build_snr_values(args.snr_start_db, args.snr_stop_db, args.snr_step_db)
    args.snr_values_db = snr_values_db
    samples = load_threshold_samples(Path(args.dataset))
    if not samples:
        raise ValueError(f"No samples loaded from dataset: {args.dataset}")

    dataset = np.load(args.dataset, allow_pickle=True)
    dataset_snr = np.asarray(dataset["snr_values_db"], dtype=float)
    if dataset_snr.shape != snr_values_db.shape or not np.allclose(dataset_snr, snr_values_db):
        raise ValueError(
            "Dataset SNR grid does not match requested grid. "
            f"dataset={dataset_snr}, requested={snr_values_db}"
        )

    svd_rates = np.asarray(dataset["svd_rates"], dtype=float)
    ucd_rates = np.asarray(dataset["ucd_rates"], dtype=float)
    predicted_thresholds = predict_thresholds(samples, Path(args.model), args)

    gmd_cache_path = Path(args.gmd_cache) if args.gmd_cache is not None else None
    if gmd_cache_path is not None and gmd_cache_path.exists():
        gmd_rates = np.load(gmd_cache_path)
        print(f"loaded cached GMD rates: {gmd_cache_path}", flush=True)
    else:
        print(
            f"loaded {len(samples)} samples; computing GMD on the same channels/SNR grid ...",
            flush=True,
        )
        gmd_rates = compute_gmd_rates(samples, snr_values_db, args)
        if gmd_cache_path is not None:
            np.save(gmd_cache_path, gmd_rates)
            print(f"saved cached GMD rates: {gmd_cache_path}", flush=True)
    curve_csv, threshold_csv, metrics_json, curve_png = save_outputs(
        snr_values_db=snr_values_db,
        samples=samples,
        predicted_thresholds=predicted_thresholds,
        svd_rates=svd_rates,
        gmd_rates=gmd_rates,
        ucd_rates=ucd_rates,
        args=args,
    )
    print(f"saved curves: {curve_csv}", flush=True)
    print(f"saved threshold predictions: {threshold_csv}", flush=True)
    print(f"saved metrics: {metrics_json}", flush=True)
    print(f"saved plot: {curve_png}", flush=True)
    with metrics_json.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    print("summary metrics:", flush=True)
    for key in (
        "avg_svd",
        "avg_gmd",
        "avg_ucd",
        "avg_ai_threshold",
        "avg_exhaustive_svd_ucd",
        "avg_oracle_svd_ucd",
        "ai_gain_vs_svd",
        "ai_gain_vs_gmd",
        "ai_gain_vs_ucd",
        "ai_gap_to_exhaustive_svd_ucd",
        "threshold_mae_db",
        "grid_decision_accuracy",
        "regret_to_oracle",
    ):
        print(f"  {key}: {metrics[key]:.6g}", flush=True)


if __name__ == "__main__":
    main()
