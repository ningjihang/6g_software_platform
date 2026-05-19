from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStageSpec:
    title: str
    summary: str


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    script: Path
    result_dir: Path
    kind: str
    stages: tuple[PipelineStageSpec, ...]


HYBRID_STAGES = (
    PipelineStageSpec("01 Channel", "Sionna CDL / UMa channel sampling"),
    PipelineStageSpec("02 CSI", "Perfect / Gaussian / MMSE FullCov"),
    PipelineStageSpec("03 RF", "Shared RF analog precoder"),
    PipelineStageSpec("04 Baseband", "F_BB = [N1F1, ..., NKFK] with SVD / GMD / UCD"),
    PipelineStageSpec("05 Metric", "UCD runtime uses B = W^H H P; report GMI / BER / leakage"),
)

DIGITAL_STAGES = (
    PipelineStageSpec("01 Channel", "Sionna CDL / UMa channel sampling"),
    PipelineStageSpec("02 CSI", "Fixed perfect CSI"),
    PipelineStageSpec("03 RF", "Identity RF stage"),
    PipelineStageSpec("04 Baseband", "F_BB = [N1F1, ..., NKFK] with SVD / GMD / UCD"),
    PipelineStageSpec("05 Metric", "Receiver GMI / BER / leakage under current chain"),
)

TASKS = {
    "hybrid": TaskSpec(
        key="hybrid",
        label="Hybrid MU Precoding",
        script=ROOT_DIR / "classical" / "compare_hybrid_svd_gmd_ucd.py",
        result_dir=ROOT_DIR / "classical" / "results",
        kind="hybrid",
        stages=HYBRID_STAGES,
    ),
    "digital": TaskSpec(
        key="digital",
        label="Full-Digital MU Precoding",
        script=ROOT_DIR / "full_digital_mu" / "compare_full_digital_svd_gmd_ucd_fair.py",
        result_dir=ROOT_DIR / "full_digital_mu" / "results",
        kind="digital",
        stages=DIGITAL_STAGES,
    ),
}

TASK_LABEL_TO_KEY = {spec.label: key for key, spec in TASKS.items()}
TASK_OPTIONS = [spec.label for spec in TASKS.values()]

CSI_MODE_LABEL_TO_VALUE = {
    "Perfect CSI": "perfect",
    "Gaussian NMSE": "gaussian",
    "MMSE FullCov": "mmse_fullcov",
}
CSI_MODE_VALUE_TO_LABEL = {value: label for label, value in CSI_MODE_LABEL_TO_VALUE.items()}
CSI_MODE_OPTIONS = list(CSI_MODE_LABEL_TO_VALUE.keys())


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class RunnerConfig:
    task_key: str = "hybrid"
    channel_type: str = "cdl-a"
    num_users: int = 1
    num_tx_antennas: int = 16
    num_rx_antennas: int = 4
    num_rf_chains: int = 4
    num_streams_per_user: int = 4
    digital_power_constraint: float | None = None
    bits_per_symbol: int = 6
    csi_mode: str = "perfect"
    csi_nmse_db: float = -20.0
    pilot_length: int = 16
    pilot_snr_db: float | None = None
    user_fairness_penalty_weight: float = 0.0
    snr_start_db: float = 10.0
    snr_stop_db: float = 30.0
    snr_step_db: float = 5.0
    num_channels: int = 10
    train_num_samples: int = 128
    train_num_repeats: int = 2
    seed: int = 20260327
    device: str = field(default_factory=default_device)


def _add_arg(command: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    command.extend([flag, str(value)])


def _maybe_add_numeric(command: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "auto":
            return
        command.extend([flag, text])
        return
    command.extend([flag, str(value)])


def build_command(config: RunnerConfig) -> list[str]:
    task = TASKS[config.task_key]
    command = [sys.executable, str(task.script)]

    _add_arg(command, "--num-users", config.num_users)
    _add_arg(command, "--num-tx-antennas", config.num_tx_antennas)
    _add_arg(command, "--num-rx-antennas", config.num_rx_antennas)
    _add_arg(command, "--num-streams-per-user", config.num_streams_per_user)
    _add_arg(command, "--bits-per-symbol", config.bits_per_symbol)
    _add_arg(command, "--channel-type", config.channel_type)

    if task.kind == "hybrid":
        _add_arg(command, "--num-rf-chains", config.num_rf_chains)
        _maybe_add_numeric(command, "--digital-power-constraint", config.digital_power_constraint)
        _add_arg(command, "--mode", config.csi_mode)
        _maybe_add_numeric(command, "--csi-nmse-db", config.csi_nmse_db)
        _maybe_add_numeric(command, "--pilot-length", config.pilot_length)
        if config.pilot_snr_db is not None:
            _add_arg(command, "--pilot-snr-db", config.pilot_snr_db)
    else:
        _maybe_add_numeric(command, "--digital-power-constraint", config.digital_power_constraint)

    _add_arg(command, "--snr-start-db", config.snr_start_db)
    _add_arg(command, "--snr-stop-db", config.snr_stop_db)
    _add_arg(command, "--snr-step-db", config.snr_step_db)
    _add_arg(command, "--num-channels", config.num_channels)
    _add_arg(command, "--train-num-samples", config.train_num_samples)
    _add_arg(command, "--train-num-repeats", config.train_num_repeats)
    _add_arg(command, "--seed", config.seed)
    _add_arg(command, "--out-dir", task.result_dir)
    return command


def validate_runner_config(config: RunnerConfig) -> str | None:
    total_streams = config.num_users * config.num_streams_per_user
    if config.num_streams_per_user > config.num_rx_antennas:
        return f"Ns / user must not exceed Nr. Got {config.num_streams_per_user} > {config.num_rx_antennas}."

    if config.task_key == "hybrid":
        if total_streams > config.num_rf_chains:
            return f"Total streams must not exceed Nrf. Got {total_streams} > {config.num_rf_chains}."
        return None

    if total_streams > config.num_tx_antennas:
        return f"Total streams must not exceed Nt. Got {total_streams} > {config.num_tx_antennas}."
    return None


def command_to_text(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    try:
        import shlex

        return shlex.join(list(command))
    except Exception:
        return " ".join(list(command))


def task_spec(task_key: str) -> TaskSpec:
    return TASKS[task_key]
