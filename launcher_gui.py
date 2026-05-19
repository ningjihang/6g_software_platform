from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageOps, ImageTk
except Exception:
    Image = None
    ImageOps = None
    ImageTk = None

try:
    from cdl_a_channel import list_supported_sionna_channel_types
except Exception:
    def list_supported_sionna_channel_types() -> list[str]:
        return []


ROOT_DIR = Path(__file__).resolve().parent
PHOTO_DIR = ROOT_DIR / "photo"
CATALOG_PATH = PHOTO_DIR / "experiment_catalog.json"
FULL_DIGITAL_RESULTS_DIR = ROOT_DIR / "full_digital_mu" / "results"

HYBRID_TRAIN_SCRIPT = ROOT_DIR / "classical" / "compare_hybrid_svd_gmd_ucd.py"
DIGITAL_TRAIN_SCRIPT = ROOT_DIR / "full_digital_mu" / "compare_full_digital_svd_gmd_ucd_fair.py"

TRAIN_TASK_TO_SCRIPT = {
    "多用户数模混合预编码": HYBRID_TRAIN_SCRIPT,
    "数字预编码": DIGITAL_TRAIN_SCRIPT,
}
TRAIN_TASK_TO_RESULT_DIR = {
    "多用户数模混合预编码": ROOT_DIR / "classical" / "results",
    "数字预编码": ROOT_DIR / "full_digital_mu" / "results",
}

ALL_OPTION = "全部"
TRAIN_QAM_OPTIONS = ("QPSK", "4QAM", "16QAM", "64QAM", "256QAM")
TRAIN_CSI_MODE_OPTIONS = (
    "有信道误差 (MMSE FullCov)",
    "无信道误差 (Perfect CSI)",
    "有信道误差 (Gaussian NMSE)",
)
TRAIN_CSI_MODE_TO_ARG = {
    "无信道误差 (Perfect CSI)": "perfect",
    "有信道误差 (Gaussian NMSE)": "gaussian",
    "有信道误差 (MMSE FullCov)": "mmse_fullcov",
}
LEGACY_SINGLE_USER_CHANNEL_BY_FILE = {
    "Nt256_Nr8_4QAM_rate_0_37.png": ("CDL-A", "UMA"),
    "Nt256_Nr8_16QAM_rate_0_48.png": ("CDL-A", "UMA"),
    "Nt256_Nr8_64QAM_rate_0_7.png": ("CDL-A", "UMA"),
}
LEGACY_IMAGE_PATTERN = re.compile(
    r"^(?:(?P<metric>BER|SE)_)?(?:(?P<channel>[A-Za-z0-9-]+)_)?Nt(?P<tx>\d+)_Nr(?P<rx>\d+)_(?P<qam>[A-Za-z0-9]+)_rate_(?P<rate>\d+(?:_\d+)?)\.png$",
    re.IGNORECASE,
)
FULL_DIGITAL_IMAGE_PATTERN = re.compile(
    r"^(?P<label>(?:compare_full_digital_(?:svd_vs_gmd_thp|svd_gmd_thp_ucd|svd_gmd_ucd|ucd)|全数字(?:_SVD_GMD(?:THP)?_)?UCD对比|全数字UCD对比))_(?P<channel>[A-Za-z0-9-]+)_(?:k|K)(?P<users>\d+)_(?:nt|Nt)(?P<tx>\d+)_(?:nr|Nr)(?P<rx>\d+)(?:_(?:ns|Ns)(?P<ns>\d+))?.*?(?:(?:_m(?P<bits>\d+)_.*)|(?:(?P<qam>\d+)QAM_.*))\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PhotoEntry:
    user_scope: str
    precoding_family: str
    metric: str
    channel_model: str
    channel_estimation: str
    tx: int
    rx: int
    num_users: str
    num_rf_chains: str
    num_streams_per_user: str
    digital_power_constraint: str
    qam: str
    rate: str
    file: str
    path: Path

    @property
    def title(self) -> str:
        return (
            f"{self.user_scope} | {self.precoding_family} | {self.metric} | "
            f"{self.channel_model} | CSI={self.channel_estimation} | Nt={self.tx} Nr={self.rx} | "
            f"K={self.num_users} Nrf={self.num_rf_chains} Ns={self.num_streams_per_user} "
            f"Pdig={self.digital_power_constraint} | {self.qam} | rate {self.rate}"
        )


class LauncherGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mix Precode Launcher")
        self.root.geometry("1060x780")
        self.root.minsize(940, 700)
        self.root.configure(bg="#0f1722")

        self.photo_entries = self._discover_photo_entries()
        self.train_channel_options = self._discover_train_channel_options()

        self.mode_var = tk.StringVar(value="mode1")

        # Mode 1 (popup) variables: only existing configs.
        self.m1_user_scope_var = tk.StringVar()
        self.m1_precoding_family_var = tk.StringVar()
        self.m1_metric_var = tk.StringVar()
        self.m1_channel_var = tk.StringVar()
        self.m1_csi_var = tk.StringVar()
        self.m1_tx_var = tk.StringVar()
        self.m1_rx_var = tk.StringVar()
        self.m1_users_var = tk.StringVar()
        self.m1_nrf_var = tk.StringVar()
        self.m1_streams_var = tk.StringVar()
        self.m1_power_var = tk.StringVar()
        self.m1_qam_var = tk.StringVar()
        self.m1_rate_var = tk.StringVar()
        self.m1_status_var = tk.StringVar()
        self.m1_preview_original = None
        self.m1_preview_tk = None
        self._m1_resize_after_id: str | None = None

        # Mode 2 (train) variables: editable/custom.
        self.m2_task_var = tk.StringVar(value=next(iter(TRAIN_TASK_TO_SCRIPT)))
        self.m2_channel_var = tk.StringVar(
            value=self.train_channel_options[0] if self.train_channel_options else "cdl-a"
        )
        self.m2_tx_var = tk.StringVar(value="16")
        self.m2_rx_var = tk.StringVar(value="4")
        self.m2_users_var = tk.StringVar(value="2")
        self.m2_nrf_var = tk.StringVar(value="8")
        self.m2_streams_var = tk.StringVar(value="4")
        self.m2_power_var = tk.StringVar(value="8")
        self.m2_qam_var = tk.StringVar(value="64QAM")
        self.m2_csi_var = tk.StringVar(value=TRAIN_CSI_MODE_OPTIONS[0])
        self.m2_csi_nmse_var = tk.StringVar(value="-20")
        self.m2_status_var = tk.StringVar(value="未开始训练")

        self.train_process: subprocess.Popen[str] | None = None
        self._is_closing = False

        self._build_style()
        self._build_layout()
        self._refresh_mode1_options()
        self._switch_mode()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------
    # Data discovery
    # ------------------------
    @staticmethod
    def _normalize_channel_model(value: object) -> str:
        text = str(value or "").strip().upper().replace("_", "-")
        compact = re.sub(r"[^A-Z0-9]", "", text)
        if compact in {"CDLA", "SIONNACDLA"}:
            return "CDL-A"
        if compact in {"UMA", "SIONNAUMA"}:
            return "UMA"
        if compact.startswith("SIONNACDL") and len(compact) >= len("SIONNACDLA"):
            suffix = compact[-1]
            if suffix in {"A", "B", "C", "D", "E"}:
                return f"CDL-{suffix}"
        if compact.startswith("CDL") and len(compact) == 4:
            return f"CDL-{compact[-1]}"
        if compact.startswith("SIONNATDL") and len(compact) >= len("SIONNATDLA"):
            suffix = compact[len("SIONNATDL") :]
            if suffix:
                return f"TDL-{suffix}"
        if compact.startswith("TDL") and len(compact) > 3:
            return f"TDL-{compact[3:]}"
        return text or "CDL-A"

    @staticmethod
    def _normalize_channel_estimation(value: object) -> str:
        text = str(value or "").strip()
        compact = re.sub(r"[^a-z0-9]", "", text.lower())
        aliases = {
            "perfect": "Perfect CSI",
            "perfectcsi": "Perfect CSI",
            "noerror": "Perfect CSI",
            "gaussian": "Gaussian NMSE",
            "gaussiannmse": "Gaussian NMSE",
            "mmse": "MMSE FullCov",
            "mmsefullcov": "MMSE FullCov",
            "imperfectcsimmsefullcov": "MMSE FullCov",
            "na": "N/A",
            "none": "N/A",
        }
        return aliases.get(compact, text or "N/A")

    @staticmethod
    def _parse_positive_int(value: object) -> int | None:
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _validate_multiuser_topology(
        num_users: int,
        num_rx_antennas: int,
        num_streams_per_user: int,
        spatial_budget: int,
        spatial_budget_label: str,
    ) -> str | None:
        if num_streams_per_user > num_rx_antennas:
            return (
                "Ns / user must not exceed Nr. "
                f"Got Ns / user = {num_streams_per_user}, Nr = {num_rx_antennas}."
            )

        if num_streams_per_user > spatial_budget:
            return (
                f"Ns / user must not exceed {spatial_budget_label}. "
                f"Got Ns / user = {num_streams_per_user}, {spatial_budget_label} = {spatial_budget}."
            )

        total_streams = num_users * num_streams_per_user
        if total_streams > spatial_budget:
            return (
                f"Total streams must not exceed {spatial_budget_label}. "
                f"Got K * (Ns / user) = {total_streams}, {spatial_budget_label} = {spatial_budget}."
            )

        bd_requirement = (num_users - 1) * num_rx_antennas + num_streams_per_user
        if bd_requirement > spatial_budget:
            return (
                "Current multi-user setting leaves insufficient BD null-space. "
                f"Need (K - 1) * Nr + Ns <= {spatial_budget_label}, got "
                f"{bd_requirement} > {spatial_budget}."
            )

        return None

    @staticmethod
    def _normalize_rate(value: object) -> str:
        text = str(value or "").strip()
        decimal_match = re.match(r"^\s*0\s*[._]\s*(\d+)\s*$", text)
        if decimal_match:
            decimal_text = f"0.{decimal_match.group(1)}".rstrip("0").rstrip(".")
            return decimal_text or "0"
        numeric_match = re.match(r"^\s*\d+(?:\.\d+)?\s*$", text)
        if numeric_match:
            return str(float(text)).rstrip("0").rstrip(".") if "." in text else str(int(text))
        match = re.match(r"^\s*(\d+)\s*[/_]\s*(\d+)\s*$", text)
        if match:
            return f"{int(match.group(1))}/{int(match.group(2))}"
        return text or "N/A"

    @staticmethod
    def _normalize_metric(value: object, default: str = "SE") -> str:
        text = str(value or "").strip().upper()
        if text in {"BER", "SE"}:
            return text
        return default

    @staticmethod
    def _normalize_qam(value: object) -> str:
        text = str(value or "").strip().upper()
        return text or "N/A"

    @classmethod
    def _qam_from_bits(cls, bits_text: object) -> str:
        bits = cls._parse_positive_int(bits_text)
        if bits is None:
            return "N/A"
        return f"{2 ** bits}QAM"

    @staticmethod
    def _bits_from_qam_label(value: object) -> int | None:
        text = str(value or "").strip().upper().replace("-", "")
        if text == "QPSK":
            return 2
        match = re.match(r"^(?P<size>\d+)\s*QAM$", text)
        if match is not None:
            size = int(match.group("size"))
            size_to_bits = {4: 2, 16: 4, 64: 6, 256: 8}
            return size_to_bits.get(size)
        try:
            bits = int(text)
        except ValueError:
            return None
        return bits if bits in {2, 4, 6, 8} else None

    @staticmethod
    def _parse_positive_float(value: object) -> float | None:
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        return number if number > 0.0 else None

    @staticmethod
    def _format_compact_number(value: float) -> str:
        number = float(value)
        if abs(number - round(number)) <= 1e-9:
            return str(int(round(number)))
        return f"{number:g}"

    @classmethod
    def _normalize_optional_int_text(cls, value: object, default: str = "N/A") -> str:
        parsed = cls._parse_positive_int(value)
        return str(parsed) if parsed is not None else default

    @classmethod
    def _normalize_power_text(cls, value: object, default: str = "N/A") -> str:
        parsed = cls._parse_positive_float(value)
        return cls._format_compact_number(parsed) if parsed is not None else default

    @staticmethod
    def _extract_tag_positive_int(text: str, tag: str) -> int | None:
        match = re.search(rf"_{re.escape(tag)}(?P<value>\d+)", text, re.IGNORECASE)
        if match is None:
            return None
        try:
            value = int(match.group("value"))
        except ValueError:
            return None
        return value if value > 0 else None

    @classmethod
    def _extract_power_text(cls, text: str, default: str = "N/A") -> str:
        match = re.search(r"_pwr_(?P<value>[A-Za-z0-9mp]+)", text, re.IGNORECASE)
        if match is None:
            return default
        token = match.group("value").lower().replace("m", "-").replace("p", ".")
        parsed = cls._parse_positive_float(token)
        return cls._format_compact_number(parsed) if parsed is not None else default

    @classmethod
    def _derive_power_text(
        cls,
        num_users_text: str,
        num_streams_text: str,
        default: str = "N/A",
    ) -> str:
        num_users = cls._parse_positive_int(num_users_text)
        num_streams = cls._parse_positive_int(num_streams_text)
        if num_users is None or num_streams is None:
            return default
        return cls._format_compact_number(num_users * num_streams)

    @staticmethod
    def _channel_sort_key(value: str) -> tuple[int, str]:
        order = {
            "CDL-A": 0,
            "CDL-B": 1,
            "CDL-C": 2,
            "CDL-D": 3,
            "CDL-E": 4,
            "UMA": 5,
        }
        return (order.get(value, 99), value)

    @staticmethod
    def _metric_sort_key(value: str) -> tuple[int, str]:
        order = {"SE": 0, "BER": 1}
        return (order.get(value, 99), value)

    @staticmethod
    def _channel_estimation_sort_key(value: str) -> tuple[int, str]:
        order = {"Perfect CSI": 0, "Gaussian NMSE": 1, "MMSE FullCov": 2, "N/A": 99}
        return (order.get(value, 50), value)

    @staticmethod
    def _qam_sort_key(value: str) -> tuple[int, str]:
        order = {"QPSK": 0, "4QAM": 1, "16QAM": 2, "64QAM": 3, "256QAM": 4}
        text = str(value).strip().upper()
        return (order.get(text, 99), text)

    @staticmethod
    def _user_scope_sort_key(value: str) -> tuple[int, str]:
        order = {"Multi-User": 0, "Single-User": 1}
        return (order.get(value, 99), value)

    @staticmethod
    def _precoding_family_sort_key(value: str) -> tuple[int, str]:
        order = {"Hybrid": 0, "Digital": 1}
        return (order.get(value, 99), value)

    @staticmethod
    def _numeric_text_sort_key(value: str) -> tuple[int, float, str]:
        try:
            return (0, float(value), value)
        except (TypeError, ValueError):
            return (1, float("inf"), str(value))

    @staticmethod
    def _mode1_rate_sort_key(value: str) -> tuple[int, float, str]:
        text = str(value).strip().upper()
        combo_match = re.match(r"^(QPSK|4QAM|16QAM|64QAM|256QAM)\s+(\d+)\s*/\s*(\d+)$", text)
        if combo_match is not None:
            qam_order = {"QPSK": 0, "4QAM": 1, "16QAM": 2, "64QAM": 3, "256QAM": 4}
            num = int(combo_match.group(2))
            den = int(combo_match.group(3))
            return (0, qam_order.get(combo_match.group(1), 99), num / den, text)
        frac_match = re.match(r"^(\d+)\s*/\s*(\d+)$", text)
        if frac_match is not None:
            num = int(frac_match.group(1))
            den = int(frac_match.group(2))
            return (1, 0, num / den, text)
        try:
            return (1, 1, float(text), text)
        except ValueError:
            pass
        return (2, 0, float("inf"), text)

    def _catalog_photo_entries(self) -> list[PhotoEntry]:
        if not CATALOG_PATH.exists():
            return []
        try:
            raw_data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        items = raw_data.get("experiments", []) if isinstance(raw_data, dict) else raw_data
        if not isinstance(items, list):
            return []

        entries: list[PhotoEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file", "")).strip()
            if not file_name:
                continue
            image_path = PHOTO_DIR / file_name
            if image_path.suffix.lower() != ".png" or not image_path.exists():
                continue

            user_scope = str(item.get("user_scope", "Multi-User")).strip() or "Multi-User"
            precoding_family = str(item.get("precoding_family", "Hybrid")).strip() or "Hybrid"
            tx = self._parse_positive_int(item.get("tx_antennas"))
            rx = self._parse_positive_int(item.get("rx_antennas"))
            if tx is None or rx is None:
                continue
            num_users_text = self._normalize_optional_int_text(item.get("num_users"), default="2")
            num_rf_text = self._normalize_optional_int_text(
                item.get("num_rf_chains"),
                default=str(self._extract_tag_positive_int(file_name, "NRF") or "N/A"),
            )
            num_streams_text = self._normalize_optional_int_text(
                item.get("num_streams_per_user"),
                default=str(self._extract_tag_positive_int(file_name, "NS") or "N/A"),
            )
            power_text = self._normalize_power_text(item.get("digital_power_constraint"))
            if power_text == "N/A":
                power_text = self._derive_power_text(num_users_text, num_streams_text)

            entries.append(
                PhotoEntry(
                    user_scope=user_scope,
                    precoding_family=precoding_family,
                    metric=self._normalize_metric(item.get("metric"), default="SE"),
                    channel_model=self._normalize_channel_model(item.get("channel_model")),
                    channel_estimation=self._normalize_channel_estimation(item.get("channel_estimation")),
                    tx=tx,
                    rx=rx,
                    num_users=num_users_text,
                    num_rf_chains=num_rf_text,
                    num_streams_per_user=num_streams_text,
                    digital_power_constraint=power_text,
                    qam=self._normalize_qam(item.get("qam")),
                    rate=self._normalize_rate(item.get("coding_rate")),
                    file=image_path.name,
                    path=image_path,
                )
            )
        return entries

    def _full_digital_photo_entries(
        self,
        allowed_channel_models: set[str] | None = None,
    ) -> list[PhotoEntry]:
        if not FULL_DIGITAL_RESULTS_DIR.exists():
            return []

        entries: list[PhotoEntry] = []
        for image_path in sorted(FULL_DIGITAL_RESULTS_DIR.glob("*.png")):
            match = FULL_DIGITAL_IMAGE_PATTERN.match(image_path.name)
            if match is None:
                continue
            channel_model = self._normalize_channel_model(match.group("channel"))
            if allowed_channel_models and channel_model not in allowed_channel_models:
                continue

            entries.append(
                PhotoEntry(
                    user_scope="Multi-User",
                    precoding_family="Digital",
                    metric="SE",
                    channel_model=channel_model,
                    channel_estimation="N/A",
                    tx=int(match.group("tx")),
                    rx=int(match.group("rx")),
                    num_users=match.group("users"),
                    num_rf_chains=match.group("tx"),
                    num_streams_per_user=match.group("ns") or "N/A",
                    digital_power_constraint=(
                        self._extract_power_text(image_path.name)
                        if self._extract_power_text(image_path.name) != "N/A"
                        else self._derive_power_text(match.group("users"), match.group("ns") or "N/A")
                    ),
                    qam=(
                        self._qam_from_bits(match.group("bits"))
                        if match.group("bits")
                        else self._normalize_qam(f"{match.group('qam')}QAM")
                    ),
                    rate="N/A",
                    file=image_path.name,
                    path=image_path,
                )
            )
        return entries

    def _discover_photo_entries(self) -> list[PhotoEntry]:
        entries: list[PhotoEntry] = []
        catalog_entries = self._catalog_photo_entries()
        entries.extend(catalog_entries)
        catalog_files = {entry.file for entry in catalog_entries}
        hybrid_channels = {
            entry.channel_model
            for entry in catalog_entries
            if entry.user_scope == "Multi-User" and entry.precoding_family == "Hybrid"
        }
        entries.extend(self._full_digital_photo_entries(allowed_channel_models=hybrid_channels))
        if not PHOTO_DIR.exists():
            return entries

        for image_path in sorted(PHOTO_DIR.glob("*.png")):
            if image_path.name in catalog_files:
                continue
            match = LEGACY_IMAGE_PATTERN.match(image_path.name)
            if match is None:
                continue

            metric = (match.group("metric") or "BER").upper()
            qam = str(match.group("qam")).upper()
            rate = self._normalize_rate(match.group("rate"))
            channel_models = (
                [self._normalize_channel_model(match.group("channel"))]
                if match.group("channel")
                else list(LEGACY_SINGLE_USER_CHANNEL_BY_FILE.get(image_path.name, ("N/A",)))
            )
            for channel_model in channel_models:
                entries.append(
                    PhotoEntry(
                        user_scope="Single-User",
                        precoding_family="Digital",
                        metric=metric,
                        channel_model=channel_model,
                        channel_estimation="N/A",
                        tx=int(match.group("tx")),
                        rx=int(match.group("rx")),
                        num_users="N/A",
                        num_rf_chains="N/A",
                        num_streams_per_user="N/A",
                        digital_power_constraint="N/A",
                        qam=self._normalize_qam(qam),
                        rate=rate,
                        file=image_path.name,
                        path=image_path,
                    )
                )

        entries.sort(
            key=lambda x: (
                self._user_scope_sort_key(x.user_scope),
                self._precoding_family_sort_key(x.precoding_family),
                self._metric_sort_key(x.metric),
                self._channel_sort_key(x.channel_model),
                self._channel_estimation_sort_key(x.channel_estimation),
                x.tx,
                x.rx,
                self._numeric_text_sort_key(x.num_users),
                self._numeric_text_sort_key(x.num_rf_chains),
                self._numeric_text_sort_key(x.num_streams_per_user),
                self._numeric_text_sort_key(x.digital_power_constraint),
                x.qam,
                x.rate,
                x.file,
            )
        )
        return entries

    def _discover_train_channel_options(self) -> list[str]:
        options = ["cdl-a", "cdl-b", "cdl-c", "cdl-d", "uma"]
        try:
            discovered = [str(item).strip().lower() for item in list_supported_sionna_channel_types()]
        except Exception:
            discovered = []
        for item in discovered:
            if item and item not in options:
                options.append(item)
        return options

    # ------------------------
    # UI build
    # ------------------------
    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background="#0f1722")
        style.configure("Panel.TFrame", background="#1a2636")
        style.configure("Card.TFrame", background="#111b2a")
        style.configure(
            "Title.TLabel",
            background="#0f1722",
            foreground="#f2f7ff",
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background="#0f1722",
            foreground="#c4d4e8",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background="#111b2a",
            foreground="#edf4ff",
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background="#111b2a",
            foreground="#d6e4f5",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#111b2a",
            foreground="#eff6ff",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Mode.TRadiobutton",
            background="#1a2636",
            foreground="#e3eefc",
            font=("Microsoft YaHei UI", 10, "bold"),
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="模式1 弹图 / 模式2 训练", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="模式1下拉只显示当前已有图片配置；模式2参数可自己设置。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(4, 12))

        mode_bar = ttk.Frame(outer, style="Panel.TFrame", padding=12)
        mode_bar.pack(fill="x")
        ttk.Radiobutton(
            mode_bar,
            text="模式1：弹出图片",
            value="mode1",
            variable=self.mode_var,
            command=self._switch_mode,
            style="Mode.TRadiobutton",
        ).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(
            mode_bar,
            text="模式2：开始训练",
            value="mode2",
            variable=self.mode_var,
            command=self._switch_mode,
            style="Mode.TRadiobutton",
        ).pack(side="left")

        self.mode_container = ttk.Frame(outer, style="App.TFrame")
        self.mode_container.pack(fill="both", expand=True, pady=(14, 0))
        self.mode_container.columnconfigure(0, weight=1)
        self.mode_container.rowconfigure(0, weight=1)

        self.mode1_panel = self._build_mode1_panel(self.mode_container)
        self.mode2_panel = self._build_mode2_panel(self.mode_container)

    def _build_mode1_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        card = ttk.Frame(panel, style="Card.TFrame", padding=16)
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)
        card.columnconfigure(5, weight=1)
        card.columnconfigure(7, weight=1)
        card.columnconfigure(9, weight=1)
        card.columnconfigure(11, weight=1)
        card.rowconfigure(6, weight=1)

        ttk.Label(card, text="模式1：单用户 / 多用户图片弹出", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=10, sticky="w"
        )

        ttk.Label(card, text="System", style="CardText.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.m1_user_scope_combo = ttk.Combobox(card, textvariable=self.m1_user_scope_var, state="readonly", width=14)
        self.m1_user_scope_combo.grid(row=1, column=1, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_user_scope_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_user_scope_combo)

        ttk.Label(card, text="Precoding", style="CardText.TLabel").grid(row=1, column=2, sticky="w", pady=(10, 0))
        self.m1_precoding_family_combo = ttk.Combobox(
            card,
            textvariable=self.m1_precoding_family_var,
            state="readonly",
            width=14,
        )
        self.m1_precoding_family_combo.grid(row=1, column=3, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_precoding_family_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_precoding_family_combo)

        ttk.Label(card, text="Metric", style="CardText.TLabel").grid(row=1, column=4, sticky="w", pady=(10, 0))
        self.m1_metric_combo = ttk.Combobox(card, textvariable=self.m1_metric_var, state="readonly", width=12)
        self.m1_metric_combo.grid(row=1, column=5, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_metric_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_metric_combo)

        ttk.Label(card, text="Channel", style="CardText.TLabel").grid(row=1, column=6, sticky="w", pady=(10, 0))
        self.m1_channel_combo = ttk.Combobox(card, textvariable=self.m1_channel_var, state="readonly", width=12)
        self.m1_channel_combo.grid(row=1, column=7, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.m1_channel_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_channel_combo)

        ttk.Label(card, text="Nt", style="CardText.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.m1_tx_combo = ttk.Combobox(card, textvariable=self.m1_tx_var, state="readonly", width=10)
        self.m1_tx_combo.grid(row=2, column=1, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_tx_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_tx_combo)

        ttk.Label(card, text="Nr", style="CardText.TLabel").grid(row=2, column=2, sticky="w", pady=(10, 0))
        self.m1_rx_combo = ttk.Combobox(card, textvariable=self.m1_rx_var, state="readonly", width=10)
        self.m1_rx_combo.grid(row=2, column=3, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_rx_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_rx_combo)

        ttk.Label(card, text="QAM", style="CardText.TLabel").grid(row=2, column=4, sticky="w", pady=(10, 0))
        self.m1_qam_combo = ttk.Combobox(card, textvariable=self.m1_qam_var, state="readonly", width=12)
        self.m1_qam_combo.grid(row=2, column=5, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_qam_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_qam_combo)

        ttk.Label(card, text="Rate", style="CardText.TLabel").grid(row=2, column=6, sticky="w", pady=(10, 0))
        self.m1_rate_combo = ttk.Combobox(card, textvariable=self.m1_rate_var, state="readonly", width=10)
        self.m1_rate_combo.grid(row=2, column=7, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.m1_rate_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_rate_combo)

        ttk.Label(card, text="CSI", style="CardText.TLabel").grid(row=2, column=8, sticky="w", padx=(12, 0), pady=(10, 0))
        self.m1_csi_combo = ttk.Combobox(card, textvariable=self.m1_csi_var, state="readonly", width=16)
        self.m1_csi_combo.grid(row=2, column=9, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.m1_csi_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_csi_combo)

        ttk.Label(card, text="Users", style="CardText.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.m1_users_combo = ttk.Combobox(card, textvariable=self.m1_users_var, state="readonly", width=10)
        self.m1_users_combo.grid(row=3, column=1, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_users_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_users_combo)

        ttk.Label(card, text="Nrf", style="CardText.TLabel").grid(row=3, column=2, sticky="w", pady=(10, 0))
        self.m1_nrf_combo = ttk.Combobox(card, textvariable=self.m1_nrf_var, state="readonly", width=10)
        self.m1_nrf_combo.grid(row=3, column=3, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_nrf_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_nrf_combo)

        ttk.Label(card, text="Ns", style="CardText.TLabel").grid(row=3, column=4, sticky="w", pady=(10, 0))
        self.m1_streams_combo = ttk.Combobox(card, textvariable=self.m1_streams_var, state="readonly", width=10)
        self.m1_streams_combo.grid(row=3, column=5, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m1_streams_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_streams_combo)

        ttk.Label(card, text="Pdig", style="CardText.TLabel").grid(row=3, column=6, sticky="w", pady=(10, 0))
        self.m1_power_combo = ttk.Combobox(card, textvariable=self.m1_power_var, state="readonly", width=10)
        self.m1_power_combo.grid(row=3, column=7, sticky="ew", padx=(8, 0), pady=(10, 0))
        self.m1_power_combo.bind("<<ComboboxSelected>>", self._on_mode1_filter_changed)
        self._disable_mouse_wheel(self.m1_power_combo)

        button_row = ttk.Frame(card, style="Card.TFrame")
        button_row.grid(row=4, column=0, columnspan=10, sticky="ew", pady=(14, 0))
        button_row.columnconfigure(0, weight=1)

        ttk.Button(
            button_row,
            text="Match Photo",
            command=self._match_mode1_photo,
        ).grid(row=0, column=0, sticky="ew")

        ttk.Label(
            card,
            textvariable=self.m1_status_var,
            style="Status.TLabel",
            justify="left",
            wraplength=900,
        ).grid(row=5, column=0, columnspan=10, sticky="w", pady=(12, 0))

        self.m1_image_container = tk.Frame(
            card,
            bg="#0b1522",
            highlightbackground="#33455c",
            highlightthickness=1,
        )
        self.m1_image_container.grid(row=6, column=0, columnspan=10, sticky="nsew", pady=(12, 0))
        self.m1_image_container.bind("<Configure>", self._on_mode1_image_container_resize)

        self.m1_image_label = tk.Label(
            self.m1_image_container,
            bg="#0b1522",
            fg="#eef5ff",
            text="Image preview area",
            font=("Microsoft YaHei UI", 11),
        )
        self.m1_image_label.pack(fill="both", expand=True, padx=12, pady=12)

        return panel

    def _build_mode2_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        control_card = ttk.Frame(panel, style="Card.TFrame", padding=16)
        control_card.grid(row=0, column=0, sticky="ew")
        control_card.columnconfigure(1, weight=1)
        control_card.columnconfigure(3, weight=1)

        ttk.Label(control_card, text="模式2：当场训练（可自定义）", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w"
        )

        ttk.Label(control_card, text="训练任务", style="CardText.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.m2_task_combo = ttk.Combobox(
            control_card,
            textvariable=self.m2_task_var,
            state="readonly",
            values=list(TRAIN_TASK_TO_SCRIPT.keys()),
            width=26,
        )
        self.m2_task_combo.grid(row=1, column=1, sticky="ew", padx=(8, 12), pady=(10, 0))
        self.m2_task_combo.bind("<<ComboboxSelected>>", self._on_mode2_task_changed)
        self._disable_mouse_wheel(self.m2_task_combo)

        ttk.Label(control_card, text="信道库", style="CardText.TLabel").grid(row=1, column=2, sticky="w", pady=(10, 0))
        self.m2_channel_combo = ttk.Combobox(
            control_card,
            textvariable=self.m2_channel_var,
            state="readonly",
            values=self.train_channel_options,
            width=34,
        )
        self.m2_channel_combo.grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(10, 0))
        self._disable_mouse_wheel(self.m2_channel_combo)

        ttk.Label(control_card, text="输入天线 Nt", style="CardText.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.m2_tx_entry = ttk.Entry(control_card, textvariable=self.m2_tx_var, width=16)
        self.m2_tx_entry.grid(row=2, column=1, sticky="w", padx=(8, 12), pady=(10, 0))
        self._disable_mouse_wheel(self.m2_tx_entry)

        ttk.Label(control_card, text="输出天线 Nr", style="CardText.TLabel").grid(row=2, column=2, sticky="w", pady=(10, 0))
        self.m2_rx_entry = ttk.Entry(control_card, textvariable=self.m2_rx_var, width=16)
        self.m2_rx_entry.grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(10, 0))
        self._disable_mouse_wheel(self.m2_rx_entry)

        self.m2_users_label = ttk.Label(control_card, text="Users", style="CardText.TLabel")
        self.m2_users_entry = ttk.Entry(control_card, textvariable=self.m2_users_var, width=16)
        self._disable_mouse_wheel(self.m2_users_entry)
        self.m2_users_label.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.m2_users_entry.grid(row=3, column=1, sticky="w", padx=(8, 12), pady=(10, 0))

        self.m2_streams_label = ttk.Label(control_card, text="Ns / user", style="CardText.TLabel")
        self.m2_streams_entry = ttk.Entry(control_card, textvariable=self.m2_streams_var, width=16)
        self._disable_mouse_wheel(self.m2_streams_entry)
        self.m2_streams_label.grid(row=3, column=2, sticky="w", pady=(10, 0))
        self.m2_streams_entry.grid(row=3, column=3, sticky="w", padx=(8, 0), pady=(10, 0))

        self.m2_nrf_label = ttk.Label(control_card, text="Nrf", style="CardText.TLabel")
        self.m2_nrf_entry = ttk.Entry(control_card, textvariable=self.m2_nrf_var, width=16)
        self._disable_mouse_wheel(self.m2_nrf_entry)
        self.m2_nrf_label.grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.m2_nrf_entry.grid(row=4, column=1, sticky="w", padx=(8, 12), pady=(10, 0))

        self.m2_power_label = ttk.Label(control_card, text="Pdig", style="CardText.TLabel")
        self.m2_power_entry = ttk.Entry(control_card, textvariable=self.m2_power_var, width=16)
        self._disable_mouse_wheel(self.m2_power_entry)
        self.m2_power_label.grid(row=4, column=2, sticky="w", pady=(10, 0))
        self.m2_power_entry.grid(row=4, column=3, sticky="w", padx=(8, 0), pady=(10, 0))

        ttk.Label(control_card, text="QAM", style="CardText.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.m2_qam_combo = ttk.Combobox(
            control_card,
            textvariable=self.m2_qam_var,
            state="readonly",
            values=TRAIN_QAM_OPTIONS,
            width=16,
        )
        self.m2_qam_combo.grid(row=5, column=1, sticky="w", padx=(8, 12), pady=(10, 0))
        self._disable_mouse_wheel(self.m2_qam_combo)

        self.m2_csi_label = ttk.Label(control_card, text="信道误差", style="CardText.TLabel")
        self.m2_csi_combo = ttk.Combobox(
            control_card,
            textvariable=self.m2_csi_var,
            state="readonly",
            values=TRAIN_CSI_MODE_OPTIONS,
            width=24,
        )
        self.m2_csi_combo.bind("<<ComboboxSelected>>", self._on_mode2_csi_changed)
        self._disable_mouse_wheel(self.m2_csi_combo)
        self.m2_csi_label.grid(row=5, column=2, sticky="w", pady=(10, 0))
        self.m2_csi_combo.grid(row=5, column=3, sticky="ew", padx=(8, 0), pady=(10, 0))

        self.m2_csi_nmse_label = ttk.Label(control_card, text="CSI NMSE (dB)", style="CardText.TLabel")
        self.m2_csi_nmse_entry = ttk.Entry(control_card, textvariable=self.m2_csi_nmse_var, width=16)
        self._disable_mouse_wheel(self.m2_csi_nmse_entry)
        self.m2_csi_nmse_label.grid(row=6, column=2, sticky="w", pady=(10, 0))
        self.m2_csi_nmse_entry.grid(row=6, column=3, sticky="w", padx=(8, 0), pady=(10, 0))

        button_row = ttk.Frame(control_card, style="Card.TFrame")
        button_row.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        button_row.columnconfigure(2, weight=1)

        self.start_train_button = ttk.Button(button_row, text="开始训练", command=self._start_training)
        self.start_train_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_train_button = ttk.Button(
            button_row,
            text="停止训练",
            command=self._stop_training,
            state="disabled",
        )
        self.stop_train_button.grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Button(
            button_row,
            text="打开结果目录",
            command=self._open_selected_result_dir,
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        ttk.Label(
            control_card,
            textvariable=self.m2_status_var,
            style="Status.TLabel",
            justify="left",
            wraplength=900,
        ).grid(row=8, column=0, columnspan=4, sticky="w", pady=(12, 0))

        log_card = ttk.Frame(panel, style="Card.TFrame", padding=12)
        log_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(1, weight=1)

        ttk.Label(log_card, text="训练日志", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")

        log_box = ttk.Frame(log_card, style="Card.TFrame")
        log_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_box,
            bg="#0b1522",
            fg="#edf4ff",
            insertbackground="#edf4ff",
            relief="flat",
            wrap="word",
            font=("Consolas", 10),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._refresh_mode2_field_visibility()
        return panel

    # ------------------------
    # Common helpers
    # ------------------------
    def _disable_mouse_wheel(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", lambda _e: "break")
        widget.bind("<Button-4>", lambda _e: "break")
        widget.bind("<Button-5>", lambda _e: "break")

    def _switch_mode(self) -> None:
        mode = self.mode_var.get()
        if mode == "mode1":
            self.mode2_panel.grid_remove()
            self.mode1_panel.grid()
            return
        self.mode1_panel.grid_remove()
        self.mode2_panel.grid()

    def _selected_train_script(self) -> Path | None:
        return TRAIN_TASK_TO_SCRIPT.get(self.m2_task_var.get().strip())

    def _selected_task_is_hybrid(self) -> bool:
        return self._selected_train_script() == HYBRID_TRAIN_SCRIPT

    def _selected_task_is_digital(self) -> bool:
        return self._selected_train_script() == DIGITAL_TRAIN_SCRIPT

    def _on_mode2_task_changed(self, _event: object | None = None) -> None:
        self._refresh_mode2_field_visibility()

    def _on_mode2_csi_changed(self, _event: object | None = None) -> None:
        self._refresh_mode2_field_visibility()

    def _selected_training_csi_mode(self) -> str:
        return TRAIN_CSI_MODE_TO_ARG.get(self.m2_csi_var.get().strip(), "mmse_fullcov")

    def _refresh_mode2_field_visibility(self) -> None:
        is_hybrid = self._selected_task_is_hybrid()
        self.m2_users_label.grid()
        self.m2_users_entry.grid()
        self.m2_streams_label.grid()
        self.m2_streams_entry.grid()
        if is_hybrid:
            self.m2_nrf_label.grid()
            self.m2_nrf_entry.grid()
            self.m2_csi_label.grid()
            self.m2_csi_combo.grid()
            if self._selected_training_csi_mode() == "gaussian":
                self.m2_csi_nmse_label.grid()
                self.m2_csi_nmse_entry.grid()
            else:
                self.m2_csi_nmse_label.grid_remove()
                self.m2_csi_nmse_entry.grid_remove()
            return
        self.m2_nrf_label.grid_remove()
        self.m2_nrf_entry.grid_remove()
        self.m2_csi_label.grid_remove()
        self.m2_csi_combo.grid_remove()
        self.m2_csi_nmse_label.grid_remove()
        self.m2_csi_nmse_entry.grid_remove()

    @staticmethod
    def _open_with_system_viewer(path: Path) -> bool:
        if not path.exists() or path.is_dir():
            return False
        if not hasattr(os, "startfile"):
            return False
        try:
            os.startfile(str(path))
            return True
        except OSError:
            return False

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _clear_mode1_preview(self, text: str) -> None:
        self.m1_preview_original = None
        self.m1_preview_tk = None
        self.m1_image_label.configure(image="", text=text)

    def _on_mode1_image_container_resize(self, _event: object) -> None:
        if self._m1_resize_after_id is not None:
            self.root.after_cancel(self._m1_resize_after_id)
        self._m1_resize_after_id = self.root.after(80, self._render_mode1_preview)

    def _render_mode1_preview(self) -> None:
        if self.m1_preview_original is None:
            return
        if ImageOps is None or ImageTk is None:
            self.m1_image_label.configure(image="", text="Pillow not available.")
            return

        width = max(self.m1_image_container.winfo_width() - 24, 200)
        height = max(self.m1_image_container.winfo_height() - 24, 200)
        resized = ImageOps.contain(self.m1_preview_original.copy(), (width, height))
        self.m1_preview_tk = ImageTk.PhotoImage(resized)
        self.m1_image_label.configure(image=self.m1_preview_tk, text="")

    def _set_mode1_preview_image(self, image_obj) -> None:
        self.m1_preview_original = image_obj
        self._render_mode1_preview()

    @staticmethod
    def _load_mode1_image(path: Path):
        if Image is None:
            return None
        try:
            with Image.open(path) as image_obj:
                rgba_image = image_obj.convert("RGBA")
                alpha_min, _alpha_max = rgba_image.getchannel("A").getextrema()
                if alpha_min == 255:
                    return rgba_image

                background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
                background.alpha_composite(rgba_image)
                return background
        except OSError:
            return None

    # ------------------------
    # Mode 1 (popup existing configs)
    # ------------------------
    def _mode1_matching_entries(
        self,
        ignore_field: str | None = None,
        ignore_fields: set[str] | None = None,
    ) -> list[PhotoEntry]:
        matched = self.photo_entries
        ignored = set(ignore_fields or ())
        if ignore_field is not None:
            ignored.add(ignore_field)
        filters = {
            "user_scope": self.m1_user_scope_var.get().strip(),
            "precoding_family": self.m1_precoding_family_var.get().strip(),
            "metric": self.m1_metric_var.get().strip(),
            "channel": self.m1_channel_var.get().strip(),
            "csi": self.m1_csi_var.get().strip(),
            "tx": self.m1_tx_var.get().strip(),
            "rx": self.m1_rx_var.get().strip(),
            "users": self.m1_users_var.get().strip(),
            "nrf": self.m1_nrf_var.get().strip(),
            "streams": self.m1_streams_var.get().strip(),
            "power": self.m1_power_var.get().strip(),
            "qam": self.m1_qam_var.get().strip(),
            "rate": self.m1_rate_var.get().strip(),
        }

        for field, selected in filters.items():
            if field in ignored or not selected or selected == ALL_OPTION:
                continue
            if field == "user_scope":
                matched = [item for item in matched if item.user_scope == selected]
            elif field == "precoding_family":
                matched = [item for item in matched if item.precoding_family == selected]
            elif field == "tx":
                matched = [item for item in matched if str(item.tx) == selected]
            elif field == "rx":
                matched = [item for item in matched if str(item.rx) == selected]
            elif field == "metric":
                matched = [item for item in matched if item.metric == selected]
            elif field == "channel":
                matched = [item for item in matched if item.channel_model == selected]
            elif field == "csi":
                matched = [item for item in matched if item.channel_estimation == selected]
            elif field == "users":
                matched = [item for item in matched if item.num_users == selected]
            elif field == "nrf":
                matched = [item for item in matched if item.num_rf_chains == selected]
            elif field == "streams":
                matched = [item for item in matched if item.num_streams_per_user == selected]
            elif field == "power":
                matched = [item for item in matched if item.digital_power_constraint == selected]
            elif field == "qam":
                matched = [item for item in matched if item.qam == selected]
            elif field == "rate":
                matched = [item for item in matched if item.rate == selected]
        return matched

    @staticmethod
    def _mode1_entry_value(item: PhotoEntry, field: str) -> str:
        if field == "user_scope":
            return item.user_scope
        if field == "precoding_family":
            return item.precoding_family
        if field == "metric":
            return item.metric
        if field == "channel":
            return item.channel_model
        if field == "csi":
            return item.channel_estimation
        if field == "tx":
            return str(item.tx)
        if field == "rx":
            return str(item.rx)
        if field == "users":
            return item.num_users
        if field == "nrf":
            return item.num_rf_chains
        if field == "streams":
            return item.num_streams_per_user
        if field == "power":
            return item.digital_power_constraint
        if field == "qam":
            return item.qam
        if field == "rate":
            return item.rate
        return ""

    def _mode1_field_from_widget(self, widget: object) -> str | None:
        mapping = (
            (self.m1_user_scope_combo, "user_scope"),
            (self.m1_precoding_family_combo, "precoding_family"),
            (self.m1_metric_combo, "metric"),
            (self.m1_channel_combo, "channel"),
            (self.m1_csi_combo, "csi"),
            (self.m1_tx_combo, "tx"),
            (self.m1_rx_combo, "rx"),
            (self.m1_users_combo, "users"),
            (self.m1_nrf_combo, "nrf"),
            (self.m1_streams_combo, "streams"),
            (self.m1_power_combo, "power"),
            (self.m1_qam_combo, "qam"),
            (self.m1_rate_combo, "rate"),
        )
        for combo, field in mapping:
            if widget is combo:
                return field
        return None

    def _mode1_selected_value(self, field: str) -> str:
        variables = {
            "user_scope": self.m1_user_scope_var,
            "precoding_family": self.m1_precoding_family_var,
            "metric": self.m1_metric_var,
            "channel": self.m1_channel_var,
            "csi": self.m1_csi_var,
            "tx": self.m1_tx_var,
            "rx": self.m1_rx_var,
            "users": self.m1_users_var,
            "nrf": self.m1_nrf_var,
            "streams": self.m1_streams_var,
            "power": self.m1_power_var,
            "qam": self.m1_qam_var,
            "rate": self.m1_rate_var,
        }
        variable = variables.get(field)
        return variable.get().strip() if variable is not None else ""

    def _mode1_filter_variables(self) -> dict[str, tk.StringVar]:
        return {
            "user_scope": self.m1_user_scope_var,
            "precoding_family": self.m1_precoding_family_var,
            "metric": self.m1_metric_var,
            "channel": self.m1_channel_var,
            "csi": self.m1_csi_var,
            "tx": self.m1_tx_var,
            "rx": self.m1_rx_var,
            "users": self.m1_users_var,
            "nrf": self.m1_nrf_var,
            "streams": self.m1_streams_var,
            "power": self.m1_power_var,
            "qam": self.m1_qam_var,
            "rate": self.m1_rate_var,
        }

    @staticmethod
    def _mode1_filter_order() -> tuple[str, ...]:
        return (
            "user_scope",
            "precoding_family",
            "metric",
            "channel",
            "csi",
            "tx",
            "rx",
            "users",
            "nrf",
            "streams",
            "power",
            "qam",
            "rate",
        )

    def _mode1_downstream_fields(self, field: str, include_self: bool = False) -> set[str]:
        order = self._mode1_filter_order()
        if field not in order:
            return set()
        start = order.index(field) if include_self else order.index(field) + 1
        return set(order[start:])

    def _clear_mode1_downstream_filters(self, field: str) -> None:
        variables = self._mode1_filter_variables()
        for downstream_field in self._mode1_downstream_fields(field):
            variables[downstream_field].set("")

    def _mode1_entries_for_combo(self, field: str) -> list[PhotoEntry]:
        return self._mode1_matching_entries(
            ignore_fields=self._mode1_downstream_fields(field, include_self=True)
        )

    def _set_mode1_filters_from_entry(self, entry: PhotoEntry) -> None:
        self.m1_user_scope_var.set(entry.user_scope)
        self.m1_precoding_family_var.set(entry.precoding_family)
        self.m1_metric_var.set(entry.metric)
        self.m1_channel_var.set(entry.channel_model)
        self.m1_csi_var.set(entry.channel_estimation)
        self.m1_tx_var.set(str(entry.tx))
        self.m1_rx_var.set(str(entry.rx))
        self.m1_users_var.set(entry.num_users)
        self.m1_nrf_var.set(entry.num_rf_chains)
        self.m1_streams_var.set(entry.num_streams_per_user)
        self.m1_power_var.set(entry.digital_power_constraint)
        self.m1_qam_var.set(entry.qam)
        self.m1_rate_var.set(entry.rate)

    def _reset_mode1_to_selected_field(self, field: str) -> None:
        fallback_entries = self._mode1_matching_entries(
            ignore_fields=self._mode1_downstream_fields(field)
        )
        if fallback_entries:
            self._set_mode1_filters_from_entry(fallback_entries[0])

    def _refresh_mode1_combo(
        self,
        field: str,
        combo: ttk.Combobox,
        var: tk.StringVar,
    ) -> None:
        values: list[str]
        if field == "user_scope":
            values = sorted(
                {item.user_scope for item in self.photo_entries},
                key=self._user_scope_sort_key,
            )
        else:
            entries = self._mode1_entries_for_combo(field)
            if field == "precoding_family":
                values = sorted({item.precoding_family for item in entries}, key=self._precoding_family_sort_key)
            else:
                if field == "metric":
                    values = sorted({item.metric for item in entries}, key=self._metric_sort_key)
                elif field == "channel":
                    values = sorted({item.channel_model for item in entries}, key=self._channel_sort_key)
                elif field == "csi":
                    values = sorted({item.channel_estimation for item in entries}, key=self._channel_estimation_sort_key)
                elif field == "tx":
                    values = [str(v) for v in sorted({item.tx for item in entries})]
                elif field == "rx":
                    values = [str(v) for v in sorted({item.rx for item in entries})]
                elif field == "users":
                    values = sorted({item.num_users for item in entries}, key=self._numeric_text_sort_key)
                elif field == "nrf":
                    values = sorted({item.num_rf_chains for item in entries}, key=self._numeric_text_sort_key)
                elif field == "streams":
                    values = sorted({item.num_streams_per_user for item in entries}, key=self._numeric_text_sort_key)
                elif field == "power":
                    values = sorted({item.digital_power_constraint for item in entries}, key=self._numeric_text_sort_key)
                elif field == "qam":
                    values = sorted({item.qam for item in entries}, key=self._qam_sort_key)
                elif field == "rate":
                    values = sorted({item.rate for item in entries}, key=self._mode1_rate_sort_key)
                else:
                    values = []

        combo["values"] = values
        if var.get() not in values:
            var.set(values[0] if values else "")

    def _refresh_mode1_options(self) -> None:
        self._refresh_mode1_combo("user_scope", self.m1_user_scope_combo, self.m1_user_scope_var)
        self._refresh_mode1_combo(
            "precoding_family",
            self.m1_precoding_family_combo,
            self.m1_precoding_family_var,
        )
        self._refresh_mode1_combo("metric", self.m1_metric_combo, self.m1_metric_var)
        self._refresh_mode1_combo("channel", self.m1_channel_combo, self.m1_channel_var)
        self._refresh_mode1_combo("csi", self.m1_csi_combo, self.m1_csi_var)
        self._refresh_mode1_combo("tx", self.m1_tx_combo, self.m1_tx_var)
        self._refresh_mode1_combo("rx", self.m1_rx_combo, self.m1_rx_var)
        self._refresh_mode1_combo("users", self.m1_users_combo, self.m1_users_var)
        self._refresh_mode1_combo("nrf", self.m1_nrf_combo, self.m1_nrf_var)
        self._refresh_mode1_combo("streams", self.m1_streams_combo, self.m1_streams_var)
        self._refresh_mode1_combo("power", self.m1_power_combo, self.m1_power_var)
        self._refresh_mode1_combo("qam", self.m1_qam_combo, self.m1_qam_var)
        self._refresh_mode1_combo("rate", self.m1_rate_combo, self.m1_rate_var)
        self._update_mode1_status()

    def _on_mode1_filter_changed(self, event: object) -> None:
        changed_field = self._mode1_field_from_widget(getattr(event, "widget", None))
        if changed_field is not None:
            self._clear_mode1_downstream_filters(changed_field)
        matched = self._mode1_matching_entries()
        if changed_field is not None and not matched:
            self._reset_mode1_to_selected_field(changed_field)
        self._refresh_mode1_options()

    def _update_mode1_status(self) -> None:
        matched = self._mode1_matching_entries()
        if not self.photo_entries:
            self.m1_status_var.set(f"No recognizable image configs found in {PHOTO_DIR}.")
            self._clear_mode1_preview("No images found.")
            return
        if not matched:
            self.m1_status_var.set("No matching images for current Match Photo filters.")
            self._clear_mode1_preview("No matched image.")
            return
        first = matched[0]
        image_obj = self._load_mode1_image(first.path)
        if image_obj is None:
            self._clear_mode1_preview(f"Failed to load: {first.file}")
            self.m1_status_var.set(f"Matched {len(matched)} image(s), but failed to load {first.file}.")
            return
        self._set_mode1_preview_image(image_obj)
        self.m1_status_var.set(
            f"Matched {len(matched)} image(s) | "
            f"{first.user_scope} | {first.precoding_family} | {first.channel_model} | "
            f"CSI={first.channel_estimation} | "
            f"K={first.num_users} Nrf={first.num_rf_chains} Ns={first.num_streams_per_user} "
            f"Pdig={first.digital_power_constraint} | {first.file}"
        )

    def _match_mode1_photo(self) -> None:
        matched = self._mode1_matching_entries()
        if not matched:
            messagebox.showwarning("No Match", "No images match current Match Photo filters.")
            return
        selected = matched[0]
        image_obj = self._load_mode1_image(selected.path)
        if image_obj is None:
            messagebox.showerror("Match Failed", f"Cannot load image:\n{selected.path}")
            return
        self._set_mode1_preview_image(image_obj)
        self.m1_status_var.set(
            f"Matched photo | {selected.user_scope} | {selected.precoding_family} | "
            f"{selected.channel_model} | CSI={selected.channel_estimation} | "
            f"K={selected.num_users} Nrf={selected.num_rf_chains} "
            f"Ns={selected.num_streams_per_user} Pdig={selected.digital_power_constraint} | "
            f"{selected.file}"
        )
        if not self._open_with_system_viewer(selected.path):
            messagebox.showwarning("Open Failed", f"Cannot open image:\n{selected.path}")

    # ------------------------
    # Mode 2 (training custom config)
    # ------------------------
    def _start_training(self) -> None:
        if self.train_process is not None and self.train_process.poll() is None:
            messagebox.showwarning("训练中", "已有训练任务在运行，请先停止或等待结束。")
            return

        task = self.m2_task_var.get().strip()
        script_path = TRAIN_TASK_TO_SCRIPT.get(task)
        if script_path is None or not script_path.exists():
            messagebox.showerror("脚本不存在", f"找不到训练脚本：{script_path}")
            return

        try:
            tx = int(self.m2_tx_var.get().strip())
            rx = int(self.m2_rx_var.get().strip())
            if tx <= 0 or rx <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "模式2的 Nt/Nr 必须是正整数。")
            return

        try:
            power_constraint = float(self.m2_power_var.get().strip())
            if power_constraint <= 0.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Parameter Error", "Pdig must be a positive number.")
            return

        qam_label = self.m2_qam_var.get().strip()
        bits_per_symbol = self._bits_from_qam_label(qam_label)
        if bits_per_symbol is None:
            messagebox.showerror("Parameter Error", "QAM must be one of QPSK, 4QAM, 16QAM, 64QAM, or 256QAM.")
            return

        is_hybrid = script_path == HYBRID_TRAIN_SCRIPT
        extra_summary_parts: list[str] = []
        csi_mode: str | None = None
        csi_nmse_db: float | None = None

        channel_type = self.m2_channel_var.get().strip().lower()
        if not channel_type:
            channel_type = "cdl-a"

        command = [
            sys.executable,
            str(script_path),
            "--num-tx-antennas",
            str(tx),
            "--num-rx-antennas",
            str(rx),
            "--channel-type",
            channel_type,
            "--digital-power-constraint",
            self._format_compact_number(power_constraint),
            "--bits-per-symbol",
            str(bits_per_symbol),
        ]

        if is_hybrid:
            try:
                num_users = int(self.m2_users_var.get().strip())
                num_streams = int(self.m2_streams_var.get().strip())
                num_rf_chains = int(self.m2_nrf_var.get().strip())
                if num_users <= 0 or num_streams <= 0 or num_rf_chains <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Parameter Error",
                    "Users, Ns / user, and Nrf must be positive integers.",
                )
                return

            csi_mode = self._selected_training_csi_mode()
            if csi_mode == "gaussian":
                try:
                    csi_nmse_db = float(self.m2_csi_nmse_var.get().strip())
                except ValueError:
                    messagebox.showerror("Parameter Error", "CSI NMSE must be a valid dB number.")
                    return

            topology_error = self._validate_multiuser_topology(
                num_users=num_users,
                num_rx_antennas=rx,
                num_streams_per_user=num_streams,
                spatial_budget=num_rf_chains,
                spatial_budget_label="Nrf",
            )
            if topology_error is not None:
                messagebox.showerror("Parameter Error", topology_error)
                return
            command.extend(
                [
                    "--num-users",
                    str(num_users),
                    "--num-streams-per-user",
                    str(num_streams),
                    "--num-rf-chains",
                    str(num_rf_chains),
                    "--mode",
                    csi_mode,
                ]
            )
            if csi_nmse_db is not None:
                command.extend(["--csi-nmse-db", self._format_compact_number(csi_nmse_db)])
            extra_summary_parts.extend(
                [
                    f"K={num_users}",
                    f"Ns={num_streams}",
                    f"Nrf={num_rf_chains}",
                    f"Pdig={self._format_compact_number(power_constraint)}",
                    f"QAM={2 ** bits_per_symbol}QAM",
                    f"CSI={csi_mode}",
                ]
            )
        else:
            try:
                num_users = int(self.m2_users_var.get().strip())
                num_streams = int(self.m2_streams_var.get().strip())
                if num_users <= 0 or num_streams <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Parameter Error",
                    "Users and Ns / user must be positive integers.",
                )
                return

            topology_error = self._validate_multiuser_topology(
                num_users=num_users,
                num_rx_antennas=rx,
                num_streams_per_user=num_streams,
                spatial_budget=tx,
                spatial_budget_label="Nt",
            )
            if topology_error is not None:
                messagebox.showerror("Parameter Error", topology_error)
                return
            command.extend(
                [
                    "--num-users",
                    str(num_users),
                    "--num-streams-per-user",
                    str(num_streams),
                ]
            )
            extra_summary_parts.extend(
                [
                    f"K={num_users}",
                    f"Ns={num_streams}",
                    f"Nrf={tx}(=Nt)",
                    f"Pdig={self._format_compact_number(power_constraint)}",
                    f"QAM={2 ** bits_per_symbol}QAM",
                ]
            )

        self._append_log(f"\n>>> 开始训练: {' '.join(command)}")
        self.m2_status_var.set(
            f"训练中：{task} | channel={channel_type} | Nt={tx} Nr={rx}"
        )

        self.m2_status_var.set(
            f"Running | {task} | channel={channel_type} | Nt={tx} Nr={rx} | "
            + " ".join(extra_summary_parts)
        )

        self.train_process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self.start_train_button.configure(state="disabled")
        self.stop_train_button.configure(state="normal")

        thread = threading.Thread(target=self._stream_training_logs, daemon=True)
        thread.start()

    def _stream_training_logs(self) -> None:
        process = self.train_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self._is_closing:
                return
            try:
                self.root.after(0, self._append_log, line.rstrip("\n"))
            except RuntimeError:
                return
        return_code = process.wait()
        if self._is_closing:
            return
        try:
            self.root.after(0, self._on_training_finished, return_code)
        except RuntimeError:
            return

    def _on_training_finished(self, return_code: int) -> None:
        self._append_log(f">>> 训练结束，返回码: {return_code}")
        if return_code == 0:
            self.m2_status_var.set("训练完成")
        else:
            self.m2_status_var.set(f"训练失败（返回码 {return_code}）")
        self.start_train_button.configure(state="normal")
        self.stop_train_button.configure(state="disabled")
        self.train_process = None

    def _stop_training(self) -> None:
        process = self.train_process
        if process is None or process.poll() is not None:
            self.m2_status_var.set("当前没有运行中的训练")
            self.stop_train_button.configure(state="disabled")
            self.start_train_button.configure(state="normal")
            return

        self._append_log(">>> 请求停止训练 ...")
        process.terminate()
        self.m2_status_var.set("正在停止训练 ...")
        self.root.after(2000, self._force_kill_training_if_needed)

    def _force_kill_training_if_needed(self) -> None:
        process = self.train_process
        if process is None or process.poll() is not None:
            return
        process.kill()
        self._append_log(">>> 训练进程已强制结束。")

    def _open_selected_result_dir(self) -> None:
        task = self.m2_task_var.get().strip()
        result_dir = TRAIN_TASK_TO_RESULT_DIR.get(task)
        if result_dir is None:
            messagebox.showerror("路径错误", f"未知训练任务：{task}")
            return
        result_dir.mkdir(parents=True, exist_ok=True)
        if not self._open_with_system_viewer(result_dir):
            # Fallback for directory opening if startfile fails.
            try:
                subprocess.Popen(["explorer", str(result_dir)])
                return
            except OSError:
                pass
            messagebox.showerror("打开失败", f"无法打开目录：\n{result_dir}")

    # ------------------------
    # Close
    # ------------------------
    def _on_close(self) -> None:
        self._is_closing = True
        process = self.train_process
        if process is not None and process.poll() is None:
            should_close = messagebox.askyesno("确认退出", "训练还在运行，是否停止训练并退出？")
            if not should_close:
                self._is_closing = False
                return
            process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LauncherGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
