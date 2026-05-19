from __future__ import annotations

import json
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageOps, ImageTk


ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name.lower() == "__pycache__":
    PHOTO_DIR = ROOT_DIR.parent
else:
    PHOTO_DIR = ROOT_DIR
CATALOG_PATH = PHOTO_DIR / "experiment_catalog.json"

LEGACY_IMAGE_PATTERN = re.compile(
    r"^(?:(?P<metric>BER|SE)_)?(?:(?P<channel>[A-Za-z0-9-]+)_)?Nt(?P<tx>\d+)_Nr(?P<rx>\d+)_(?P<qam>[A-Za-z0-9]+)_rate_(?P<rate>\d+(?:_\d+)?)\.png$",
    re.IGNORECASE,
)
RATE_PATTERN = re.compile(r"^\s*(?P<num>\d+)\s*[/_]\s*(?P<den>\d+)\s*$")

QAM_ORDER = {"QPSK": 0, "4QAM": 0, "16QAM": 1, "64QAM": 2, "256QAM": 3}
METRIC_ORDER = {"BER": 0, "SE": 1}
CHANNEL_MODEL_ORDER = {
    "CDL-A": 0,
    "CDL-B": 1,
    "CDL-C": 2,
    "CDL-D": 3,
    "CDL-E": 4,
    "UMA": 5,
    "RAYLEIGH": 6,
}
PRECODER_ORDER = {
    "baseline": 0,
    "svd": 1,
    "gmd": 2,
    "gmdthp": 3,
    "softao": 4,
}
ESTIMATION_ORDER = {
    "perfectcsi": 0,
    "gaussiannmse": 1,
    "mmsefullcov": 2,
}
DETECTION_ORDER = {
    "parallel": 0,
    "sic": 1,
    "thpaware": 2,
}
CODEC_ORDER = {
    "none": 0,
    "bicm": 1,
    "ldpc": 2,
    "polar": 3,
}
PRECODER_CANONICAL = {
    "baseline": "Baseline",
    "svd": "SVD",
    "gmd": "GMD",
    "gmdthp": "GMD+THP",
    "softao": "Soft-AO",
}
ESTIMATION_CANONICAL = {
    "perfectcsi": "Perfect CSI",
    "gaussiannmse": "Gaussian NMSE",
    "mmsefullcov": "MMSE FullCov",
}
DETECTION_CANONICAL = {
    "parallel": "Parallel",
    "sic": "SIC",
    "thpaware": "THP-aware",
}
CODEC_CANONICAL = {
    "none": "None",
    "bicm": "BICM",
    "ldpc": "LDPC",
    "polar": "Polar",
}

DEFAULT_METADATA = {
    "metric": "BER",
    "channel_model": "CDL-A",
    "precoder": "GMD+THP",
    "channel_estimation": "MMSE FullCov",
    "detection": "SIC",
    "codec": "BICM",
    "channel_note": "",
}

CHANNEL_MODEL_HINTS = {
    "UMA": "UMa: 3GPP Urban Macro scenario, larger-cell urban BS/UT geometry.",
    "CDL-A": "CDL-A: 小延迟扩展，LoS/NLoS 混合，常用于基线对比。",
    "CDL-B": "CDL-B: 比 CDL-A 更强的多径离散度，频率选择性更明显。",
    "CDL-C": "CDL-C: 延迟扩展更大，多径更丰富，对估计与检测更敏感。",
    "CDL-D": "CDL-D: 含明显 LoS 分量，阵列方向性和波束对齐影响更突出。",
    "CDL-E": "CDL-E: 高延迟扩展 + 角度扩散，链路条件更复杂。",
    "RAYLEIGH": "Rayleigh: 无显式 LoS，常作为随机衰落理论参考。",
}
BASE_OPTIONS = {
    "metric": ["BER", "SE"],
    "channel_model": ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E", "UMA", "RAYLEIGH"],
    "precoder": ["Baseline", "SVD", "GMD", "GMD+THP", "Soft-AO"],
    "channel_estimation": ["Perfect CSI", "Gaussian NMSE", "MMSE FullCov"],
    "detection": ["Parallel", "SIC", "THP-aware"],
    "codec": ["None", "BICM", "LDPC", "Polar"],
}
LEGACY_SINGLE_USER_CHANNEL_BY_FILE = {
    "Nt256_Nr8_4QAM_rate_0_37.png": ("CDL-A", "UMA"),
    "Nt256_Nr8_16QAM_rate_0_48.png": ("CDL-A", "UMA"),
    "Nt256_Nr8_64QAM_rate_0_7.png": ("CDL-A", "UMA"),
}


class PhotoSelectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Massive MIMO BER/SE Viewer")
        self.root.geometry("1460x920")
        self.root.minsize(1160, 760)
        self.root.configure(bg="#0f1722")

        self.entries = self._discover_entries()

        self.metric_var = tk.StringVar()
        self.channel_model_var = tk.StringVar()
        self.channel_hint_var = tk.StringVar()
        self.precoder_var = tk.StringVar()
        self.channel_estimation_var = tk.StringVar()
        self.detection_var = tk.StringVar()
        self.codec_var = tk.StringVar()

        self.tx_antennas_var = tk.StringVar()
        self.rx_antennas_var = tk.StringVar()
        self.qam_var = tk.StringVar()
        self.coding_rate_var = tk.StringVar()

        self.status_var = tk.StringVar()
        self.file_var = tk.StringVar()

        self.original_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self._resize_after_id: str | None = None

        self._build_style()
        self._build_layout()
        self._initialize_defaults()

    def _discover_entries(self) -> list[dict]:
        entries = self._load_catalog_entries()
        if not entries:
            entries = self._load_legacy_entries()

        if not entries:
            raise RuntimeError(
                "当前文件夹没有找到可识别图片。可使用 legacy 命名："
                "<CHANNEL>_Nt256_Nr8_QPSK_rate_1_2.png 或 Nt256_Nr8_QPSK_rate_1_2.png，或创建 experiment_catalog.json。"
            )

        entries.sort(key=self._entry_sort_key)
        return entries

    def _load_catalog_entries(self) -> list[dict]:
        if not CATALOG_PATH.exists():
            return []

        try:
            raw_data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"读取 {CATALOG_PATH.name} 失败: {exc}") from exc

        if isinstance(raw_data, dict):
            items = raw_data.get("experiments")
        elif isinstance(raw_data, list):
            items = raw_data
        else:
            raise RuntimeError(
                f"{CATALOG_PATH.name} 格式错误：顶层必须是 list 或 {{\"experiments\": [...]}}。"
            )

        if not isinstance(items, list):
            raise RuntimeError(
                f"{CATALOG_PATH.name} 格式错误：experiments 字段必须是 list。"
            )

        entries: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            file_name = str(item.get("file", "")).strip()
            if not file_name:
                continue

            image_path = PHOTO_DIR / file_name
            if image_path.suffix.lower() != ".png":
                continue

            legacy_match = LEGACY_IMAGE_PATTERN.match(image_path.name)

            tx_antennas = self._to_positive_int(item.get("tx_antennas"))
            rx_antennas = self._to_positive_int(item.get("rx_antennas"))
            qam = str(item.get("qam", "")).strip().upper()
            metric = self._normalize_metric(item.get("metric"))
            coding_rate = str(item.get("coding_rate", "")).strip()

            if legacy_match is not None:
                if not metric:
                    metric = self._normalize_metric(legacy_match.group("metric"))
                if tx_antennas is None:
                    tx_antennas = int(legacy_match.group("tx"))
                if rx_antennas is None:
                    rx_antennas = int(legacy_match.group("rx"))
                if not qam:
                    qam = legacy_match.group("qam").upper()
                if not coding_rate:
                    coding_rate = legacy_match.group("rate")

            if tx_antennas is None or rx_antennas is None:
                continue

            if not qam:
                qam = "QPSK"
            if not coding_rate:
                coding_rate = "1/2"

            parsed_rate = self._parse_rate(coding_rate)
            if parsed_rate is None:
                continue

            coding_rate_text, coding_rate_value = parsed_rate

            entry = {
                "metric": metric or DEFAULT_METADATA["metric"],
                "channel_model": self._normalize_channel_model(
                    item.get("channel_model", DEFAULT_METADATA["channel_model"])
                ),
                "precoder": self._canonical_precoder(
                    item.get("precoder", DEFAULT_METADATA["precoder"])
                ),
                "channel_estimation": self._canonical_channel_estimation(
                    item.get("channel_estimation", DEFAULT_METADATA["channel_estimation"])
                ),
                "detection": self._canonical_detection(
                    item.get("detection", DEFAULT_METADATA["detection"])
                ),
                "codec": self._canonical_codec(
                    item.get("codec", DEFAULT_METADATA["codec"])
                ),
                "channel_note": str(
                    item.get("channel_note", DEFAULT_METADATA["channel_note"])
                ).strip(),
                "tx_antennas": tx_antennas,
                "rx_antennas": rx_antennas,
                "qam": qam,
                "coding_rate": coding_rate_text,
                "coding_rate_value": coding_rate_value,
                "file": image_path.name,
                "path": image_path,
            }
            entry["title"] = self._build_entry_title(entry)
            entries.append(entry)

        return entries

    def _load_legacy_entries(self) -> list[dict]:
        entries: list[dict] = []
        for image_path in sorted(PHOTO_DIR.glob("*.png")):
            match = LEGACY_IMAGE_PATTERN.match(image_path.name)
            if match is None:
                continue

            qam = match.group("qam").upper()
            parsed_rate = self._parse_rate(match.group("rate"))
            if parsed_rate is None:
                continue
            coding_rate_text, coding_rate_value = parsed_rate
            metric = self._normalize_metric(match.group("metric"))
            channel_models = (
                [self._normalize_channel_model(match.group("channel"))]
                if match.group("channel")
                else list(LEGACY_SINGLE_USER_CHANNEL_BY_FILE.get(image_path.name, ("N/A",)))
            )
            for channel_model in channel_models:
                entry = {
                    "metric": metric or DEFAULT_METADATA["metric"],
                    "channel_model": channel_model,
                    "precoder": DEFAULT_METADATA["precoder"],
                    "channel_estimation": DEFAULT_METADATA["channel_estimation"],
                    "detection": DEFAULT_METADATA["detection"],
                    "codec": DEFAULT_METADATA["codec"],
                    "channel_note": DEFAULT_METADATA["channel_note"],
                    "tx_antennas": int(match.group("tx")),
                    "rx_antennas": int(match.group("rx")),
                    "qam": qam,
                    "coding_rate": coding_rate_text,
                    "coding_rate_value": coding_rate_value,
                    "file": image_path.name,
                    "path": image_path,
                }
                entry["title"] = self._build_entry_title(entry)
                entries.append(entry)
        return entries

    @staticmethod
    def _to_positive_int(value: object) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            number = int(text)
        except ValueError:
            return None
        return number if number > 0 else None

    @staticmethod
    def _normalize_metric(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip().upper()
        if not text:
            return ""
        if text in {"SE", "SPECTRAL_EFFICIENCY", "SPECTRAL-EFFICIENCY", "RATE"}:
            return "SE"
        return "BER"

    @staticmethod
    def _normalize_channel_model(value: object) -> str:
        text = str(value).strip().upper().replace("_", "-")
        if not text:
            return DEFAULT_METADATA["channel_model"]
        return text

    @classmethod
    def _canonical_precoder(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            return DEFAULT_METADATA["precoder"]
        return PRECODER_CANONICAL.get(cls._normalize_token(text), text)

    @classmethod
    def _canonical_channel_estimation(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            return DEFAULT_METADATA["channel_estimation"]
        return ESTIMATION_CANONICAL.get(cls._normalize_token(text), text)

    @classmethod
    def _canonical_detection(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            return DEFAULT_METADATA["detection"]
        return DETECTION_CANONICAL.get(cls._normalize_token(text), text)

    @classmethod
    def _canonical_codec(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            return DEFAULT_METADATA["codec"]
        return CODEC_CANONICAL.get(cls._normalize_token(text), text)

    @staticmethod
    def _parse_rate(rate_text: str) -> tuple[str, float] | None:
        text = str(rate_text or "").strip()
        decimal_match = re.match(r"^\s*0\s*[._]\s*(\d+)\s*$", text)
        if decimal_match is not None:
            normalized = f"0.{decimal_match.group(1)}".rstrip("0").rstrip(".")
            if normalized in {"", "0"}:
                return None
            return normalized, float(normalized)
        numeric_match = re.match(r"^\s*\d+(?:\.\d+)?\s*$", text)
        if numeric_match is not None:
            value = float(text)
            if value <= 0:
                return None
            normalized = str(value).rstrip("0").rstrip(".")
            return normalized, value
        match = RATE_PATTERN.match(rate_text)
        if match is None:
            return None
        num = int(match.group("num"))
        den = int(match.group("den"))
        if num <= 0 or den <= 0:
            return None
        return f"{num}/{den}", num / den

    def _entry_sort_key(self, entry: dict) -> tuple:
        return (
            METRIC_ORDER.get(entry["metric"].upper(), 99),
            CHANNEL_MODEL_ORDER.get(entry["channel_model"], 99),
            entry["channel_model"],
            entry["tx_antennas"],
            entry["rx_antennas"],
            QAM_ORDER.get(entry["qam"], 99),
            entry["qam"],
            entry["coding_rate_value"],
            PRECODER_ORDER.get(self._normalize_token(entry["precoder"]), 99),
            ESTIMATION_ORDER.get(self._normalize_token(entry["channel_estimation"]), 99),
            DETECTION_ORDER.get(self._normalize_token(entry["detection"]), 99),
            CODEC_ORDER.get(self._normalize_token(entry["codec"]), 99),
            entry["file"],
        )

    @staticmethod
    def _normalize_token(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _build_entry_title(self, entry: dict) -> str:
        return (
            f"{entry['metric']} | {entry['channel_model']} | "
            f"{entry['precoder']} | {entry['channel_estimation']} | "
            f"{entry['detection']} | {entry['codec']} | "
            f"Nt={entry['tx_antennas']} Nr={entry['rx_antennas']} | "
            f"{entry['qam']} | rate {entry['coding_rate']}"
        )

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background="#0f1722")
        style.configure("Panel.TFrame", background="#1a2636")
        style.configure("Section.TFrame", background="#111b2a")

        style.configure(
            "Title.TLabel",
            background="#0f1722",
            foreground="#f2f7ff",
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#0f1722",
            foreground="#c4d4e8",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "PanelTitle.TLabel",
            background="#1a2636",
            foreground="#f1f6ff",
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure(
            "SectionTitle.TLabel",
            background="#111b2a",
            foreground="#edf4ff",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "SectionInfo.TLabel",
            background="#111b2a",
            foreground="#d7e3f2",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "SectionHint.TLabel",
            background="#111b2a",
            foreground="#b8c9dc",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Footer.TLabel",
            background="#111b2a",
            foreground="#dbe9f8",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background="#111b2a",
            foreground="#ecf4ff",
            font=("Microsoft YaHei UI", 10),
        )

        style.configure(
            "App.TCombobox",
            fieldbackground="#edf4ff",
            background="#edf4ff",
            foreground="#0c1624",
            padding=6,
            arrowsize=16,
        )
        style.map(
            "App.TCombobox",
            fieldbackground=[("readonly", "#edf4ff")],
            background=[("readonly", "#edf4ff")],
            foreground=[("readonly", "#0c1624")],
            selectbackground=[("readonly", "#d7e6fb")],
            selectforeground=[("readonly", "#0c1624")],
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x")

        ttk.Label(header, text="Massive MIMO 实验可视化面板", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text=(
                "支持 BER / SE 指标、预编码/信道估计/检测/编解码模块筛选，"
                "并联动天线、调制和码率查看结果图。"
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        content = ttk.Frame(outer, style="App.TFrame")
        content.pack(fill="both", expand=True, pady=(18, 0))
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        control_panel = ttk.Frame(content, style="Panel.TFrame", padding=18)
        control_panel.grid(row=0, column=0, sticky="nsw")
        control_panel.columnconfigure(0, weight=1)

        ttk.Label(control_panel, text="控制面板", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )

        experiment_panel = ttk.Frame(control_panel, style="Section.TFrame", padding=12)
        experiment_panel.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        experiment_panel.columnconfigure(0, weight=1)

        ttk.Label(experiment_panel, text="实验维度", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(experiment_panel, text="性能指标", style="SectionInfo.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.metric_combo = ttk.Combobox(
            experiment_panel,
            textvariable=self.metric_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.metric_combo.grid(row=2, column=0, sticky="ew", pady=(6, 10))
        self.metric_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(experiment_panel, text="信道模型", style="SectionInfo.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        self.channel_combo = ttk.Combobox(
            experiment_panel,
            textvariable=self.channel_model_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.channel_combo.grid(row=4, column=0, sticky="ew", pady=(6, 8))
        self.channel_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(
            experiment_panel,
            textvariable=self.channel_hint_var,
            style="SectionHint.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=5, column=0, sticky="w")

        algorithm_panel = ttk.Frame(control_panel, style="Section.TFrame", padding=12)
        algorithm_panel.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        algorithm_panel.columnconfigure(0, weight=1)

        ttk.Label(algorithm_panel, text="算法模块", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(algorithm_panel, text="预编码算法", style="SectionInfo.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.precoder_combo = ttk.Combobox(
            algorithm_panel,
            textvariable=self.precoder_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.precoder_combo.grid(row=2, column=0, sticky="ew", pady=(6, 10))
        self.precoder_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(algorithm_panel, text="信道估计算法", style="SectionInfo.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        self.channel_estimation_combo = ttk.Combobox(
            algorithm_panel,
            textvariable=self.channel_estimation_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.channel_estimation_combo.grid(row=4, column=0, sticky="ew", pady=(6, 10))
        self.channel_estimation_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(algorithm_panel, text="检测算法", style="SectionInfo.TLabel").grid(
            row=5, column=0, sticky="w"
        )
        self.detection_combo = ttk.Combobox(
            algorithm_panel,
            textvariable=self.detection_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.detection_combo.grid(row=6, column=0, sticky="ew", pady=(6, 10))
        self.detection_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(algorithm_panel, text="信道编解码", style="SectionInfo.TLabel").grid(
            row=7, column=0, sticky="w"
        )
        self.codec_combo = ttk.Combobox(
            algorithm_panel,
            textvariable=self.codec_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.codec_combo.grid(row=8, column=0, sticky="ew", pady=(6, 8))
        self.codec_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(
            algorithm_panel,
            text="模块选项由图片文件名或 experiment_catalog.json 自动汇总。",
            style="SectionHint.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=9, column=0, sticky="w")

        antenna_panel = ttk.Frame(control_panel, style="Section.TFrame", padding=12)
        antenna_panel.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        antenna_panel.columnconfigure(0, weight=1)

        ttk.Label(antenna_panel, text="天线配置", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(antenna_panel, text="发射天线 Nt", style="SectionInfo.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.tx_combo = ttk.Combobox(
            antenna_panel,
            textvariable=self.tx_antennas_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.tx_combo.grid(row=2, column=0, sticky="ew", pady=(6, 10))
        self.tx_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(antenna_panel, text="接收天线 Nr", style="SectionInfo.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        self.rx_combo = ttk.Combobox(
            antenna_panel,
            textvariable=self.rx_antennas_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.rx_combo.grid(row=4, column=0, sticky="ew", pady=(6, 8))
        self.rx_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(
            antenna_panel,
            text="自动按当前算法与信道筛选可用 Nt/Nr 组合。",
            style="SectionHint.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=5, column=0, sticky="w")

        modulation_panel = ttk.Frame(control_panel, style="Section.TFrame", padding=12)
        modulation_panel.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        modulation_panel.columnconfigure(0, weight=1)

        ttk.Label(modulation_panel, text="调制与码率", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(modulation_panel, text="调制阶数 (QAM)", style="SectionInfo.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.qam_combo = ttk.Combobox(
            modulation_panel,
            textvariable=self.qam_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.qam_combo.grid(row=2, column=0, sticky="ew", pady=(6, 10))
        self.qam_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(modulation_panel, text="Coding Rate", style="SectionInfo.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        self.rate_combo = ttk.Combobox(
            modulation_panel,
            textvariable=self.coding_rate_var,
            state="readonly",
            width=20,
            style="App.TCombobox",
        )
        self.rate_combo.grid(row=4, column=0, sticky="ew", pady=(6, 8))
        self.rate_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        ttk.Label(
            modulation_panel,
            text="同一组算法与天线下，只显示可用的调制和码率。",
            style="SectionHint.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=5, column=0, sticky="w")

        status_panel = ttk.Frame(control_panel, style="Section.TFrame", padding=12)
        status_panel.grid(row=5, column=0, sticky="ew")
        status_panel.columnconfigure(0, weight=1)

        ttk.Label(status_panel, text="当前状态", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            status_panel,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 8))
        ttk.Label(
            status_panel,
            textvariable=self.file_var,
            style="Footer.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=2, column=0, sticky="w")
        action_row = ttk.Frame(status_panel, style="Section.TFrame")
        action_row.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        action_row.columnconfigure(0, weight=1)
        ttk.Button(
            action_row,
            text="Match Photo",
            command=self._show_selected_image,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            status_panel,
            text=(
                "Legacy 命名: <CHANNEL>_Nt256_Nr8_QPSK_rate_1_2.png 或 Nt256_Nr8_QPSK_rate_1_2.png\n"
                "可选 metric 前缀: BER_ / SE_\n"
                "更多模块请在 experiment_catalog.json 中配置。"
            ),
            style="Footer.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(12, 0))

        image_panel = ttk.Frame(content, style="Panel.TFrame", padding=14)
        image_panel.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(1, weight=1)

        ttk.Label(image_panel, text="结果图预览", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.image_container = tk.Frame(
            image_panel,
            bg="#0b1522",
            highlightbackground="#33455c",
            highlightthickness=1,
        )
        self.image_container.grid(row=1, column=0, sticky="nsew")
        self.image_container.bind("<Configure>", self._on_image_container_resize)

        self.image_label = tk.Label(
            self.image_container,
            bg="#0b1522",
            fg="#eef5ff",
            text="正在加载图片...",
            font=("Microsoft YaHei UI", 11),
        )
        self.image_label.pack(fill="both", expand=True, padx=12, pady=12)

    def _initialize_defaults(self) -> None:
        self._refresh_all_options()
        self._show_selected_image()

    def _entry_field_text(self, entry: dict, field: str) -> str:
        value = entry.get(field, "")
        if field in {"tx_antennas", "rx_antennas"}:
            return str(value)
        return str(value)

    def _matching_entries(self, ignore_field: str | None = None) -> list[dict]:
        filters = {
            "metric": self.metric_var.get(),
            "channel_model": self.channel_model_var.get(),
            "precoder": self.precoder_var.get(),
            "channel_estimation": self.channel_estimation_var.get(),
            "detection": self.detection_var.get(),
            "codec": self.codec_var.get(),
            "tx_antennas": self.tx_antennas_var.get(),
            "rx_antennas": self.rx_antennas_var.get(),
            "qam": self.qam_var.get(),
            "coding_rate": self.coding_rate_var.get(),
        }

        filtered = self.entries
        for field, selected in filters.items():
            if field == ignore_field or not selected:
                continue
            filtered = [
                entry
                for entry in filtered
                if self._entry_field_text(entry, field) == selected
            ]
        return filtered

    def _refresh_combo(
        self,
        field: str,
        combo: ttk.Combobox,
        variable: tk.StringVar,
        sort_key,
    ) -> None:
        values = set(BASE_OPTIONS.get(field, []))
        values.update(
            self._entry_field_text(entry, field)
            for entry in self._matching_entries(ignore_field=field)
        )
        ordered_values = sorted((value for value in values if value), key=sort_key)
        combo["values"] = ordered_values
        if variable.get() not in ordered_values:
            variable.set(ordered_values[0] if ordered_values else "")

    def _refresh_all_options(self) -> None:
        self._refresh_combo("metric", self.metric_combo, self.metric_var, self._sort_metric)
        self._refresh_combo(
            "channel_model",
            self.channel_combo,
            self.channel_model_var,
            self._sort_channel_model,
        )
        self._refresh_combo(
            "precoder",
            self.precoder_combo,
            self.precoder_var,
            self._sort_precoder,
        )
        self._refresh_combo(
            "channel_estimation",
            self.channel_estimation_combo,
            self.channel_estimation_var,
            self._sort_channel_estimation,
        )
        self._refresh_combo(
            "detection",
            self.detection_combo,
            self.detection_var,
            self._sort_detection,
        )
        self._refresh_combo("codec", self.codec_combo, self.codec_var, self._sort_codec)
        self._refresh_combo(
            "tx_antennas", self.tx_combo, self.tx_antennas_var, self._sort_numeric
        )
        self._refresh_combo(
            "rx_antennas", self.rx_combo, self.rx_antennas_var, self._sort_numeric
        )
        self._refresh_combo("qam", self.qam_combo, self.qam_var, self._sort_qam)
        self._refresh_combo(
            "coding_rate",
            self.rate_combo,
            self.coding_rate_var,
            self._sort_coding_rate,
        )
        self._update_channel_hint()

    @staticmethod
    def _sort_metric(value: str) -> tuple:
        return (METRIC_ORDER.get(value.upper(), 99), value)

    @staticmethod
    def _sort_channel_model(value: str) -> tuple:
        return (CHANNEL_MODEL_ORDER.get(value, 99), value)

    def _sort_precoder(self, value: str) -> tuple:
        return (PRECODER_ORDER.get(self._normalize_token(value), 99), value.lower())

    def _sort_channel_estimation(self, value: str) -> tuple:
        return (ESTIMATION_ORDER.get(self._normalize_token(value), 99), value.lower())

    def _sort_detection(self, value: str) -> tuple:
        return (DETECTION_ORDER.get(self._normalize_token(value), 99), value.lower())

    def _sort_codec(self, value: str) -> tuple:
        return (CODEC_ORDER.get(self._normalize_token(value), 99), value.lower())

    @staticmethod
    def _sort_numeric(value: str) -> tuple:
        try:
            return int(value), value
        except ValueError:
            return 10**9, value

    @staticmethod
    def _sort_qam(value: str) -> tuple:
        upper = value.upper()
        if upper in QAM_ORDER:
            return QAM_ORDER[upper], upper
        qam_match = re.match(r"^(?P<num>\d+)QAM$", upper)
        if qam_match is not None:
            return int(qam_match.group("num")), upper
        return 10**9, upper

    def _sort_coding_rate(self, value: str) -> tuple:
        parsed = self._parse_rate(value)
        if parsed is None:
            return 10**9, value
        rate_text, rate_value = parsed
        return rate_value, rate_text

    def _on_filter_changed(self, _event: object) -> None:
        self._refresh_all_options()
        self._show_selected_image()

    def _on_image_container_resize(self, _event: object) -> None:
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(80, self._render_current_image)

    def _update_channel_hint(self) -> None:
        selected_channel = self.channel_model_var.get().upper()
        hint = CHANNEL_MODEL_HINTS.get(
            selected_channel, "自定义信道：可在 experiment_catalog.json 的 channel_note 写说明。"
        )
        selected_entry = self._get_selected_entry()
        if selected_entry is not None and selected_entry.get("channel_note"):
            hint = selected_entry["channel_note"]
        self.channel_hint_var.set(hint)

    def _get_selected_entry(self) -> dict | None:
        matched = self._matching_entries()
        if not matched:
            return None
        matched.sort(key=self._entry_sort_key)
        return matched[0]

    @staticmethod
    def _prepare_display_image(image_obj: Image.Image) -> Image.Image:
        rgba_image = image_obj.convert("RGBA")
        alpha_min, _alpha_max = rgba_image.getchannel("A").getextrema()
        if alpha_min == 255:
            return rgba_image

        background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        background.alpha_composite(rgba_image)
        return background

    def _open_with_system_viewer(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        if not hasattr(os, "startfile"):
            messagebox.showerror(
                "无法弹出图片",
                f"当前系统不支持 startfile，请手动打开:\n{image_path}",
            )
            return False
        try:
            os.startfile(str(image_path))
            return True
        except OSError as exc:
            messagebox.showerror(
                "无法弹出图片",
                f"打开图片失败:\n{image_path.name}\n{exc}",
            )
            return False

    def _popup_current_image(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            messagebox.showwarning("没有图片", "当前筛选条件下没有可弹出的图片。")
            return
        if not self._open_with_system_viewer(entry["path"]):
            messagebox.showwarning("弹出失败", f"找不到图片文件：{entry['file']}")
            return
        self.status_var.set(f"已弹出当前图片：{entry['file']}")

    def _popup_matched_images(self) -> None:
        matched = self._matching_entries()
        if not matched:
            messagebox.showwarning("没有图片", "当前筛选条件下没有可弹出的图片。")
            return

        opened_count = 0
        for entry in matched:
            if self._open_with_system_viewer(entry["path"]):
                opened_count += 1

        if opened_count == 0:
            messagebox.showwarning("弹出失败", "匹配到的图片都无法打开。")
            return
        self.status_var.set(f"已弹出 {opened_count} 张匹配图片。")

    def _show_selected_image(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            self.original_image = None
            self.status_var.set("当前参数组合没有对应图片，请补充该配置的 BER/SE 图。")
            self.file_var.set(f"数据源：{PHOTO_DIR}")
            self.image_label.configure(image="", text="没有对应图片")
            return

        image_path = entry["path"]
        if not image_path.exists():
            self.original_image = None
            self.status_var.set(f"找不到图片：{entry['title']}")
            self.file_var.set(f"文件不存在：{image_path.name}")
            self.image_label.configure(image="", text="图片文件不存在")
            return

        try:
            with Image.open(image_path) as image_obj:
                self.original_image = self._prepare_display_image(image_obj)
        except OSError as exc:
            self.original_image = None
            self.status_var.set(f"图片无法打开：{entry['title']}")
            self.file_var.set(f"{image_path.name} ({exc})")
            self.image_label.configure(image="", text="图片读取失败")
            return

        self.status_var.set(
            f"{entry['metric']} | {entry['channel_model']} | "
            f"Precoder={entry['precoder']} | Est={entry['channel_estimation']} | "
            f"Det={entry['detection']} | Codec={entry['codec']} | "
            f"Nt={entry['tx_antennas']} Nr={entry['rx_antennas']} | "
            f"{entry['qam']} | rate {entry['coding_rate']}"
        )
        self.file_var.set(f"图像文件：{entry['file']}")
        self._render_current_image()

    def _render_current_image(self) -> None:
        if self.original_image is None:
            return

        width = max(self.image_container.winfo_width() - 24, 200)
        height = max(self.image_container.winfo_height() - 24, 200)

        resized = ImageOps.contain(self.original_image.copy(), (width, height))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.image_label.configure(image=self.tk_image, text="")


def main() -> None:
    try:
        root = tk.Tk()
        PhotoSelectorApp(root)
        root.mainloop()
    except RuntimeError as exc:
        messagebox.showerror("配置错误", str(exc))


if __name__ == "__main__":
    main()
