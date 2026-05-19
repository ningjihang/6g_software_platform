from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT_DIR / "photo"
RESULT_DIR = ROOT_DIR / "result"
CLASSICAL_RESULTS_DIR = ROOT_DIR / "classical" / "results"
FULL_DIGITAL_RESULTS_DIR = ROOT_DIR / "full_digital_mu" / "results"
CATALOG_PATH = PHOTO_DIR / "experiment_catalog.json"

ALL_OPTION = "ALL"

LEGACY_IMAGE_PATTERN = re.compile(
    r"^(?:(?P<metric>BER|SE)_)?(?:(?P<channel>[A-Za-z0-9-]+)_)?"
    r"Nt(?P<tx>\d+)_Nr(?P<rx>\d+)_(?P<qam>[A-Za-z0-9]+)_rate_(?P<rate>\d+(?:_\d+)?)\.png$",
    re.IGNORECASE,
)
FULL_DIGITAL_IMAGE_PATTERN = re.compile(
    r"^(?P<label>(?:compare_full_digital_(?:svd_vs_gmd_thp|svd_gmd_thp_ucd|svd_gmd_ucd|ucd)|全数字(?:_SVD_GMD(?:THP)?_)?UCD对比|全数字UCD对比))_(?P<channel>[A-Za-z0-9-]+)"
    r"_(?:k|K)(?P<users>\d+)_(?:nt|Nt)(?P<tx>\d+)_(?:nr|Nr)(?P<rx>\d+)(?:_(?:ns|Ns)(?P<ns>\d+))?.*?(?:(?:_m(?P<bits>\d+)_.*)|(?:(?P<qam>\d+)QAM_.*))\.png$",
    re.IGNORECASE,
)
HYBRID_COMPARE_IMAGE_PATTERN = re.compile(
    r"^(?P<label>.+?)_mode_(?P<csi>[a-z0-9_]+)_(?P<channel>[A-Za-z0-9-]+)"
    r"_k(?P<users>\d+)_nt(?P<tx>\d+)_nr(?P<rx>\d+)_nrf(?P<nrf>\d+)_ns(?P<ns>\d+)"
    r".*?_m(?P<bits>\d+)_.*\.png$",
    re.IGNORECASE,
)
SINGLE_USER_CHANNEL_BY_FILE = {
    "Nt256_Nr8_4QAM_rate_0_37.png": ("CDL-A", "UMA"),
    "Nt256_Nr8_16QAM_rate_0_48.png": ("CDL-A", "UMA"),
    "Nt256_Nr8_64QAM_rate_0_7.png": ("CDL-A", "UMA"),
}

CHANNEL_ORDER = {
    "CDL-A": 0,
    "CDL-B": 1,
    "CDL-C": 2,
    "CDL-D": 3,
    "CDL-E": 4,
    "UMA": 5,
    "N/A": 99,
}
FAMILY_ORDER = {"Hybrid": 0, "Digital": 1, "N/A": 99}
METRIC_ORDER = {"SE": 0, "BER": 1, "N/A": 99}
CSI_ORDER = {"Perfect CSI": 0, "Gaussian NMSE": 1, "MMSE FullCov": 2, "N/A": 99}
QAM_ORDER = {"QPSK": 0, "4QAM": 0, "16QAM": 1, "64QAM": 2, "256QAM": 3, "N/A": 99}
SCOPE_ORDER = {"Multi-User": 0, "Single-User": 1, "N/A": 99}


@dataclass(frozen=True)
class ExperimentRecord:
    file: str
    path: Path
    user_scope: str
    precoding_family: str
    metric: str
    channel_model: str
    channel_estimation: str
    tx_antennas: int
    rx_antennas: int
    num_users: str = "N/A"
    num_rf_chains: str = "N/A"
    num_streams_per_user: str = "N/A"
    digital_power_constraint: str = "N/A"
    qam: str = "N/A"
    rate: str = "N/A"
    note: str = ""

    @property
    def title(self) -> str:
        return (
            f"{self.user_scope} | {self.precoding_family} | {self.metric} | "
            f"{self.channel_model} | CSI={self.channel_estimation} | "
            f"Nt={self.tx_antennas} Nr={self.rx_antennas} | "
            f"K={self.num_users} Nrf={self.num_rf_chains} Ns={self.num_streams_per_user} "
            f"Pdig={self.digital_power_constraint} | {self.qam} | code rate {self.rate}"
        )


def normalize_channel_model(value: object) -> str:
    text = str(value or "").strip().upper().replace("_", "-")
    compact = re.sub(r"[^A-Z0-9]", "", text)
    if compact in {"CDLA", "SIONNACDLA"}:
        return "CDL-A"
    if compact in {"CDLB", "SIONNACDLB"}:
        return "CDL-B"
    if compact in {"CDLC", "SIONNACDLC"}:
        return "CDL-C"
    if compact in {"CDLD", "SIONNACDLD"}:
        return "CDL-D"
    if compact in {"CDLE", "SIONNACDLE"}:
        return "CDL-E"
    if compact in {"UMA", "SIONNAUMA"}:
        return "UMA"
    if compact.startswith("CDL") and len(compact) == 4:
        return f"CDL-{compact[-1]}"
    if compact.startswith("SIONNACDL") and len(compact) > len("SIONNACDL"):
        suffix = compact[-1]
        if suffix in {"A", "B", "C", "D", "E"}:
            return f"CDL-{suffix}"
    return text or "N/A"


def normalize_metric(value: object, default: str = "SE") -> str:
    text = str(value or "").strip().upper()
    if text in {"BER", "SE"}:
        return text
    return default


def normalize_channel_estimation(value: object) -> str:
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
        "": "N/A",
    }
    return aliases.get(compact, text or "N/A")


def normalize_qam(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "")
    if not text:
        return "N/A"
    if text == "QPSK":
        return "QPSK"
    match = re.match(r"^(?P<size>\d+)\s*QAM$", text)
    if match is not None:
        return f"{int(match.group('size'))}QAM"
    return text


def normalize_rate(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
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
    return text


def parse_positive_int(value: object) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def parse_positive_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def format_compact_number(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) <= 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def derive_power_text(num_users_text: str, num_streams_text: str, default: str = "N/A") -> str:
    num_users = parse_positive_int(num_users_text)
    num_streams = parse_positive_int(num_streams_text)
    if num_users is None or num_streams is None:
        return default
    return format_compact_number(num_users * num_streams)


def extract_power_text(text: str, default: str = "N/A") -> str:
    match = re.search(r"_pwr_(?P<value>[A-Za-z0-9mp]+)", text, re.IGNORECASE)
    if match is None:
        return default
    token = match.group("value").lower().replace("m", "-").replace("p", ".")
    parsed = parse_positive_float(token)
    return format_compact_number(parsed) if parsed is not None else default


def qam_from_bits(bits_text: object) -> str:
    bits = parse_positive_int(bits_text)
    if bits is None:
        return "N/A"
    return f"{2 ** bits}QAM"


def _resolve_existing_path(file_name: str) -> Path | None:
    for directory in (
        PHOTO_DIR,
        RESULT_DIR,
        CLASSICAL_RESULTS_DIR,
        FULL_DIGITAL_RESULTS_DIR,
    ):
        candidate = directory / file_name
        if candidate.exists():
            return candidate
    return None


def _precoding_family_for_path(path: Path, precoder_text: str, num_users: str, tx: int) -> str:
    full_digital = (
        "compare_full_digital" in path.name.lower()
        or "全数字ucd对比" in path.name.lower()
        or "全数字_svd_gmd" in path.name.lower()
        or "full_digital_mu" in str(path).lower()
    )
    if full_digital:
        return "Digital"
    precoder = precoder_text.strip().lower()
    if any(token in precoder for token in ("hybrid", "ao")):
        return "Hybrid"
    num_users_int = parse_positive_int(num_users) or 1
    if num_users_int > 1 or tx > 0:
        return "Hybrid"
    return "Digital"


def _record_sort_key(record: ExperimentRecord) -> tuple:
    return (
        SCOPE_ORDER.get(record.user_scope, 99),
        FAMILY_ORDER.get(record.precoding_family, 99),
        METRIC_ORDER.get(record.metric, 99),
        CHANNEL_ORDER.get(record.channel_model, 99),
        CSI_ORDER.get(record.channel_estimation, 99),
        record.tx_antennas,
        record.rx_antennas,
        QAM_ORDER.get(record.qam, 99),
        record.rate,
        record.file,
    )


def _record_from_catalog_item(item: dict) -> ExperimentRecord | None:
    file_name = str(item.get("file", "")).strip()
    if not file_name:
        return None
    path = _resolve_existing_path(file_name)
    if path is None:
        return None

    legacy_match = LEGACY_IMAGE_PATTERN.match(path.name)
    tx = parse_positive_int(item.get("tx_antennas"))
    rx = parse_positive_int(item.get("rx_antennas"))
    qam = normalize_qam(item.get("qam"))
    metric = normalize_metric(item.get("metric"), default="SE")
    csi = normalize_channel_estimation(item.get("channel_estimation"))
    num_users = str(item.get("num_users", "")).strip() or "N/A"
    num_rf = str(item.get("num_rf_chains", "")).strip() or "N/A"
    num_streams = str(item.get("num_streams_per_user", "")).strip() or "N/A"
    power = str(item.get("digital_power_constraint", "")).strip() or "N/A"
    precoder_text = str(item.get("precoder", "")).strip()

    if legacy_match is not None:
        if tx is None:
            tx = parse_positive_int(legacy_match.group("tx"))
        if rx is None:
            rx = parse_positive_int(legacy_match.group("rx"))
        if qam == "N/A":
            qam = normalize_qam(legacy_match.group("qam"))

    if tx is None or rx is None:
        return None

    if num_users == "N/A":
        num_users = "2" if tx > rx else "1"
    if num_rf == "N/A":
        num_rf = str(tx if "digital" in path.name.lower() else max(tx // 2, 1))
    if num_streams == "N/A":
        num_streams = str(min(rx, parse_positive_int(num_users) or 1))
    if power == "N/A":
        power = derive_power_text(num_users, num_streams)

    user_scope = str(item.get("user_scope", "")).strip()
    if not user_scope:
        user_scope = "Multi-User" if (parse_positive_int(num_users) or 1) > 1 else "Single-User"

    precoding_family = str(item.get("precoding_family", "")).strip()
    if not precoding_family:
        precoding_family = _precoding_family_for_path(
            path=path,
            precoder_text=precoder_text,
            num_users=num_users,
            tx=tx,
        )
    precoding_family = "Digital" if precoding_family.lower().startswith("digital") else "Hybrid"

    return ExperimentRecord(
        file=path.name,
        path=path,
        user_scope=user_scope or "N/A",
        precoding_family=precoding_family,
        metric=metric,
        channel_model=normalize_channel_model(item.get("channel_model")),
        channel_estimation=csi,
        tx_antennas=tx,
        rx_antennas=rx,
        num_users=num_users,
        num_rf_chains=num_rf,
        num_streams_per_user=num_streams,
        digital_power_constraint=power,
        qam=qam,
        rate=normalize_rate(item.get("coding_rate")),
        note=str(item.get("channel_note", "")).strip(),
    )


def _record_from_legacy_file(path: Path) -> ExperimentRecord | None:
    is_ber_plot = path.name.lower().endswith("_ber.png")

    match = FULL_DIGITAL_IMAGE_PATTERN.match(path.name)
    if match is not None:
        num_users = str(match.group("users"))
        num_streams = str(match.group("ns") or "N/A")
        power = derive_power_text(num_users, num_streams)
        qam = (
            qam_from_bits(match.group("bits"))
            if match.group("bits")
            else normalize_qam(f"{match.group('qam')}QAM")
        )
        return ExperimentRecord(
            file=path.name,
            path=path,
            user_scope="Multi-User",
            precoding_family="Digital",
            metric="BER" if is_ber_plot else "SE",
            channel_model=normalize_channel_model(match.group("channel")),
            channel_estimation="N/A",
            tx_antennas=int(match.group("tx")),
            rx_antennas=int(match.group("rx")),
            num_users=num_users,
            num_rf_chains=str(match.group("tx")),
            num_streams_per_user=num_streams,
            digital_power_constraint=power,
            qam=qam,
            rate="N/A",
        )

    match = HYBRID_COMPARE_IMAGE_PATTERN.match(path.name)
    if match is not None:
        num_users = str(match.group("users"))
        num_streams = str(match.group("ns"))
        power = extract_power_text(path.name, default=derive_power_text(num_users, num_streams))
        return ExperimentRecord(
            file=path.name,
            path=path,
            user_scope="Multi-User",
            precoding_family="Hybrid",
            metric="BER" if is_ber_plot else "SE",
            channel_model=normalize_channel_model(match.group("channel")),
            channel_estimation=normalize_channel_estimation(match.group("csi")),
            tx_antennas=int(match.group("tx")),
            rx_antennas=int(match.group("rx")),
            num_users=num_users,
            num_rf_chains=str(match.group("nrf")),
            num_streams_per_user=num_streams,
            digital_power_constraint=power,
            qam=qam_from_bits(match.group("bits")),
            rate="N/A",
        )

    match = LEGACY_IMAGE_PATTERN.match(path.name)
    if match is None:
        return None

    channel_model = normalize_channel_model(match.group("channel"))
    if channel_model == "N/A":
        fallback = SINGLE_USER_CHANNEL_BY_FILE.get(path.name)
        if fallback is not None:
            channel_model = fallback[0]

    return ExperimentRecord(
        file=path.name,
        path=path,
        user_scope="Single-User",
        precoding_family="Digital",
        metric=normalize_metric(match.group("metric"), default="BER"),
        channel_model=channel_model,
        channel_estimation="N/A",
        tx_antennas=int(match.group("tx")),
        rx_antennas=int(match.group("rx")),
        qam=normalize_qam(match.group("qam")),
        rate=normalize_rate(match.group("rate")),
    )


def load_catalog_records() -> list[ExperimentRecord]:
    if not CATALOG_PATH.exists():
        return []
    try:
        raw_data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(raw_data, dict):
        items = raw_data.get("experiments", [])
    elif isinstance(raw_data, list):
        items = raw_data
    else:
        return []

    records: list[ExperimentRecord] = []
    seen_files: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        record = _record_from_catalog_item(item)
        if record is None or record.file in seen_files:
            continue
        records.append(record)
        seen_files.add(record.file)
    return records


def load_legacy_records(skip_files: Iterable[str] | None = None) -> list[ExperimentRecord]:
    skip = {str(item) for item in (skip_files or [])}
    records: list[ExperimentRecord] = []
    seen: set[str] = set(skip)
    directories = (
        PHOTO_DIR,
        RESULT_DIR,
        CLASSICAL_RESULTS_DIR,
        FULL_DIGITAL_RESULTS_DIR,
    )
    for directory in directories:
        if not directory.exists():
            continue
        for image_path in sorted(directory.glob("*.png")):
            if image_path.name in seen:
                continue
            record = _record_from_legacy_file(image_path)
            if record is None:
                continue
            records.append(record)
            seen.add(record.file)
    return records


def discover_records() -> list[ExperimentRecord]:
    records = load_catalog_records()
    records.extend(load_legacy_records(skip_files=(record.file for record in records)))
    records.sort(key=_record_sort_key)
    return records


def discover_train_channel_options() -> list[str]:
    options = ["cdl-a", "cdl-b", "cdl-c", "cdl-d", "uma"]
    try:
        from cdl_a_channel import list_supported_sionna_channel_types

        discovered = [str(item).strip().lower() for item in list_supported_sionna_channel_types()]
    except Exception:
        discovered = []
    for item in discovered:
        if item and item not in options:
            options.append(item)
    return options
