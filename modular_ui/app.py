from __future__ import annotations

import os
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

if __package__:
    from .catalog import (
        ALL_OPTION,
        CATALOG_PATH,
        ExperimentRecord,
        discover_records,
        discover_train_channel_options,
        format_compact_number,
        load_catalog_records,
        load_legacy_records,
        normalize_channel_model,
        parse_positive_float,
        parse_positive_int,
        qam_from_bits,
    )
    from .tasks import (
        CSI_MODE_LABEL_TO_VALUE,
        CSI_MODE_OPTIONS,
        default_device,
        TASK_LABEL_TO_KEY,
        TASK_OPTIONS,
        RunnerConfig,
        build_command,
        command_to_text,
        task_spec,
        validate_runner_config,
    )
    from .widgets import BlockCard, ImagePreview, LogConsole, make_labeled_combo, make_labeled_entry
    from .workflow import WorkflowPage
else:
    from catalog import (
        ALL_OPTION,
        CATALOG_PATH,
        ExperimentRecord,
        discover_records,
        discover_train_channel_options,
        format_compact_number,
        load_catalog_records,
        load_legacy_records,
        normalize_channel_model,
        parse_positive_float,
        parse_positive_int,
        qam_from_bits,
    )
    from tasks import (
        CSI_MODE_LABEL_TO_VALUE,
        CSI_MODE_OPTIONS,
        default_device,
        TASK_LABEL_TO_KEY,
        TASK_OPTIONS,
        RunnerConfig,
        build_command,
        command_to_text,
        task_spec,
        validate_runner_config,
    )
    from widgets import BlockCard, ImagePreview, LogConsole, make_labeled_combo, make_labeled_entry
    from workflow import WorkflowPage


BG = "#0f1722"
PANEL = "#16212f"
SURFACE = "#1a2636"
CARD = "#111b2a"
CARD_ALT = "#172435"
TEXT = "#eef5ff"
MUTED = "#b9cbe0"
ACCENT = "#5bb6ff"
ACCENT_2 = "#f59e0b"


def _sorted_unique(values: list[str], *, all_option: bool = True) -> list[str]:
    cleaned = sorted({str(value) for value in values if str(value).strip()})
    if all_option:
        return [ALL_OPTION] + cleaned
    return cleaned


def _record_text(record: ExperimentRecord) -> str:
    lines = [
        f"File: {record.file}",
        f"Scope: {record.user_scope} | Family: {record.precoding_family}",
        f"Channel: {record.channel_model} | CSI: {record.channel_estimation}",
        f"Topology: K={record.num_users} Nt={record.tx_antennas} Nr={record.rx_antennas} "
        f"Nrf={record.num_rf_chains} Ns={record.num_streams_per_user}",
        f"Power: {record.digital_power_constraint} | QAM: {record.qam} | Coding Rate: {record.rate}",
    ]
    if record.note:
        lines.append(f"Note: {record.note}")
    return "\n".join(lines)


def _match_field(value: str, selected: str) -> bool:
    return selected == ALL_OPTION or value == selected


def _parse_optional_number(text: str, *, allow_auto: bool = True) -> float | None:
    token = str(text).strip()
    if not token:
        return None
    if allow_auto and token.lower() == "auto":
        return None
    try:
        return float(token)
    except ValueError:
        return None


class ModularLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Modular Sionna-Style Launcher")
        self.root.geometry("1400x920")
        self.root.minsize(1180, 780)
        self.root.configure(bg=BG)

        self.records = discover_records()
        self.channel_options = discover_train_channel_options()
        self.status_var = tk.StringVar(value="Ready")
        self._active_record: ExperimentRecord | None = self.records[0] if self.records else None
        self._active_task_key = "hybrid"

        self._build_style()
        self._build_layout()
        self._sync_status("Ready")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Card.TFrame", background=CARD)
        style.configure("AltCard.TFrame", background=CARD_ALT)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 20, "bold"))
        style.configure("Hint.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("Field.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("BlockTitle.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("BlockText.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", SURFACE)], foreground=[("selected", TEXT)])
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        root = ttk.Frame(self.root, style="App.TFrame", padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        ttk.Label(root, text="Modular Sionna-Style Launcher", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            root,
            text="Gallery, runner, and pipeline are split into independent blocks.",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        notebook = ttk.Notebook(root)
        notebook.grid(row=2, column=0, sticky="nsew", pady=(12, 0))

        self.workflow_page = WorkflowPage(
            notebook,
            channel_options=self.channel_options,
            on_status=self._sync_status,
            on_results_updated=self.refresh_results,
        )
        self.gallery_page = GalleryPage(
            notebook,
            records=self.records,
            on_select=self._on_record_selected,
            on_open_status=self._sync_status,
        )

        notebook.add(self.gallery_page, text="Gallery")
        notebook.add(self.workflow_page, text="Workflow")

        status_bar = ttk.Frame(root, style="Panel.TFrame", padding=(12, 8))
        status_bar.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )

    def _sync_status(self, text: str) -> None:
        self.status_var.set(text)

    def refresh_results(self) -> None:
        records = discover_records()
        self.records = records
        gallery_page = getattr(self, "gallery_page", None)
        if gallery_page is not None:
            gallery_page.refresh_records(records)
            gallery_page.show_latest_record()
        self._sync_status("Results refreshed")

    def _on_record_selected(self, record: ExperimentRecord | None) -> None:
        self._active_record = record
        workflow_page = getattr(self, "workflow_page", None)
        if workflow_page is not None:
            workflow_page.set_record(record)
        if record is not None:
            self._sync_status(f"Selected {record.file}")

    def _on_close(self) -> None:
        workflow_page = getattr(self, "workflow_page", None)
        if workflow_page is not None:
            workflow_page.stop_running()
        self.root.destroy()


class GalleryPage(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Misc,
        *,
        records: list[ExperimentRecord],
        on_select: Callable[[ExperimentRecord | None], None],
        on_open_status: Callable[[str], None],
    ) -> None:
        super().__init__(parent, style="App.TFrame")
        self.records = records
        self.on_select = on_select
        self.on_open_status = on_open_status
        self._active_record: ExperimentRecord | None = None
        self._image_preview: ImagePreview | None = None
        self._tree_record_map: dict[str, ExperimentRecord] = {}

        self.scope_var = tk.StringVar(value=ALL_OPTION)
        self.family_var = tk.StringVar(value=ALL_OPTION)
        self.metric_var = tk.StringVar(value=ALL_OPTION)
        self.channel_var = tk.StringVar(value=ALL_OPTION)
        self.csi_var = tk.StringVar(value=ALL_OPTION)
        self.qam_var = tk.StringVar(value=ALL_OPTION)
        self.rate_var = tk.StringVar(value=ALL_OPTION)

        self._build_layout()
        self._refresh_records(select_first=True)

    def refresh_records(self, records: list[ExperimentRecord]) -> None:
        self.records = records
        self.scope_combo.configure(values=self._options_for("user_scope"))
        self.family_combo.configure(values=self._options_for("precoding_family"))
        self.metric_combo.configure(values=self._options_for("metric"))
        self.channel_combo.configure(values=self._options_for("channel_model"))
        self.csi_combo.configure(values=self._options_for("channel_estimation"))
        self.qam_combo.configure(values=self._options_for("qam"))
        self.rate_combo.configure(values=self._options_for("rate"))
        self._refresh_records(select_first=True)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        filter_band = ttk.Frame(self, style="Panel.TFrame", padding=12)
        filter_band.grid(row=0, column=0, sticky="ew")
        filter_band.columnconfigure(1, weight=1)
        filter_band.columnconfigure(3, weight=1)
        filter_band.columnconfigure(5, weight=1)
        filter_band.columnconfigure(7, weight=1)

        meta_frame = ttk.Frame(filter_band, style="Panel.TFrame")
        meta_frame.grid(row=0, column=0, sticky="ew")
        meta_frame.columnconfigure(1, weight=1)
        meta_frame.columnconfigure(3, weight=1)
        meta_frame.columnconfigure(5, weight=1)
        meta_frame.columnconfigure(7, weight=1)

        top_frame = ttk.LabelFrame(meta_frame, text="Metadata", padding=10)
        top_frame.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(3, weight=1)
        top_frame.columnconfigure(5, weight=1)
        top_frame.columnconfigure(7, weight=1)

        self.scope_combo = make_labeled_combo(top_frame, 0, 0, "Scope", self.scope_var, self._options_for("user_scope"), width=16, on_change=self._on_filter_changed)
        self.family_combo = make_labeled_combo(top_frame, 0, 2, "Family", self.family_var, self._options_for("precoding_family"), width=18, on_change=self._on_filter_changed)
        self.metric_combo = make_labeled_combo(top_frame, 0, 4, "Metric", self.metric_var, self._options_for("metric"), width=12, on_change=self._on_filter_changed)
        self.channel_combo = make_labeled_combo(top_frame, 0, 6, "Channel", self.channel_var, self._options_for("channel_model"), width=12, on_change=self._on_filter_changed)

        topology_frame = ttk.LabelFrame(meta_frame, text="Topology", padding=10)
        topology_frame.grid(row=0, column=1, sticky="ew")
        topology_frame.columnconfigure(1, weight=1)
        topology_frame.columnconfigure(3, weight=1)
        topology_frame.columnconfigure(5, weight=1)
        topology_frame.columnconfigure(7, weight=1)

        self.csi_combo = make_labeled_combo(topology_frame, 0, 0, "CSI", self.csi_var, self._options_for("channel_estimation"), width=16, on_change=self._on_filter_changed)
        self.qam_combo = make_labeled_combo(topology_frame, 0, 2, "QAM", self.qam_var, self._options_for("qam"), width=12, on_change=self._on_filter_changed)
        self.rate_combo = make_labeled_combo(topology_frame, 0, 4, "Coding Rate", self.rate_var, self._options_for("rate"), width=12, on_change=self._on_filter_changed)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        list_frame = ttk.Frame(body, style="Panel.TFrame", padding=12)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)
        body.add(list_frame, weight=3)

        ttk.Label(list_frame, text="Records", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        tree_holder = ttk.Frame(list_frame, style="Panel.TFrame")
        tree_holder.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        tree_holder.columnconfigure(0, weight=1)
        tree_holder.rowconfigure(0, weight=1)

        columns = ("scope", "family", "metric", "channel", "csi", "nt", "nr", "users", "nrf", "ns", "qam", "rate")
        self.tree = ttk.Treeview(tree_holder, columns=columns, show="headings", height=18)
        headings = {
            "scope": "Scope",
            "family": "Family",
            "metric": "Metric",
            "channel": "Channel",
            "csi": "CSI",
            "nt": "Nt",
            "nr": "Nr",
            "users": "K",
            "nrf": "Nrf",
            "ns": "Ns",
            "qam": "QAM",
            "rate": "Code Rate",
        }
        widths = {"scope": 90, "family": 90, "metric": 70, "channel": 82, "csi": 110, "nt": 58, "nr": 58, "users": 58, "nrf": 58, "ns": 58, "qam": 82, "rate": 78}
        for key in columns:
            self.tree.heading(key, text=headings[key])
            self.tree.column(key, width=widths[key], anchor="w", stretch=True)

        tree_scroll_y = ttk.Scrollbar(tree_holder, orient="vertical", command=self.tree.yview)
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x = ttk.Scrollbar(tree_holder, orient="horizontal", command=self.tree.xview)
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._open_selected_file)

        preview_frame = ttk.Frame(body, style="Panel.TFrame", padding=12)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        body.add(preview_frame, weight=4)

        ttk.Label(preview_frame, text="Preview", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.preview = ImagePreview(preview_frame)
        self.preview.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.summary_card = BlockCard(preview_frame, "Selected Record", "No record selected", accent=ACCENT, wraplength=320)
        self.summary_card.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        button_row = ttk.Frame(preview_frame, style="Panel.TFrame")
        button_row.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        ttk.Button(button_row, text="Open File", command=self._open_selected_file).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(button_row, text="Open Folder", command=self._open_selected_folder).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _options_for(self, attr: str) -> list[str]:
        values = [getattr(record, attr) for record in self.records]
        return _sorted_unique(values)

    def _on_filter_changed(self, _event: object | None = None) -> None:
        self._refresh_records(select_first=True)

    def _matches_record(self, record: ExperimentRecord) -> bool:
        return (
            _match_field(record.user_scope, self.scope_var.get())
            and _match_field(record.precoding_family, self.family_var.get())
            and _match_field(record.metric, self.metric_var.get())
            and _match_field(record.channel_model, self.channel_var.get())
            and _match_field(record.channel_estimation, self.csi_var.get())
            and _match_field(record.qam, self.qam_var.get())
            and _match_field(record.rate, self.rate_var.get())
        )

    def _refresh_records(self, *, select_first: bool) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._tree_record_map.clear()

        filtered = [record for record in self.records if self._matches_record(record)]
        for index, record in enumerate(filtered):
            item_id = self.tree.insert(
                "",
                "end",
                values=(
                    record.user_scope,
                    record.precoding_family,
                    record.metric,
                    record.channel_model,
                    record.channel_estimation,
                    record.tx_antennas,
                    record.rx_antennas,
                    record.num_users,
                    record.num_rf_chains,
                    record.num_streams_per_user,
                    record.qam,
                    record.rate,
                ),
            )
            self._tree_record_map[item_id] = record

        if filtered and select_first:
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self.tree.focus(first_item)
            self.tree.see(first_item)
            self._show_record(self._tree_record_map[first_item])
            self.on_open_status(f"Gallery filtered to {len(filtered)} record(s)")
            return

        self.preview.clear("No matching record")
        self.summary_card.set_summary(f"{len(filtered)} record(s) match the current filters")
        self.on_select(None)

    def _on_tree_select(self, _event: object | None = None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        record = self._tree_record_map.get(selection[0])
        if record is not None:
            self._show_record(record)

    def _show_record(self, record: ExperimentRecord) -> None:
        self._active_record = record
        self.preview.show(record.path)
        self.summary_card.set_title(record.file)
        self.summary_card.set_summary(_record_text(record))
        self.on_select(record)

    def show_latest_record(self) -> None:
        if not self.tree.get_children():
            self.preview.clear("No record available")
            return
        first_item = self.tree.get_children()[0]
        self.tree.selection_set(first_item)
        self.tree.focus(first_item)
        self.tree.see(first_item)
        record = self._tree_record_map.get(first_item)
        if record is not None:
            self._show_record(record)

    def _selected_record(self) -> ExperimentRecord | None:
        selection = self.tree.selection()
        if not selection:
            return self._active_record
        return self._tree_record_map.get(selection[0], self._active_record)

    def _open_selected_file(self, _event: object | None = None) -> None:
        record = self._selected_record()
        if record is None or not record.path.exists():
            self.on_open_status("No file selected")
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(record.path))
                self.on_open_status(f"Opened {record.file}")
                return
        except OSError as exc:
            messagebox.showerror("Open file failed", str(exc))
        self.on_open_status(f"Unable to open {record.file}")

    def _open_selected_folder(self) -> None:
        record = self._selected_record()
        if record is None:
            self.on_open_status("No file selected")
            return
        folder = record.path.parent
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(folder))
                self.on_open_status(f"Opened {folder}")
                return
        except OSError as exc:
            messagebox.showerror("Open folder failed", str(exc))
        self.on_open_status(f"Unable to open {folder}")


class RunnerPage(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Misc,
        *,
        channel_options: list[str],
        on_task_change: Callable[[str], None],
        on_status: Callable[[str], None],
    ) -> None:
        super().__init__(parent, style="App.TFrame")
        self.channel_options = channel_options
        self.on_task_change = on_task_change
        self.on_status = on_status
        self._process: subprocess.Popen[str] | None = None
        self._running = False

        self.task_var = tk.StringVar(value=TASK_OPTIONS[0])
        self.channel_var = tk.StringVar(value=channel_options[0] if channel_options else "cdl-a")
        self.device_var = tk.StringVar(value=default_device())
        self.users_var = tk.StringVar(value="1")
        self.tx_var = tk.StringVar(value="16")
        self.rx_var = tk.StringVar(value="4")
        self.streams_var = tk.StringVar(value="4")
        self.nrf_var = tk.StringVar(value="4")
        self.power_var = tk.StringVar(value="auto")
        self.bits_var = tk.StringVar(value="6")
        self.csi_mode_var = tk.StringVar(value="Perfect CSI")
        self.csi_nmse_var = tk.StringVar(value="-20")
        self.pilot_length_var = tk.StringVar(value="16")
        self.pilot_snr_var = tk.StringVar(value="")
        self.fairness_var = tk.StringVar(value="0")
        self.snr_start_var = tk.StringVar(value="10")
        self.snr_stop_var = tk.StringVar(value="30")
        self.snr_step_var = tk.StringVar(value="5")
        self.num_channels_var = tk.StringVar(value="10")
        self.train_samples_var = tk.StringVar(value="128")
        self.train_repeats_var = tk.StringVar(value="2")
        self.seed_var = tk.StringVar(value="20260327")

        self._build_layout()
        self._update_visibility()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        config = ttk.Frame(self, style="Panel.TFrame", padding=12)
        config.grid(row=0, column=0, sticky="ew")
        config.columnconfigure(0, weight=1)
        config.columnconfigure(1, weight=1)

        common = ttk.LabelFrame(config, text="Common", padding=10)
        common.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        common.columnconfigure(1, weight=1)
        common.columnconfigure(3, weight=1)
        common.columnconfigure(5, weight=1)
        common.columnconfigure(7, weight=1)

        make_labeled_combo(common, 0, 0, "Task", self.task_var, TASK_OPTIONS, width=24, on_change=self._on_task_change)
        make_labeled_combo(common, 0, 2, "Channel", self.channel_var, self.channel_options, width=16, on_change=self._on_any_change)
        self.csi_mode_combo = make_labeled_combo(
            common,
            0,
            4,
            "CSI Mode",
            self.csi_mode_var,
            CSI_MODE_OPTIONS,
            width=16,
            on_change=self._on_any_change,
        )
        make_labeled_combo(common, 0, 6, "Device", self.device_var, ["cpu", "cuda"], width=12, on_change=self._on_any_change)
        make_labeled_combo(common, 1, 0, "Users", self.users_var, ["1", "2", "3", "4"], width=10, on_change=self._on_any_change)
        make_labeled_entry(common, 1, 2, "Nt", self.tx_var, width=10)
        make_labeled_entry(common, 1, 4, "Nr", self.rx_var, width=10)
        make_labeled_entry(common, 1, 6, "Ns / user", self.streams_var, width=10)
        self.nrf_label = ttk.Label(common, text="Nrf", style="Field.TLabel")
        self.nrf_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.nrf_entry = ttk.Entry(common, textvariable=self.nrf_var, width=10)
        self.nrf_entry.grid(row=2, column=1, sticky="w", padx=(8, 12), pady=(8, 0))
        make_labeled_entry(common, 2, 2, "Power", self.power_var, width=10)
        make_labeled_entry(common, 2, 4, "Bits", self.bits_var, width=10)

        self.hybrid_frame = ttk.LabelFrame(config, text="Hybrid settings", padding=10)
        self.hybrid_frame.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.hybrid_frame.columnconfigure(1, weight=1)
        self.hybrid_frame.columnconfigure(3, weight=1)
        self.hybrid_frame.columnconfigure(5, weight=1)
        self.hybrid_frame.columnconfigure(7, weight=1)
        make_labeled_entry(self.hybrid_frame, 0, 0, "CSI NMSE (dB)", self.csi_nmse_var, width=12)
        make_labeled_entry(self.hybrid_frame, 0, 2, "Pilot Len", self.pilot_length_var, width=12)
        make_labeled_entry(self.hybrid_frame, 0, 4, "Pilot SNR", self.pilot_snr_var, width=12)
        make_labeled_entry(self.hybrid_frame, 1, 0, "Fairness", self.fairness_var, width=12)

        self.digital_frame = ttk.LabelFrame(config, text="Digital sweep", padding=10)
        self.digital_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.digital_frame.columnconfigure(1, weight=1)
        self.digital_frame.columnconfigure(3, weight=1)
        self.digital_frame.columnconfigure(5, weight=1)
        self.digital_frame.columnconfigure(7, weight=1)
        make_labeled_entry(self.digital_frame, 0, 0, "SNR start", self.snr_start_var, width=12)
        make_labeled_entry(self.digital_frame, 0, 2, "SNR stop", self.snr_stop_var, width=12)
        make_labeled_entry(self.digital_frame, 0, 4, "SNR step", self.snr_step_var, width=12)
        make_labeled_entry(self.digital_frame, 0, 6, "Channels", self.num_channels_var, width=12)
        make_labeled_entry(self.digital_frame, 1, 0, "Samples", self.train_samples_var, width=12)
        make_labeled_entry(self.digital_frame, 1, 2, "Repeats", self.train_repeats_var, width=12)
        make_labeled_entry(self.digital_frame, 1, 4, "Seed", self.seed_var, width=12)

        buttons = ttk.Frame(self, style="Panel.TFrame", padding=(12, 0))
        buttons.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        buttons.rowconfigure(1, weight=1)
        buttons.columnconfigure(0, weight=1)

        actions = ttk.Frame(buttons, style="Panel.TFrame")
        actions.grid(row=0, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        self.run_button = ttk.Button(actions, text="Run", style="Accent.TButton", command=self.start_running)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop_running, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(actions, text="Open Results", command=self._open_results).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.status_card = BlockCard(buttons, "Runner", "Ready", accent=ACCENT_2, wraplength=400)
        self.status_card.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        self.log = LogConsole(buttons)
        self.log.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        buttons.rowconfigure(2, weight=1)

    def _task_key(self) -> str:
        return TASK_LABEL_TO_KEY[self.task_var.get()]

    def _task_kind(self) -> str:
        return task_spec(self._task_key()).kind

    def _on_task_change(self, _event: object | None = None) -> None:
        self._update_visibility()
        self.on_task_change(self._task_key())

    def _on_any_change(self, _event: object | None = None) -> None:
        self._update_visibility()

    def _update_visibility(self) -> None:
        is_hybrid = self._task_kind() == "hybrid"
        if is_hybrid:
            self.nrf_label.grid()
            self.nrf_entry.grid()
            self.csi_mode_combo._label_widget.grid()  # type: ignore[attr-defined]
            self.csi_mode_combo.grid()
            self.hybrid_frame.grid()
            self.digital_frame.grid_remove()
        else:
            self.nrf_label.grid_remove()
            self.nrf_entry.grid_remove()
            self.csi_mode_combo._label_widget.grid_remove()  # type: ignore[attr-defined]
            self.csi_mode_combo.grid_remove()
            self.hybrid_frame.grid_remove()
            self.digital_frame.grid()
        self.on_status(f"Task set to {self.task_var.get()}")
        self.status_card.set_title(task_spec(self._task_key()).label)
        self.status_card.set_summary(
            "Hybrid mode uses shared RF plus F_BB=[N1F1,...,NKFK]. Digital mode uses the same user-wise baseband chain with identity RF."
            if is_hybrid
            else "Digital mode uses identity RF and the same SVD/GMD/UCD user-wise baseband comparison."
        )

    def _config_from_form(self) -> RunnerConfig | None:
        task_key = self._task_key()
        channel = self.channel_var.get().strip()
        if not channel:
            messagebox.showerror("Invalid input", "Channel cannot be empty.")
            return None

        num_users = parse_positive_int(self.users_var.get())
        num_tx = parse_positive_int(self.tx_var.get())
        num_rx = parse_positive_int(self.rx_var.get())
        num_streams = parse_positive_int(self.streams_var.get())
        if None in {num_users, num_tx, num_rx, num_streams}:
            messagebox.showerror("Invalid input", "Users, Nt, Nr, and Ns / user must be positive integers.")
            return None

        num_rf = parse_positive_int(self.nrf_var.get())
        if num_rf is None:
            messagebox.showerror("Invalid input", "Nrf must be a positive integer.")
            return None

        bits = parse_positive_int(self.bits_var.get())
        if bits not in {2, 4, 6, 8}:
            messagebox.showerror("Invalid input", "Bits must be one of 2, 4, 6, or 8.")
            return None

        power = _parse_optional_number(self.power_var.get())
        if power is not None and power <= 0.0:
            messagebox.showerror("Invalid input", "Power must be positive or auto.")
            return None

        csi_mode = CSI_MODE_LABEL_TO_VALUE[self.csi_mode_var.get()]
        csi_nmse = _parse_optional_number(self.csi_nmse_var.get())
        pilot_length = parse_positive_int(self.pilot_length_var.get())
        pilot_snr = _parse_optional_number(self.pilot_snr_var.get())
        fairness = _parse_optional_number(self.fairness_var.get()) or 0.0

        snr_start = _parse_optional_number(self.snr_start_var.get())
        snr_stop = _parse_optional_number(self.snr_stop_var.get())
        snr_step = _parse_optional_number(self.snr_step_var.get())
        num_channels = parse_positive_int(self.num_channels_var.get())
        train_samples = parse_positive_int(self.train_samples_var.get())
        train_repeats = parse_positive_int(self.train_repeats_var.get())
        seed = parse_positive_int(self.seed_var.get())

        if task_key == "digital":
            if None in {snr_start, snr_stop, snr_step, num_channels, train_samples, train_repeats, seed}:
                messagebox.showerror("Invalid input", "Digital sweep fields must be positive numbers.")
                return None
        else:
            if csi_mode == "gaussian" and csi_nmse is None:
                messagebox.showerror("Invalid input", "Gaussian CSI mode requires an NMSE value.")
                return None

        config = RunnerConfig(
            task_key=task_key,
            channel_type=channel,
            num_users=num_users,
            num_tx_antennas=num_tx,
            num_rx_antennas=num_rx,
            num_rf_chains=num_rf,
            num_streams_per_user=num_streams,
            digital_power_constraint=power,
            bits_per_symbol=bits,
            csi_mode=csi_mode,
            csi_nmse_db=csi_nmse if csi_nmse is not None else -20.0,
            pilot_length=pilot_length or 16,
            pilot_snr_db=pilot_snr,
            user_fairness_penalty_weight=fairness,
            snr_start_db=snr_start if snr_start is not None else -10.0,
            snr_stop_db=snr_stop if snr_stop is not None else 40.0,
            snr_step_db=snr_step if snr_step is not None else 5.0,
            num_channels=num_channels or 2,
            train_num_samples=train_samples or 128,
            train_num_repeats=train_repeats or 1,
            seed=seed or 20260327,
            device=self.device_var.get().strip() or "cpu",
        )
        return config

    def start_running(self) -> None:
        if self._running:
            self.on_status("A job is already running.")
            return
        config = self._config_from_form()
        if config is None:
            return

        validation_error = validate_runner_config(config)
        if validation_error is not None:
            messagebox.showerror("Invalid topology", validation_error)
            return

        command = build_command(config)
        self.log.clear()
        self.log.append(f"$ {command_to_text(command)}")
        self.status_card.set_title("Running")
        self.status_card.set_summary(command_to_text(command))
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._running = True
        self.on_status(f"Starting {task_spec(config.task_key).label}")

        def on_line(line: str) -> None:
            self.after(0, lambda: self.log.append(line))

        def on_finish(return_code: int) -> None:
            self.after(0, lambda: self._finish_run(return_code, config))

        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            self._running = False
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            messagebox.showerror("Launch failed", str(exc))
            self.on_status("Launch failed")
            return

        def reader() -> None:
            assert self._process is not None
            assert self._process.stdout is not None
            for line in self._process.stdout:
                on_line(line.rstrip())
            return_code = self._process.wait()
            on_finish(return_code)

        threading.Thread(target=reader, daemon=True).start()

    def _finish_run(self, return_code: int, config: RunnerConfig) -> None:
        self._running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._process = None
        result = "finished successfully" if return_code == 0 else f"finished with code {return_code}"
        self.status_card.set_title("Runner")
        self.status_card.set_summary(result)
        self.on_status(f"{task_spec(config.task_key).label} {result}")

    def stop_running(self) -> None:
        if not self._running:
            self.on_status("No active job.")
            return
        if self._process is None:
            return
        try:
            self._process.terminate()
            self.on_status("Stopping current job...")
        except Exception as exc:
            messagebox.showerror("Stop failed", str(exc))

    def _open_results(self) -> None:
        folder = task_spec(self._task_key()).result_dir
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(folder))
                self.on_status(f"Opened {folder}")
                return
        except OSError as exc:
            messagebox.showerror("Open failed", str(exc))
        self.on_status(f"Unable to open {folder}")


def main() -> None:
    root = tk.Tk()
    ModularLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
