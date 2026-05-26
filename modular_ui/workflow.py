from __future__ import annotations

import os
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

if __package__:
    from .catalog import format_compact_number, parse_positive_float, parse_positive_int
    from .tasks import (
        CSI_MODE_LABEL_TO_VALUE,
        CSI_MODE_OPTIONS,
        TASK_LABEL_TO_KEY,
        TASK_OPTIONS,
        RunnerConfig,
        build_command,
        command_to_text,
        default_device,
        task_spec,
        validate_runner_config,
    )
    from .widgets import BlockCard, ImagePreview, LogConsole
else:
    from catalog import format_compact_number, parse_positive_float, parse_positive_int
    from tasks import (
        CSI_MODE_LABEL_TO_VALUE,
        CSI_MODE_OPTIONS,
        TASK_LABEL_TO_KEY,
        TASK_OPTIONS,
        RunnerConfig,
        build_command,
        command_to_text,
        default_device,
        task_spec,
        validate_runner_config,
    )
    from widgets import BlockCard, ImagePreview, LogConsole


BG = "#0f1722"
PANEL = "#16212f"
SURFACE = "#1a2636"
CARD = "#111b2a"
CARD_ALT = "#172435"
TEXT = "#eef5ff"
MUTED = "#b9cbe0"
ACCENT = "#5bb6ff"
ACCENT_2 = "#f59e0b"
ACCENT_3 = "#7dd3fc"


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str
    options: tuple[str, ...] = ()
    allow_auto: bool = False


@dataclass(frozen=True)
class NodeSpec:
    key: str
    title: str
    summary: str
    accent: str
    fields: tuple[FieldSpec, ...]


@dataclass
class NodeState:
    spec: NodeSpec
    x: float
    y: float
    width: int = 250
    height: int = 98
    values: dict[str, str] = field(default_factory=dict)


def _parse_float(text: object, *, allow_auto: bool = False) -> float | None:
    token = str(text).strip()
    if not token:
        return None
    if allow_auto and token.lower() == "auto":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _node_specs(channel_options: list[str]) -> dict[str, NodeSpec]:
    channel_values = tuple(channel_options or ["cdl-a"])
    return {
        "task": NodeSpec(
            key="task",
            title="Task",
            summary="Choose the execution mode.",
            accent=ACCENT,
            fields=(
                FieldSpec("task_key", "Task", "choice", TASK_OPTIONS),
            ),
        ),
        "topology": NodeSpec(
            key="topology",
            title="Topology",
            summary="System size, RF chains, and total-stream power.",
            accent=ACCENT_2,
            fields=(
                FieldSpec("channel_type", "Channel", "choice", channel_values),
                FieldSpec("num_users", "Users", "int"),
                FieldSpec("num_tx_antennas", "Nt", "int"),
                FieldSpec("num_rx_antennas", "Nr", "int"),
                FieldSpec("num_streams_per_user", "Ns / user", "int"),
                FieldSpec("num_rf_chains", "Nrf", "int"),
                FieldSpec("bits_per_symbol", "Bits", "int"),
                FieldSpec("digital_power_constraint", "Power", "float", allow_auto=True),
            ),
        ),
        "training": NodeSpec(
            key="training",
            title="CSI + Sweep",
            summary="CSI mode plus SNR sweep / Monte Carlo controls.",
            accent=ACCENT_3,
            fields=(
                FieldSpec("csi_mode", "CSI Mode", "choice", CSI_MODE_OPTIONS),
                FieldSpec("csi_nmse_db", "CSI NMSE (dB)", "float"),
                FieldSpec("pilot_length", "Pilot Len", "int"),
                FieldSpec("pilot_snr_db", "Pilot SNR", "float", allow_auto=True),
                FieldSpec("user_fairness_penalty_weight", "Fairness", "float"),
                FieldSpec("snr_start_db", "SNR Start", "float"),
                FieldSpec("snr_stop_db", "SNR Stop", "float"),
                FieldSpec("snr_step_db", "SNR Step", "float"),
                FieldSpec("num_channels", "Channels", "int"),
                FieldSpec("train_num_samples", "Samples", "int"),
                FieldSpec("train_num_repeats", "Repeats", "int"),
                FieldSpec("seed", "Seed", "int"),
            ),
        ),
        "runtime": NodeSpec(
            key="runtime",
            title="Runtime",
            summary="Execution device and output context.",
            accent=ACCENT,
            fields=(
                FieldSpec("device", "Device", "choice", ("cpu", "cuda")),
            ),
        ),
        "launch": NodeSpec(
            key="launch",
            title="Launch",
            summary="Run the compiled workflow.",
            accent=ACCENT_2,
            fields=(),
        ),
    }


class WorkflowPage(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Misc,
        *,
        channel_options: list[str],
        on_status: Callable[[str], None],
        on_results_updated: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, style="App.TFrame")
        self.on_status = on_status
        self.on_results_updated = on_results_updated
        self.specs = _node_specs(channel_options)
        self.nodes: dict[str, NodeState] = {}
        self._node_order = ["task", "topology", "training", "runtime", "launch"]
        self._selected_node_key = "task"
        self._drag_node_key: str | None = None
        self._drag_offset = (0.0, 0.0)
        self._process: subprocess.Popen[str] | None = None
        self._running = False
        self._field_vars: dict[tuple[str, str], tk.StringVar] = {}
        self._tab_by_node: dict[str, ttk.Frame] = {}
        self._node_by_tab: dict[ttk.Frame, str] = {}
        self._node_items: dict[str, list[int]] = {}
        self._connection_items: list[int] = []
        self._selected_record_text = "No record selected"

        self._defaults = RunnerConfig()
        self._build_nodes()
        self._build_layout()
        self._sync_canvas()
        self._select_node("task")

    def set_record(self, record: object | None) -> None:
        if record is None:
            self._selected_record_text = "No record selected"
        else:
            file_name = getattr(record, "file", None)
            self._selected_record_text = f"Record: {file_name}" if file_name else "Record selected"
        self._refresh_summary()

    def _build_nodes(self) -> None:
        defaults = self._defaults
        channel = self.specs["topology"].fields[0].options[0] if self.specs["topology"].fields[0].options else "cdl-a"
        self.nodes = {
            "task": NodeState(
                self.specs["task"],
                40,
                60,
                values={"task_key": TASK_OPTIONS[0]},
            ),
            "topology": NodeState(
                self.specs["topology"],
                350,
                60,
                values={
                    "channel_type": channel,
                    "num_users": str(defaults.num_users),
                    "num_tx_antennas": str(defaults.num_tx_antennas),
                    "num_rx_antennas": str(defaults.num_rx_antennas),
                    "num_streams_per_user": str(defaults.num_streams_per_user),
                    "num_rf_chains": str(defaults.num_rf_chains),
                    "bits_per_symbol": str(defaults.bits_per_symbol),
                    "digital_power_constraint": "auto",
                },
            ),
            "training": NodeState(
                self.specs["training"],
                660,
                60,
                values={
                    "csi_mode": CSI_MODE_OPTIONS[0],
                    "csi_nmse_db": str(defaults.csi_nmse_db),
                    "pilot_length": str(defaults.pilot_length),
                    "pilot_snr_db": "",
                    "user_fairness_penalty_weight": str(defaults.user_fairness_penalty_weight),
                    "snr_start_db": str(defaults.snr_start_db),
                    "snr_stop_db": str(defaults.snr_stop_db),
                    "snr_step_db": str(defaults.snr_step_db),
                    "num_channels": str(defaults.num_channels),
                    "train_num_samples": str(defaults.train_num_samples),
                    "train_num_repeats": str(defaults.train_num_repeats),
                    "seed": str(defaults.seed),
                },
            ),
            "runtime": NodeState(
                self.specs["runtime"],
                970,
                60,
                values={"device": default_device()},
            ),
            "launch": NodeState(
                self.specs["launch"],
                1280,
                60,
                values={},
            ),
        }

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        ttk.Label(header, text="Workflow Editor", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Drag blocks on the canvas, edit parameters on the right, then run the compiled configuration.",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        actions = ttk.Frame(header, style="App.TFrame")
        actions.grid(row=0, column=1, rowspan=2, sticky="e")
        self.run_button = ttk.Button(actions, text="Run Flow", style="Accent.TButton", command=self.start_running)
        self.run_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop_running, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=8)
        ttk.Button(actions, text="Reset Layout", command=self.reset_layout).grid(row=0, column=2, padx=(8, 0))

        body = ttk.PanedWindow(self, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        canvas_panel = ttk.Frame(body, style="Panel.TFrame", padding=12)
        inspector_panel = ttk.Frame(body, style="Panel.TFrame", padding=12)
        body.add(canvas_panel, weight=3)
        body.add(inspector_panel, weight=2)

        canvas_panel.columnconfigure(0, weight=1)
        canvas_panel.rowconfigure(1, weight=1)
        ttk.Label(canvas_panel, text="Canvas", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        canvas_frame = ttk.Frame(canvas_panel, style="Panel.TFrame")
        canvas_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#0b1522", highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", lambda _event: self._sync_canvas())

        inspector_panel.columnconfigure(0, weight=1)
        inspector_panel.rowconfigure(1, weight=1)
        self.selection_card = BlockCard(inspector_panel, "Selected Block", "Pick a block to edit it.", accent=ACCENT)
        self.selection_card.grid(row=0, column=0, sticky="ew")

        self.tabs = ttk.Notebook(inspector_panel)
        self.tabs.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        self._build_node_tabs()

        self.run_tab = ttk.Frame(self.tabs, style="App.TFrame", padding=12)
        self.tabs.add(self.run_tab, text="Run")
        self._build_run_tab()

        self.summary_card = BlockCard(inspector_panel, "Workflow Summary", "Ready", accent=ACCENT_2)
        self.summary_card.grid(row=2, column=0, sticky="ew", pady=(12, 0))

    def _build_node_tabs(self) -> None:
        for node_key in self._node_order:
            node = self.nodes[node_key]
            tab = ttk.Frame(self.tabs, style="App.TFrame", padding=12)
            self.tabs.add(tab, text=node.spec.title)
            self._tab_by_node[node_key] = tab
            self._node_by_tab[tab] = node_key
            tab.columnconfigure(1, weight=1)
            ttk.Label(tab, text=node.spec.summary, style="Hint.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
            self._build_fields(tab, node)

    def _build_fields(self, parent: ttk.Frame, node: NodeState) -> None:
        for row, field_spec in enumerate(node.spec.fields, start=1):
            ttk.Label(parent, text=field_spec.label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=(8, 0))
            var = tk.StringVar(value=node.values.get(field_spec.key, ""))
            self._field_vars[(node.spec.key, field_spec.key)] = var

            def _commit(*_args: object, _node=node, _field=field_spec, _var=var) -> None:
                _node.values[_field.key] = _var.get()
                self._redraw_node(_node.spec.key)
                self._refresh_summary()

            if field_spec.kind == "choice":
                widget = ttk.Combobox(parent, textvariable=var, values=list(field_spec.options), state="readonly")
            else:
                widget = ttk.Entry(parent, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", pady=(8, 0), padx=(8, 0))
            var.trace_add("write", _commit)

    def _build_run_tab(self) -> None:
        self.run_tab.columnconfigure(0, weight=1)
        self.run_tab.rowconfigure(2, weight=0)
        self.run_tab.rowconfigure(3, weight=2)

        action_row = ttk.Frame(self.run_tab, style="App.TFrame")
        action_row.grid(row=0, column=0, sticky="ew")
        action_row.columnconfigure(0, weight=1)
        action_row.columnconfigure(1, weight=1)
        action_row.columnconfigure(2, weight=1)
        ttk.Button(action_row, text="Open Results", command=self._open_results).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(action_row, text="Copy Command", command=self._copy_command).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(action_row, text="Validate", command=self._validate_current).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.run_card = BlockCard(self.run_tab, "Runner", "Ready", accent=ACCENT_2)
        self.run_card.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        self.preview = ImagePreview(self.run_tab, fallback_text="No result image yet")
        self.preview.grid(row=2, column=0, sticky="nsew", pady=(12, 0))

        self.log = LogConsole(self.run_tab)
        self.log.grid(row=3, column=0, sticky="nsew", pady=(12, 0))

    def reset_layout(self) -> None:
        positions = {
            "task": (40, 60),
            "topology": (350, 60),
            "training": (660, 60),
            "runtime": (970, 60),
            "launch": (1280, 60),
        }
        for key, (x, y) in positions.items():
            node = self.nodes[key]
            node.x = x
            node.y = y
        self._sync_canvas()
        self.on_status("Layout reset.")

    def _node_tag(self, node_key: str) -> str:
        return f"node::{node_key}"

    def _selected_node(self) -> NodeState:
        return self.nodes[self._selected_node_key]

    def _node_display_summary(self, node_key: str) -> str:
        node = self.nodes[node_key]
        if node_key == "task":
            label = node.values.get("task_key", TASK_OPTIONS[0])
            return label
        if node_key == "topology":
            return (
                f"{node.values.get('channel_type', 'cdl-a')} | "
                f"K={node.values.get('num_users', '?')} Nt={node.values.get('num_tx_antennas', '?')} "
                f"Nr={node.values.get('num_rx_antennas', '?')} Ns={node.values.get('num_streams_per_user', '?')} "
                f"Nrf={node.values.get('num_rf_chains', '?')}"
            )
        if node_key == "training":
            return (
                f"CSI {node.values.get('csi_mode', 'MMSE FullCov')} | "
                f"NMSE {node.values.get('csi_nmse_db', '?')} dB | "
                f"SNR {node.values.get('snr_start_db', '?')}..{node.values.get('snr_stop_db', '?')}"
            )
        if node_key == "runtime":
            return f"Device {node.values.get('device', 'cpu')}"
        return "Drag to place the run block."

    def _sync_canvas(self) -> None:
        self.canvas.delete("all")
        self._node_items.clear()
        for key in self._node_order:
            self._draw_node(key)
        self._draw_connections()
        self.canvas.configure(scrollregion=self.canvas.bbox("all") or (0, 0, 1800, 620))

    def _draw_node(self, node_key: str) -> None:
        node = self.nodes[node_key]
        tag = self._node_tag(node_key)
        outline = "#ffffff" if node_key == self._selected_node_key else "#253447"
        fill = CARD_ALT if node_key != "launch" else CARD
        rect = self.canvas.create_rectangle(
            node.x,
            node.y,
            node.x + node.width,
            node.y + node.height,
            fill=fill,
            outline=outline,
            width=2,
            tags=(tag,),
        )
        accent = self.canvas.create_rectangle(
            node.x,
            node.y,
            node.x + 6,
            node.y + node.height,
            fill=node.spec.accent,
            outline=node.spec.accent,
            tags=(tag,),
        )
        title = self.canvas.create_text(
            node.x + 16,
            node.y + 16,
            anchor="nw",
            text=node.spec.title,
            fill=TEXT,
            font=("Segoe UI", 11, "bold"),
            tags=(tag,),
        )
        summary = self.canvas.create_text(
            node.x + 16,
            node.y + 42,
            anchor="nw",
            text=self._node_display_summary(node_key),
            fill=MUTED,
            font=("Segoe UI", 9),
            width=node.width - 30,
            tags=(tag,),
        )
        self._node_items[node_key] = [rect, accent, title, summary]
        for item in self._node_items[node_key]:
            self.canvas.tag_bind(item, "<Button-1>", lambda event, key=node_key: self._on_node_press(event, key))
            self.canvas.tag_bind(item, "<B1-Motion>", lambda event, key=node_key: self._on_node_drag(event, key))
            self.canvas.tag_bind(item, "<ButtonRelease-1>", lambda event, key=node_key: self._on_node_release(event, key))

    def _draw_connections(self) -> None:
        for item in self._connection_items:
            self.canvas.delete(item)
        self._connection_items.clear()
        for left_key, right_key in zip(self._node_order, self._node_order[1:], strict=False):
            left = self.nodes[left_key]
            right = self.nodes[right_key]
            x1 = left.x + left.width
            y1 = left.y + left.height / 2
            x2 = right.x
            y2 = right.y + right.height / 2
            item = self.canvas.create_line(x1, y1, x2, y2, fill="#4b617a", width=3, arrow="last")
            self._connection_items.append(item)

    def _redraw_node(self, node_key: str) -> None:
        self._sync_canvas()
        if self._selected_node_key in self.nodes:
            self._select_node(self._selected_node_key, sync_tab=False)

    def _select_node(self, node_key: str, *, sync_tab: bool = True) -> None:
        if node_key not in self.nodes:
            return
        self._selected_node_key = node_key
        node = self.nodes[node_key]
        self.selection_card.set_title(node.spec.title)
        self.selection_card.set_summary(self._node_display_summary(node_key))
        self._refresh_summary()
        if sync_tab:
            tab = self._tab_by_node.get(node_key)
            if tab is not None:
                self.tabs.select(tab)
        self._sync_canvas()

    def _on_node_press(self, event: tk.Event, node_key: str) -> None:
        self._select_node(node_key)
        node = self.nodes[node_key]
        self._drag_node_key = node_key
        self._drag_offset = (self.canvas.canvasx(event.x) - node.x, self.canvas.canvasy(event.y) - node.y)

    def _on_node_drag(self, event: tk.Event, node_key: str) -> None:
        if self._drag_node_key != node_key:
            return
        node = self.nodes[node_key]
        new_x = self.canvas.canvasx(event.x) - self._drag_offset[0]
        new_y = self.canvas.canvasy(event.y) - self._drag_offset[1]
        node.x = new_x
        node.y = new_y
        self._sync_canvas()
        self._select_node(node_key, sync_tab=False)

    def _on_node_release(self, _event: tk.Event, node_key: str) -> None:
        if self._drag_node_key == node_key:
            self._drag_node_key = None

    def _on_canvas_click(self, event: tk.Event) -> None:
        item = self.canvas.find_closest(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if not item:
            return
        tags = self.canvas.gettags(item[0])
        for tag in tags:
            if tag.startswith("node::"):
                self._select_node(tag.removeprefix("node::"))
                break

    def _on_canvas_drag(self, _event: tk.Event) -> None:
        if self._drag_node_key is not None:
            return

    def _on_canvas_release(self, _event: tk.Event) -> None:
        self._drag_node_key = None

    def _refresh_summary(self) -> None:
        try:
            config = self._config_from_graph()
        except ValueError as exc:
            self.summary_card.set_title("Workflow Summary")
            self.summary_card.set_summary(str(exc))
            self.run_card.set_title("Runner")
            self.run_card.set_summary(str(exc))
            return

        task_label = task_spec(config.task_key).label
        top = f"{task_label} | {config.channel_type}"
        mid = f"K={config.num_users} Nt={config.num_tx_antennas} Nr={config.num_rx_antennas} Ns={config.num_streams_per_user}"
        tail = f"Device {config.device} | Power {config.digital_power_constraint if config.digital_power_constraint is not None else 'auto'}"
        self.summary_card.set_title("Workflow Summary")
        self.summary_card.set_summary("\n".join([self._selected_record_text, top, mid, tail]))
        self.run_card.set_title(task_label)
        self.run_card.set_summary(self._command_text(config))

    def _current_task_key(self) -> str:
        return TASK_LABEL_TO_KEY[self.nodes["task"].values.get("task_key", TASK_OPTIONS[0])]

    def _config_from_graph(self) -> RunnerConfig:
        task_label = self.nodes["task"].values.get("task_key", TASK_OPTIONS[0])
        task_key = TASK_LABEL_TO_KEY[task_label]

        topology = self.nodes["topology"].values
        training = self.nodes["training"].values
        runtime = self.nodes["runtime"].values

        num_users = parse_positive_int(topology.get("num_users"))
        num_tx = parse_positive_int(topology.get("num_tx_antennas"))
        num_rx = parse_positive_int(topology.get("num_rx_antennas"))
        num_streams = parse_positive_int(topology.get("num_streams_per_user"))
        num_rf = parse_positive_int(topology.get("num_rf_chains"))
        bits = parse_positive_int(topology.get("bits_per_symbol"))
        pilot_length = parse_positive_int(training.get("pilot_length"))
        num_channels = parse_positive_int(training.get("num_channels"))
        train_samples = parse_positive_int(training.get("train_num_samples"))
        train_repeats = parse_positive_int(training.get("train_num_repeats"))
        seed = parse_positive_int(training.get("seed"))

        if None in {num_users, num_tx, num_rx, num_streams, num_rf, bits, pilot_length, num_channels, train_samples, train_repeats, seed}:
            raise ValueError("Fix invalid integer fields before running.")

        csi_label = training.get("csi_mode", CSI_MODE_OPTIONS[0])
        if csi_label not in CSI_MODE_LABEL_TO_VALUE:
            raise ValueError("Choose a valid CSI mode.")

        power = _parse_float(topology.get("digital_power_constraint"), allow_auto=True)
        if str(topology.get("digital_power_constraint", "")).strip().lower() in {"", "auto"}:
            power = None
        elif power is None or power <= 0.0:
            raise ValueError("Power must be positive or auto.")

        csi_nmse = _parse_float(training.get("csi_nmse_db"))
        if csi_nmse is None:
            raise ValueError("CSI NMSE must be a number.")

        pilot_snr = _parse_float(training.get("pilot_snr_db"), allow_auto=True)
        fairness = _parse_float(training.get("user_fairness_penalty_weight"))
        snr_start = _parse_float(training.get("snr_start_db"))
        snr_stop = _parse_float(training.get("snr_stop_db"))
        snr_step = _parse_float(training.get("snr_step_db"))
        if None in {fairness, snr_start, snr_stop, snr_step}:
            raise ValueError("Fix invalid sweep fields before running.")

        config = RunnerConfig(
            task_key=task_key,
            channel_type=str(topology.get("channel_type", "cdl-a")).strip() or "cdl-a",
            num_users=num_users,
            num_tx_antennas=num_tx,
            num_rx_antennas=num_rx,
            num_rf_chains=num_rf,
            num_streams_per_user=num_streams,
            digital_power_constraint=power,
            bits_per_symbol=bits,
            csi_mode=CSI_MODE_LABEL_TO_VALUE[csi_label],
            csi_nmse_db=csi_nmse,
            pilot_length=pilot_length,
            pilot_snr_db=pilot_snr,
            user_fairness_penalty_weight=fairness,
            snr_start_db=snr_start,
            snr_stop_db=snr_stop,
            snr_step_db=snr_step,
            num_channels=num_channels,
            train_num_samples=train_samples,
            train_num_repeats=train_repeats,
            seed=seed,
            device=str(runtime.get("device", "cpu")).strip() or "cpu",
        )

        validation_error = validate_runner_config(config)
        if validation_error:
            raise ValueError(validation_error)
        return config

    def _command_text(self, config: RunnerConfig | None = None) -> str:
        if config is None:
            config = self._config_from_graph()
        return command_to_text(build_command(config))

    def _copy_command(self) -> None:
        try:
            command_text = self._command_text()
        except ValueError as exc:
            messagebox.showerror("Invalid workflow", str(exc))
            return
        self.clipboard_clear()
        self.clipboard_append(command_text)
        self.on_status("Command copied to clipboard.")

    def _validate_current(self) -> None:
        try:
            config = self._config_from_graph()
        except ValueError as exc:
            messagebox.showerror("Invalid workflow", str(exc))
            return
        messagebox.showinfo("Valid", self._command_text(config))

    def start_running(self) -> None:
        if self._running:
            self.on_status("A job is already running.")
            return
        try:
            config = self._config_from_graph()
        except ValueError as exc:
            messagebox.showerror("Invalid workflow", str(exc))
            return

        command = build_command(config)
        self.log.clear()
        self.log.append(f"$ {command_to_text(command)}")
        self.run_card.set_title("Running")
        self.run_card.set_summary(command_to_text(command))
        self.preview.clear("Training in progress...")
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
        self.run_card.set_title(task_spec(config.task_key).label)
        self.run_card.set_summary(result)
        if return_code == 0:
            latest_image = self._show_latest_result_image(config)
            if latest_image is not None:
                self.on_status(f"{task_spec(config.task_key).label} {result} | {latest_image.name}")
            else:
                self.on_status(f"{task_spec(config.task_key).label} {result} | no PNG found")
        else:
            self.preview.clear(f"Run failed ({return_code})")
            self.on_status(f"{task_spec(config.task_key).label} {result}")
        if self.on_results_updated is not None:
            self.on_results_updated()

    def _latest_result_image(self, result_dir: Path) -> Path | None:
        if not result_dir.exists():
            return None
        latest_path: Path | None = None
        latest_mtime = -1.0
        for image_path in result_dir.rglob("*.png"):
            if not image_path.is_file():
                continue
            try:
                mtime = image_path.stat().st_mtime
            except OSError:
                continue
            if latest_path is None or mtime > latest_mtime or (mtime == latest_mtime and image_path.name > latest_path.name):
                latest_path = image_path
                latest_mtime = mtime
        return latest_path

    def _show_latest_result_image(self, config: RunnerConfig) -> Path | None:
        latest_path = self._latest_result_image(task_spec(config.task_key).result_dir)
        if latest_path is None:
            self.preview.clear("No result image found")
            return None
        self.preview.show(latest_path)
        return latest_path

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
        folder = task_spec(self._current_task_key()).result_dir
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(folder))
                self.on_status(f"Opened {folder}")
                return
        except OSError as exc:
            messagebox.showerror("Open failed", str(exc))
        self.on_status(f"Unable to open {folder}")
