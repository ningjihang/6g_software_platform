from __future__ import annotations

import math
import random
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np
from scipy.linalg import svd

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

if __package__:
    from .channel_visualizer import ChannelAnalyzer
    from .components import (
        BS_NODE_COLOR,
        USER_NODE_COLORS,
        NodeComponent,
        draw_arrow_line,
        draw_grid_background,
    )
    from ..classical.multiuser_simulation_environment import MultiUserSimulationEnvironment
    from ..classical.sic_sample_average import build_multiuser_sample_average
else:
    from channel_visualizer import ChannelAnalyzer
    from components import (
        BS_NODE_COLOR,
        USER_NODE_COLORS,
        NodeComponent,
        draw_arrow_line,
        draw_grid_background,
    )
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR / "classical") not in sys.path:
        sys.path.insert(0, str(ROOT_DIR / "classical"))
    from multiuser_simulation_environment import MultiUserSimulationEnvironment
    from sic_sample_average import build_multiuser_sample_average


try:
    from full_digital_mu import FullyDigitalMuMimoBicmEnvironment
    FULL_DIGITAL_AVAILABLE = True
except ImportError as e:
    FULL_DIGITAL_AVAILABLE = False
    print(f"Warning: Cannot import full_digital_mu: {e}")


BG = "#0a0e14"
PANEL = "#10151c"
SURFACE = "#151c25"
CARD = "#1a2332"
TEXT = "#ffffff"
MUTED = "#c8d6e5"
ACCENT = "#00d4ff"
ACCENT_2 = "#ff9f43"
ACCENT_3 = "#54a0ff"
ACCENT_4 = "#00e676"
BORDER = "#1e2d3d"


PRECODING_COLORS = {
    "SVD": "#5bb6ff",
    "GMD": "#34d399",
    "UCD": "#f59e0b",
}


MOD_BITS = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}


@dataclass
class SimulationParams:
    num_users: int = 1
    num_tx_antennas: int = 16
    num_rx_antennas: int = 4
    num_rf_chains: int = 4
    num_streams_per_user: int = 4
    channel_type: str = "cdl-a"
    csi_mode: str = "perfect"
    snr_db: float = 10.0
    modulation: str = "64QAM"
    precoding_method: str = "SVD"


@dataclass
class UserState:
    id: int
    x: float
    y: float
    snr_db: float = 0.0
    gmi: float = 0.0
    ber: float = 0.0
    sinr_db: float = 0.0
    stream_snrs_db: list[float] = field(default_factory=list)
    singular_values: list[float] = field(default_factory=list)
    color: str = "#5bb6ff"


class SimulinkViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Physical Layer Simulink Viewer")
        self.geometry("1600x950")
        self.minsize(1200, 800)
        self.configure(bg=BG)

        self.params = SimulationParams()
        self.params.num_tx_antennas = 16
        self.channel_analyzer = ChannelAnalyzer(self.params)

        self.hybrid_env: MultiUserSimulationEnvironment | None = None
        self.fd_env = None
        self._init_environments()
        self._init_channels()
        self._init_users()

        self._selected_user: int | None = None
        self._running = False
        self._anim_time = 0.0
        self._anim_running = False

        self._build_style()
        self._build_layout()

    def _start_animation(self) -> None:
        if not self._anim_running:
            self._anim_running = True
            self._animate_loop()

    def _stop_animation(self) -> None:
        self._anim_running = False

    def _animate_loop(self) -> None:
        if not self._anim_running:
            return
        
        self._anim_time += 0.02
        width = self.canvas.winfo_width() or 1000
        height = self.canvas.winfo_height() or 800
        center_x, center_y = width // 2, height // 2
        
        for i, user in enumerate(self.user_states):
            base_angle = (2 * math.pi * i) / len(self.user_states) - math.pi / 2
            wobble = math.sin(self._anim_time * 1.5 + i * 0.8) * 15
            radius = 260 + wobble
            user.x = center_x + radius * math.cos(base_angle)
            user.y = center_y + radius * math.sin(base_angle)
        
        self._sync_canvas()
        self.after(33, self._animate_loop)

    def _init_environments(self) -> None:
        self.hybrid_env = MultiUserSimulationEnvironment(
            num_users=self.params.num_users,
            num_tx_antennas=self.params.num_tx_antennas,
            num_rx_antennas=self.params.num_rx_antennas,
            num_rf_chains=self.params.num_rf_chains,
            num_streams_per_user=self.params.num_streams_per_user,
            channel_type=self.params.channel_type,
            digital_power_constraint=float(self.params.num_users * self.params.num_streams_per_user),
        )
        if FULL_DIGITAL_AVAILABLE:
            self.fd_env = FullyDigitalMuMimoBicmEnvironment(
                num_users=self.params.num_users,
                num_tx_antennas=self.params.num_tx_antennas,
                num_rx_antennas=self.params.num_rx_antennas,
                num_streams_per_user=self.params.num_streams_per_user,
                channel_type=self.params.channel_type,
                digital_power_constraint=float(self.params.num_users * self.params.num_streams_per_user),
            )

    def _evaluate_method_at_snr(self, method: str, snr_db: float) -> dict[str, object]:
        if self.hybrid_env is None:
            raise RuntimeError("Hybrid environment is not initialized.")

        bits_per_symbol = MOD_BITS.get(self.params.modulation, 4)
        snr_linear = 10 ** (float(snr_db) / 10.0)
        snr_per_stream = snr_linear / self.hybrid_env.total_streams
        f_rf = self.hybrid_env.build_analog_precoder(self.channel_matrix)
        chain = self.hybrid_env.build_structured_digital_chain(
            user_channels=self.channel_matrix,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            strategy=method.lower(),
        )
        sample_average = build_multiuser_sample_average(
            env=self.hybrid_env,
            bits_per_symbol=bits_per_symbol,
            num_samples=64,
            num_repeats=1,
            base_seed=20260327,
            labeling="gray_standard",
        )

        if method.upper() == "UCD":
            evaluation = self.hybrid_env.evaluate_ucd_precoder_current_receiver_average_b_chain(
                user_channels=self.channel_matrix,
                f_rf=f_rf,
                f_bb=chain.f_bb,
                q_chains=chain.q_chains,
                r_chains=chain.r_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=bits_per_symbol,
                sample_average=sample_average,
                labeling="gray_standard",
            )
        else:
            evaluation = self.hybrid_env.evaluate_precoder_current_receiver_average_fixed_chain(
                user_channels=self.channel_matrix,
                f_rf=f_rf,
                f_bb=chain.f_bb,
                r_chains=chain.r_chains,
                q_chains=chain.q_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=bits_per_symbol,
                sample_average=sample_average,
                labeling="gray_standard",
            )

        singular_values = []
        stream_snrs_db = []
        user_gmi = []
        user_ber = []
        user_sinr_db = []
        for user_idx in range(self.params.num_users):
            n_k = chain.bd_digital_bases[user_idx]
            h_red = chain.effective_channels[user_idx] @ n_k
            s_eff = svd(h_red, compute_uv=False)
            s_eff = s_eff[: self.params.num_streams_per_user]
            singular_values.append(s_eff.tolist())

            rho_values = np.maximum(np.asarray(evaluation.user_rho[user_idx], dtype=float), 1e-12)
            stream_snrs_db.append((10.0 * np.log10(rho_values)).tolist())
            user_gmi.append(float(evaluation.user_rates[user_idx]))
            user_ber.append(float(evaluation.user_bit_error_rates[user_idx]))
            user_sinr_db.append(float(10.0 * np.log10(np.mean(rho_values))))

        snr_per_user_db = float(snr_db - 10 * math.log10(max(self.params.num_users, 1)))
        return {
            "snr_per_user_db": snr_per_user_db,
            "sinr_per_user_db": user_sinr_db,
            "stream_snrs_db": stream_snrs_db,
            "singular_values": singular_values,
            "user_gmi": user_gmi,
            "user_ber": user_ber,
            "sum_rate": float(evaluation.sum_rate),
        }

    def _evaluate_current_chain(self) -> dict[str, object]:
        return self._evaluate_method_at_snr(self.params.precoding_method, self.params.snr_db)

    def _init_channels(self) -> None:
        if self.hybrid_env is not None:
            self.channel_matrix = self.hybrid_env.generate_user_channels()
        else:
            self.channel_matrix = np.random.randn(
                self.params.num_users,
                self.params.num_rx_antennas,
                self.params.num_tx_antennas
            ) + 1j * np.random.randn(
                self.params.num_users,
                self.params.num_rx_antennas,
                self.params.num_tx_antennas
            )

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("Field.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _init_users(self) -> None:
        self.user_states = []
        metrics = self._evaluate_current_chain()

        for i in range(self.params.num_users):
            angle = (2 * math.pi * i) / self.params.num_users - math.pi / 2
            x = 800 + 280 * math.cos(angle)
            y = 500 + 280 * math.sin(angle)
            self.user_states.append(UserState(
                id=i + 1,
                x=x,
                y=y,
                snr_db=float(metrics["snr_per_user_db"]),
                sinr_db=float(metrics["sinr_per_user_db"][i]),
                stream_snrs_db=list(metrics["stream_snrs_db"][i]),
                singular_values=list(metrics["singular_values"][i]),
                gmi=float(metrics["user_gmi"][i]),
                ber=float(metrics["user_ber"][i]),
                color=USER_NODE_COLORS[i % len(USER_NODE_COLORS)]
            ))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Physical Layer Simulink Viewer", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="SVD/GMD Precoding Visualization", style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))

        control_frame = ttk.Frame(header, style="App.TFrame")
        control_frame.grid(row=0, column=2, rowspan=2, sticky="e")
        self.run_button = ttk.Button(control_frame, text="Start Simulation", style="Accent.TButton", command=self._toggle_simulation)
        self.run_button.pack(side="right", padx=(8, 0))
        ttk.Button(control_frame, text="Reset", command=self._reset_simulation).pack(side="right", padx=(8, 0))
        ttk.Button(control_frame, text="Generate Comparison Plot", command=self._generate_comparison_plot).pack(side="right", padx=(8, 0))

        self.left_panel = ttk.Frame(root, style="Panel.TFrame", width=340)
        self.left_panel.grid(row=1, column=0, sticky="ns", padx=(0, 12))
        self.left_panel.pack_propagate(False)
        
        self._build_params_panel(self.left_panel)
        self._build_user_panel(self.left_panel)

        canvas_frame = ttk.Frame(root, style="Surface.TFrame")
        canvas_frame.grid(row=1, column=1, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#0b1522", highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.right_panel = ttk.Frame(root, style="Panel.TFrame", width=360)
        self.right_panel.grid(row=1, column=2, sticky="ns", padx=(12, 0))
        self.right_panel.pack_propagate(False)
        
        self._build_info_panel(self.right_panel)

        self._sync_canvas()

    def _build_params_panel(self, parent: ttk.Frame) -> None:
        params_frame = ttk.LabelFrame(parent, text="System Parameters", padding=10)
        params_frame.pack(fill="x", pady=(0, 12))

        row = 0
        ttk.Label(params_frame, text="Users", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.users_combo = ttk.Combobox(params_frame, values=["1", "2", "3", "4", "5", "6", "7", "8"], state="readonly", width=8)
        self.users_combo.set(str(self.params.num_users))
        self.users_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Tx", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.nt_combo = ttk.Combobox(params_frame, values=["4", "8", "16", "32", "64"], state="readonly", width=8)
        self.nt_combo.set(str(self.params.num_tx_antennas))
        self.nt_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Rx", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.nr_combo = ttk.Combobox(params_frame, values=["2", "4", "8", "16"], state="readonly", width=8)
        self.nr_combo.set(str(self.params.num_rx_antennas))
        self.nr_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Nrf", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.nrf_combo = ttk.Combobox(params_frame, values=["4", "8", "16", "32", "64"], state="readonly", width=8)
        idx = ["4", "8", "16", "32", "64"].index(str(self.params.num_rf_chains))
        self.nrf_combo.current(idx)
        self.nrf_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Ns/User", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.streams_combo = ttk.Combobox(params_frame, values=["1", "2", "4", "8"], state="readonly", width=8)
        idx = ["1", "2", "4", "8"].index(str(self.params.num_streams_per_user))
        self.streams_combo.current(idx)
        self.streams_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Precoding", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        PRECODING_METHODS = ["SVD", "GMD", "UCD"]
        self.precoding_combo = ttk.Combobox(params_frame, values=PRECODING_METHODS, state="readonly", width=8)
        precoding_idx = PRECODING_METHODS.index(self.params.precoding_method) if self.params.precoding_method in PRECODING_METHODS else 0
        self.precoding_combo.current(precoding_idx)
        self.precoding_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Modulation", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        modulation_values = ["QPSK", "16QAM", "64QAM", "256QAM"]
        self.modulation_combo = ttk.Combobox(params_frame, values=modulation_values, state="readonly", width=10)
        modulation_idx = modulation_values.index(self.params.modulation) if self.params.modulation in modulation_values else 0
        self.modulation_combo.current(modulation_idx)
        self.modulation_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Total SNR (dB)", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.snr_slider = ttk.Scale(params_frame, from_=0, to=40, orient="horizontal", length=150)
        self.snr_slider.set(self.params.snr_db)
        self.snr_slider.grid(row=row, column=1, sticky="w", padx=(8, 0))
        
        self.snr_value_label = ttk.Label(params_frame, text=f"{self.params.snr_db:.1f} dB", foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self.snr_value_label.grid(row=row, column=2, sticky="w", padx=(4, 0))
        self.snr_slider.bind("<ButtonRelease-1>", lambda e: self._on_snr_slider_release())

        apply_btn = ttk.Button(params_frame, text="Apply", command=self._apply_params)
        apply_btn.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def _build_user_panel(self, parent: ttk.Frame) -> None:
        user_frame = ttk.LabelFrame(parent, text="User Metrics", padding=12)
        user_frame.pack(fill="both", expand=True)
        
        self.user_scrollbar = tk.Scrollbar(user_frame, orient="vertical", bg="#1e2d3d", activebackground="#2d4a6a", troughcolor="#1a2332", borderwidth=1, relief="flat", width=16)
        self.user_scrollbar.pack(side="right", fill="y")
        
        self.user_canvas = tk.Canvas(user_frame, bg="#1a2332", highlightthickness=0, bd=0, yscrollcommand=self.user_scrollbar.set)
        self.user_canvas.pack(side="left", fill="both", expand=True)
        self.user_scrollbar.config(command=self.user_canvas.yview)
        
        self.user_scrollable = tk.Frame(self.user_canvas, bg="#1a2332")
        self.user_canvas.create_window((0, 0), window=self.user_scrollable, anchor="nw")
        
        def _on_user_scrollable_configure(event):
            self.user_canvas.configure(scrollregion=self.user_canvas.bbox("all"))
        
        self.user_scrollable.bind("<Configure>", _on_user_scrollable_configure)
        
        self._build_all_user_rows()
        
        self._bind_mousewheel_to_widget_and_children(self.user_canvas, self.user_scrollable)
        
        self.after(100, lambda: self.user_canvas.configure(scrollregion=self.user_canvas.bbox("all")))
    
    def _bind_mousewheel_to_widget_and_children(self, canvas, scrollable):
        def _mousewheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), "units")
            return "break"
        
        def _bind_recursive(widget):
            widget.bind("<MouseWheel>", _mousewheel, add=True)
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"), add=True)
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"), add=True)
            for child in widget.winfo_children():
                _bind_recursive(child)
        
        _bind_recursive(scrollable)
        canvas.bind("<MouseWheel>", _mousewheel, add=True)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"), add=True)
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"), add=True)

    def _build_all_user_rows(self) -> None:
        for widget in self.user_scrollable.winfo_children():
            widget.destroy()
        
        self.user_rows = []
        
        for user in self.user_states:
            user_frame = tk.Frame(self.user_scrollable, bg="#223044", highlightbackground="#2d4a6a", highlightthickness=2, padx=10, pady=10)
            user_frame.pack(fill="x", pady=(0, 8))
            
            color_box = tk.Label(user_frame, bg=user.color, width=2, height=1)
            color_box.pack(side="left", padx=(0, 10))
            
            info_frame = tk.Frame(user_frame, bg="#223044")
            info_frame.pack(side="left", fill="both", expand=True)
            
            tk.Label(info_frame, text=f"User {user.id}", fg="#ffffff", bg="#223044", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            
            rate_label = tk.Label(info_frame, text=f"Rate: {user.gmi:.2f} bits/symbol", fg="#00e676", bg="#223044", font=("Segoe UI", 14, "bold"))
            rate_label.pack(anchor="w", pady=(6, 4))
            
            tk.Label(info_frame, text=f"SNR: {user.snr_db:.1f} dB  |  SINR: {user.sinr_db:.1f} dB", fg="#7dd3fc", bg="#223044", font=("Segoe UI", 9)).pack(anchor="w")
            
            stream_text = "Streams: " + ", ".join([f"{s:.1f} dB" for s in user.stream_snrs_db])
            tk.Label(info_frame, text=stream_text, fg="#b9cbe0", bg="#223044", font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))
            
            self.user_rows.append((user_frame, rate_label))


    def _build_decomp_panel(self, parent: ttk.Frame) -> None:
        decomp_frame = ttk.LabelFrame(parent, text=f"{self.params.precoding_method} Decomposition", padding=10)
        decomp_frame.pack(fill="both", expand=True)
        
        self.decomp_scrollbar = tk.Scrollbar(decomp_frame, orient="vertical", bg="#1e2d3d", activebackground="#2d4a6a", troughcolor="#1a2332", borderwidth=1, relief="flat", width=16)
        self.decomp_scrollbar.pack(side="right", fill="y")
        
        self.decomp_canvas = tk.Canvas(decomp_frame, bg=PANEL, highlightthickness=0, bd=0, yscrollcommand=self.decomp_scrollbar.set)
        self.decomp_canvas.pack(side="left", fill="both", expand=True)
        self.decomp_scrollbar.config(command=self.decomp_canvas.yview)
        
        self.decomp_scrollable = tk.Frame(self.decomp_canvas, bg="#1a2332")
        self.decomp_canvas.create_window((0, 0), window=self.decomp_scrollable, anchor="nw")
        
        self.decomp_scrollable.bind(
            "<Configure>",
            lambda e: self.decomp_canvas.configure(scrollregion=self.decomp_canvas.bbox("all"))
        )
        
        self._build_all_decomp_rows()
        
        self._bind_mousewheel_to_widget_and_children(self.decomp_canvas, self.decomp_scrollable)
        
        self.after(100, lambda: self.decomp_canvas.configure(scrollregion=self.decomp_canvas.bbox("all")))

    def _build_all_decomp_rows(self) -> None:
        for widget in self.decomp_scrollable.winfo_children():
            widget.destroy()
        
        for user in self.user_states:
            user_frame = tk.Frame(self.decomp_scrollable, bg="#151c25", highlightbackground="#1e2d3d", highlightthickness=1, padx=8, pady=8)
            user_frame.pack(fill="x", pady=(0, 6))
            
            color_box = tk.Label(user_frame, bg=user.color, width=2, height=1)
            color_box.pack(side="left", padx=(0, 8))
            
            info_frame = tk.Frame(user_frame, bg="#151c25")
            info_frame.pack(side="left", fill="both", expand=True)
            
            tk.Label(info_frame, text=f"User {user.id}", fg="#ffffff", bg="#151c25", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            
            sv_frame = tk.Frame(info_frame, bg="#151c25")
            sv_frame.pack(fill="x", pady=(4, 0))
            
            if self.params.precoding_method == "SVD":
                sv_text = "Singular Values: " + ", ".join([f"{sv:.3f}" for sv in user.singular_values])
                sv_color = "#00d4ff"
            else:
                sigma_bar = np.prod(user.singular_values) ** (1.0 / len(user.singular_values)) if user.singular_values else 1.0
                sv_text = f"\u03c3_bar: {sigma_bar:.3f} (GMD) | Uniform SNR: {user.stream_snrs_db[0]:.1f} dB" if user.stream_snrs_db else f"\u03c3_bar: {sigma_bar:.3f} (GMD)"
                sv_color = "#00e676"
            
            tk.Label(sv_frame, text=sv_text, fg=sv_color, bg="#151c25", font=("Segoe UI", 9)).pack(anchor="w")
            
            snr_frame = tk.Frame(info_frame, bg="#151c25")
            snr_frame.pack(fill="x", pady=(4, 0))
            avg_snr_db = np.mean(user.stream_snrs_db) if user.stream_snrs_db else 0
            snr_text = f"Avg Stream SNR: {avg_snr_db:.1f} dB"
            tk.Label(snr_frame, text=snr_text, fg="#ff9f43", bg="#151c25", font=("Segoe UI", 9)).pack(anchor="w")
            
            stream_frame = tk.Frame(info_frame, bg="#151c25")
            stream_frame.pack(fill="x", pady=(4, 0))
            stream_text = "Stream SNRs: " + ", ".join([f"{s:.1f} dB" for s in user.stream_snrs_db])
            tk.Label(stream_frame, text=stream_text, fg="#8b949e", bg="#151c25", font=("Segoe UI", 8)).pack(anchor="w")

    def _build_system_info(self, parent: ttk.Frame) -> None:
        info_frame = ttk.LabelFrame(parent, text="System Info", padding=10)
        info_frame.pack(fill="both", expand=True)
        
        total_gmi = sum(u.gmi for u in self.user_states)
        ttk.Label(info_frame, text=f"Total GMI: {total_gmi:.2f} bits", foreground=ACCENT, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(info_frame, text=f"Channel: {self.params.channel_type.upper()}", style="Status.TLabel").pack(anchor="w")
        ttk.Label(info_frame, text=f"RF Chains: {self.params.num_rf_chains}", style="Status.TLabel").pack(anchor="w")
        ttk.Label(info_frame, text=f"Modulation: {self.params.modulation} ({MOD_BITS.get(self.params.modulation, 4)} bits/symbol)", style="Status.TLabel").pack(anchor="w")
        
        snr_total = self.params.tx_power_dbm - (-70)
        ttk.Label(info_frame, text=f"Tx Power: {self.params.tx_power_dbm:.1f} dBm | Noise: -70 dBm", foreground="#ff9f43", font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Label(info_frame, text=f"System SNR: {snr_total:.1f} dB", foreground=ACCENT, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        
        if FULL_DIGITAL_AVAILABLE:
            ttk.Label(info_frame, text="✓ Using full_digital_mu", foreground="#34d399", font=("Segoe UI", 9)).pack(anchor="w")
        else:
            ttk.Label(info_frame, text="⚠ Using fallback implementation", foreground="#f59e0b", font=("Segoe UI", 9)).pack(anchor="w")

    @staticmethod
    def _geometric_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        sigma = np.maximum(np.asarray(values, dtype=float), 1e-12)
        return float(np.exp(np.mean(np.log(sigma))))

    @staticmethod
    def _format_float_vector(values: list[float], decimals: int = 2) -> str:
        if not values:
            return "N/A"
        return ", ".join(f"{value:.{decimals}f}" for value in values)

    def _build_info_panel(self, parent: ttk.Frame) -> None:
        info_frame = ttk.LabelFrame(parent, text="Current Info", padding=10)
        info_frame.pack(fill="both", expand=True)

        self.decomp_scrollbar = tk.Scrollbar(
            info_frame,
            orient="vertical",
            bg="#1e2d3d",
            activebackground="#2d4a6a",
            troughcolor="#1a2332",
            borderwidth=1,
            relief="flat",
            width=16,
        )
        self.decomp_scrollbar.pack(side="right", fill="y")

        self.decomp_canvas = tk.Canvas(
            info_frame,
            bg=PANEL,
            highlightthickness=0,
            bd=0,
            yscrollcommand=self.decomp_scrollbar.set,
        )
        self.decomp_canvas.pack(side="left", fill="both", expand=True)
        self.decomp_scrollbar.config(command=self.decomp_canvas.yview)

        self.decomp_scrollable = tk.Frame(self.decomp_canvas, bg="#1a2332")
        self.decomp_canvas.create_window((0, 0), window=self.decomp_scrollable, anchor="nw")
        self.decomp_scrollable.bind(
            "<Configure>",
            lambda _e: self.decomp_canvas.configure(scrollregion=self.decomp_canvas.bbox("all")),
        )

        self._build_all_info_rows()
        self._bind_mousewheel_to_widget_and_children(self.decomp_canvas, self.decomp_scrollable)
        self.after(100, lambda: self.decomp_canvas.configure(scrollregion=self.decomp_canvas.bbox("all")))

    def _build_all_info_rows(self) -> None:
        for widget in self.decomp_scrollable.winfo_children():
            widget.destroy()

        summary_frame = tk.Frame(
            self.decomp_scrollable,
            bg="#151c25",
            highlightbackground="#1e2d3d",
            highlightthickness=1,
            padx=10,
            pady=10,
        )
        summary_frame.pack(fill="x", pady=(0, 8))

        topology_text = (
            f"K={self.params.num_users} | Nt={self.params.num_tx_antennas} | "
            f"Nr={self.params.num_rx_antennas} | Nrf={self.params.num_rf_chains} | "
            f"Ns={self.params.num_streams_per_user}"
        )
        tk.Label(
            summary_frame,
            text=topology_text,
            fg="#ffffff",
            bg="#151c25",
            font=("Segoe UI", 10, "bold"),
            justify="left",
            wraplength=300,
        ).pack(anchor="w")
        tk.Label(
            summary_frame,
            text=f"Current Method: {self.params.precoding_method}",
            fg=PRECODING_COLORS.get(self.params.precoding_method, ACCENT),
            bg="#151c25",
            font=("Segoe UI", 10, "bold"),
            justify="left",
            wraplength=300,
        ).pack(anchor="w", pady=(6, 0))

        for user in self.user_states:
            user_frame = tk.Frame(
                self.decomp_scrollable,
                bg="#151c25",
                highlightbackground="#1e2d3d",
                highlightthickness=1,
                padx=8,
                pady=8,
            )
            user_frame.pack(fill="x", pady=(0, 6))

            color_box = tk.Label(user_frame, bg=user.color, width=2, height=1)
            color_box.pack(side="left", padx=(0, 8))

            detail_frame = tk.Frame(user_frame, bg="#151c25")
            detail_frame.pack(side="left", fill="both", expand=True)

            tk.Label(
                detail_frame,
                text=f"User {user.id}",
                fg="#ffffff",
                bg="#151c25",
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w")
            tk.Label(
                detail_frame,
                text=f"Real GMI: {user.gmi:.2f} | Real BER: {user.ber:.3e}",
                fg="#00e676",
                bg="#151c25",
                font=("Segoe UI", 9),
                justify="left",
                wraplength=300,
            ).pack(anchor="w", pady=(4, 0))
            tk.Label(
                detail_frame,
                text="Stream SNR: " + self._format_float_vector(user.stream_snrs_db, decimals=1) + " dB",
                fg="#ff9f43",
                bg="#151c25",
                font=("Segoe UI", 9),
                justify="left",
                wraplength=300,
            ).pack(anchor="w", pady=(4, 0))
            tk.Label(
                detail_frame,
                text="Reduced sigma: " + self._format_float_vector(user.singular_values, decimals=3),
                fg="#00d4ff",
                bg="#151c25",
                font=("Segoe UI", 9),
                justify="left",
                wraplength=300,
            ).pack(anchor="w", pady=(4, 0))
            tk.Label(
                detail_frame,
                text=f"Reduced gmean: {self._geometric_mean(user.singular_values):.3f}",
                fg="#8b949e",
                bg="#151c25",
                font=("Segoe UI", 9),
                justify="left",
                wraplength=300,
            ).pack(anchor="w", pady=(4, 0))

    def _apply_params(self) -> None:
        try:
            self.params.num_users = int(self.users_combo.get())
            self.params.num_tx_antennas = int(self.nt_combo.get())
            self.params.num_rx_antennas = int(self.nr_combo.get())
            self.params.num_rf_chains = int(self.nrf_combo.get())
            self.params.num_streams_per_user = int(self.streams_combo.get())
            self.params.precoding_method = self.precoding_combo.get()
            self.params.modulation = self.modulation_combo.get()
            self.params.snr_db = self.snr_slider.get()
            
            total_streams = self.params.num_users * self.params.num_streams_per_user
            if total_streams > self.params.num_tx_antennas:
                print(f"Warning: {total_streams} streams > {self.params.num_tx_antennas} Tx antennas, reducing streams per user")
                self.params.num_streams_per_user = self.params.num_tx_antennas // self.params.num_users
                self.streams_combo.set(str(self.params.num_streams_per_user))

            if self.params.num_users > 1 and self.params.precoding_method == "UCD":
                self.params.precoding_method = "GMD"
                self.precoding_combo.set("GMD")
                messagebox.showinfo("Info", "Multi-user viewer currently supports SVD/GMD only. Switched UCD to GMD.")
            
            self._init_environments()
            self._init_channels()
            self._init_users()
            
            for widget in self.left_panel.winfo_children():
                widget.destroy()
            self._build_params_panel(self.left_panel)
            self._build_user_panel(self.left_panel)
            
            for widget in self.right_panel.winfo_children():
                widget.destroy()
            self._build_info_panel(self.right_panel)
            
            self._sync_canvas()
            self.update_idletasks()
            
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def _toggle_simulation(self) -> None:
        if self._running:
            self._running = False
            self.run_button.config(text="Start Simulation")
            self._stop_animation()
        else:
            self._running = True
            self.run_button.config(text="Stop Simulation")
            self._start_animation()
            self._update_simulation()

    def _reset_simulation(self) -> None:
        self._running = False
        self._stop_animation()
        self.run_button.config(text="Start Simulation")
        self.snr_slider.set(10.0)
        self.snr_value_label.config(text="10.0 dB")
        self.params.snr_db = 10.0
        self._init_environments()
        self._init_channels()
        self._init_users()
        self._sync_canvas()
        for widget in self.left_panel.winfo_children():
            widget.destroy()
        self._build_params_panel(self.left_panel)
        self._build_user_panel(self.left_panel)
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        self._build_info_panel(self.right_panel)

    def _on_snr_slider_release(self) -> None:
        val = self.snr_slider.get()
        self.params.snr_db = val
        self.snr_value_label.config(text=f"{val:.1f} dB")
        self._recalculate_users()

    def _recalculate_users(self) -> None:
        metrics = self._evaluate_current_chain()
        for i, user in enumerate(self.user_states):
            user.singular_values = list(metrics["singular_values"][i])
            user.stream_snrs_db = list(metrics["stream_snrs_db"][i])
            user.snr_db = float(metrics["snr_per_user_db"])
            user.sinr_db = float(metrics["sinr_per_user_db"][i])
            user.gmi = float(metrics["user_gmi"][i])
            user.ber = float(metrics["user_ber"][i])

        self._sync_canvas()
        self._update_user_panel()
        self._rebuild_info_display()

    def _update_simulation(self) -> None:
        if not self._running:
            return

        self._recalculate_users()
        self.after(500, self._update_simulation)

    def _update_user_panel(self) -> None:
        for i, (user_frame, rate_label) in enumerate(self.user_rows):
            if i < len(self.user_states):
                user = self.user_states[i]
                rate_label.config(text=f"Rate: {user.gmi:.2f} bits/symbol")

    def _rebuild_info_display(self) -> None:
        if hasattr(self, 'decomp_scrollable'):
            for widget in self.decomp_scrollable.winfo_children():
                widget.destroy()
            self._build_all_info_rows()
            if hasattr(self, 'decomp_canvas'):
                self.decomp_canvas.configure(scrollregion=self.decomp_canvas.bbox("all"))

    def _on_canvas_click(self, event: tk.Event) -> None:
        canvas = event.widget
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        
        for i, user in enumerate(self.user_states):
            dist = math.hypot(x - user.x, y - user.y)
            if dist < 35:
                self._selected_user = i
                self._sync_canvas()
                return
        
        self._selected_user = None
        self._sync_canvas()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._sync_canvas()

    def _sync_canvas(self) -> None:
        self.canvas.delete("all")
        width = self.canvas.winfo_width() or 1000
        height = self.canvas.winfo_height() or 800
        
        center_x, center_y = width // 2, height // 2
        
        draw_grid_background(self.canvas, width, height)
        
        self._draw_decomp_visualization(center_x, center_y)
        
        radius = 260
        for i, user in enumerate(self.user_states):
            angle = (2 * math.pi * i) / len(self.user_states) - math.pi / 2
            user.x = center_x + radius * math.cos(angle)
            user.y = center_y + radius * math.sin(angle)
            is_selected = self._selected_user == user.id - 1
            self._draw_user(user, is_selected)
        
        self._draw_base_station(center_x)
        
        for user in self.user_states:
            self._draw_channel_link(user, center_x)

    def _draw_base_station(self, center_x: int) -> None:
        y = 60
        
        self.canvas.create_rectangle(
            center_x - 60, y - 40,
            center_x + 60, y + 20,
            fill="#1e293b", outline="#f59e0b", width=2
        )
        
        self.canvas.create_polygon(
            center_x, y - 60,
            center_x - 15, y - 40,
            center_x + 15, y - 40,
            fill="#f59e0b", outline="#fbbf24"
        )
        
        for i in range(4):
            for j in range(4):
                tx_x = center_x - 45 + i * 30
                tx_y = y - 25 + j * 15
                self.canvas.create_rectangle(
                    tx_x - 8, tx_y - 4,
                    tx_x + 8, tx_y + 4,
                    fill="#f59e0b", outline="#fbbf24"
                )
        
        self.canvas.create_text(center_x, y + 35, text="Base Station", fill=TEXT, font=("Segoe UI", 12, "bold"))
        self.canvas.create_text(center_x, y + 52, text=f"{self.params.num_tx_antennas} Antennas", fill=MUTED, font=("Segoe UI", 9))

    def _draw_user(self, user: UserState, is_selected: bool) -> None:
        size = 30 + 15 * min(user.gmi / (self.params.num_streams_per_user * MOD_BITS.get(self.params.modulation, 4)), 1.0)
        
        if is_selected:
            self.canvas.create_oval(
                user.x - size - 8, user.y - size - 8,
                user.x + size + 8, user.y + size + 8,
                fill="", outline=user.color, width=2, dash=(4, 2)
            )
        
        gradient_steps = 8
        for i in range(gradient_steps):
            angle_start = (360 * i) / gradient_steps
            angle_end = (360 * (i + 1)) / gradient_steps
            inner_r = size / 2 - 3
            outer_r = size / 2
            
            self.canvas.create_arc(
                user.x - outer_r, user.y - outer_r,
                user.x + outer_r, user.y + outer_r,
                start=angle_start, extent=360/gradient_steps,
                fill=user.color if i % 2 == 0 else "#1e293b",
                outline=""
            )
        
        self.canvas.create_oval(
            user.x - size / 3, user.y - size / 3,
            user.x + size / 3, user.y + size / 3,
            fill="#0f1722", outline=user.color, width=2
        )
        
        dot_size = 6
        self.canvas.create_oval(
            user.x - dot_size / 2, user.y - dot_size / 2,
            user.x + dot_size / 2, user.y + dot_size / 2,
            fill=user.color, outline=""
        )
        
        self.canvas.create_text(user.x, user.y + size / 2 + 14, text=f"User {user.id}", fill=TEXT, font=("Segoe UI", 14, "bold"))
        rate_text = f"Rate: {user.gmi:.2f} bits/symbol"
        self.canvas.create_text(user.x, user.y + size / 2 + 32, text=rate_text, fill="#00e676", font=("Segoe UI", 12, "bold"))
        self.canvas.create_text(user.x, user.y + size / 2 + 48, text=f"SNR: {user.snr_db:.1f} dB", fill="#5bb6ff", font=("Segoe UI", 10))

    def _draw_channel_link(self, user: UserState, bs_x: int) -> None:
        bs_y = 60
        snr_db = user.snr_db
        opacity = min(1.0, snr_db / 30.0) if snr_db > 0 else 0.1
        
        rays = self.channel_analyzer.generate_rays(bs_x, bs_y + 20, user.x, user.y - 30, user.id)
        for ray in rays:
            color = self.channel_analyzer.get_ray_color(ray)
            draw_arrow_line(self.canvas, ray.start_x, ray.start_y, ray.end_x, ray.end_y, 
                           color=color, width=2, opacity=opacity)

    def _draw_decomp_visualization(self, cx: int, cy: int) -> None:
        radius = 120
        
        self.canvas.create_oval(
            cx - radius, cy - radius,
            cx + radius, cy + radius,
            fill="", outline="#2a3f5f", width=1, dash=(4, 4)
        )
        
        method = self.params.precoding_method
        color = PRECODING_COLORS.get(method, "#5bb6ff")
        self.canvas.create_text(cx, cy - radius - 20, text=f"{method} Decomposition", fill=color, font=("Segoe UI", 12, "bold"))
        
        if method == "SVD":
            for i, user in enumerate(self.user_states):
                angle = (2 * math.pi * i) / self.params.num_users
                user_cx = cx + radius * 0.6 * math.cos(angle)
                user_cy = cy + radius * 0.6 * math.sin(angle)
                
                self.canvas.create_oval(user_cx - 25, user_cy - 25, user_cx + 25, user_cy + 25, fill="", outline=user.color, width=1)
                
                max_sv = max(user.singular_values) if user.singular_values else 1.0
                for j, sv in enumerate(user.singular_values[:4]):
                    sv_angle = angle + (2 * math.pi * j) / len(user.singular_values)
                    sv_r = 15 * (sv / max_sv)
                    sv_x = user_cx + sv_r * math.cos(sv_angle)
                    sv_y = user_cy + sv_r * math.sin(sv_angle)
                    self.canvas.create_oval(sv_x - 5, sv_y - 5, sv_x + 5, sv_y + 5, fill=user.color, outline="")
                
                self.canvas.create_text(user_cx, user_cy + 35, text=f"U{user.id}", fill=MUTED, font=("Segoe UI", 8))
        
        elif method == "GMD":
            sigma_bar_values = []
            for user in self.user_states:
                if user.singular_values:
                    sigma_bar_values.append(np.prod(user.singular_values) ** (1.0 / len(user.singular_values)))
                else:
                    sigma_bar_values.append(1.0)
            
            avg_sigma_bar = np.mean(sigma_bar_values) if sigma_bar_values else 1.0
            
            for i, user in enumerate(self.user_states):
                angle = (2 * math.pi * i) / self.params.num_users
                user_cx = cx + radius * 0.6 * math.cos(angle)
                user_cy = cy + radius * 0.6 * math.sin(angle)
                
                self.canvas.create_polygon(
                    user_cx, user_cy - 20,
                    user_cx - 18, user_cy + 10,
                    user_cx + 18, user_cy + 10,
                    fill="", outline=user.color, width=2
                )
                
                gmd_r = 12 * (avg_sigma_bar / 2.0)
                self.canvas.create_oval(user_cx - gmd_r, user_cy - gmd_r, user_cx + gmd_r, user_cy + gmd_r, fill=user.color, outline="")
                
                self.canvas.create_text(user_cx, user_cy + 35, text=f"U{user.id}", fill=MUTED, font=("Segoe UI", 8))
            
            self.canvas.create_text(cx, cy + radius + 15, text=f"σ_bar = {avg_sigma_bar:.3f}", fill=color, font=("Segoe UI", 10))

    def _generate_comparison_plot(self) -> None:
        if not PLOTTING_AVAILABLE:
            messagebox.showerror("Error", "Matplotlib is not available. Please install matplotlib to generate comparison plots.")
            return
        
        snr_range = np.linspace(0, 40, 21)
        methods = ["SVD", "GMD"] if self.params.num_users > 1 else ["SVD", "GMD", "UCD"]
        colors = {"SVD": "#5bb6ff", "GMD": "#34d399", "UCD": "#f59e0b"}
        results = {method: [] for method in methods}
        for snr_db in snr_range:
            for method in methods:
                metrics = self._evaluate_method_at_snr(method, float(snr_db))
                results[method].append(float(metrics["sum_rate"]))

        title_scope = f"Single-User" if self.params.num_users == 1 else f"Multi-User (K={self.params.num_users})"
        title_methods = " vs ".join(methods)
        self._plot_results(snr_range, results, colors, title=f"{title_scope} - {title_methods}")
    
    def _plot_results(self, snr_range: np.ndarray, results: dict[str, list[float]], colors: dict[str, str], title: str) -> None:
        plt.figure(figsize=(10, 6), dpi=120)
        
        for method, gmi_values in results.items():
            plt.plot(snr_range, gmi_values, label=method, color=colors[method], linewidth=3, marker='o', markersize=6)
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('GMI (bits/symbol)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-2, 42)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = SimulinkViewer()
    app.mainloop()
