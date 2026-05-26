"""
6G MIMO Precoding Simulator - Based on Original Code Style
"""
from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk, messagebox

import numpy as np
from scipy.linalg import svd

# 导入 Sionna CDL-A 信道模型
from cdl_a_channel import sample_cdl_a_channels_numpy

# 导入全数字多用户 MIMO 环境
from full_digital_mu.fd_mu_environment import FullyDigitalMuMimoBicmEnvironment

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

USER_NODE_COLORS = ["#5bb6ff", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a855f7", "#fb7185", "#34d399", "#fbbf24"]

MOD_BITS = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}


@dataclass
class SimulationParams:
    num_users: int = 2
    num_tx_antennas: int = 16
    num_rx_antennas: int = 4
    num_streams_per_user: int = 4
    snr_db: float = 10.0
    modulation: str = "64QAM"
    precoding_method: str = "SVD"
    carrier_freq: float = 3.5e9


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
    distance: float = 1.0


class MIMOSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6G MIMO Precoding Simulator")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.configure(bg=BG)

        self.params = SimulationParams()
        self.user_states = []
        self.channel_matrix = None
        self._anim_running = False
        self._anim_time = 0.0

        self._build_style()
        self._init_channels()
        self._init_users()
        self._build_layout()

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("Field.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))

    def _init_channels(self):
        # 频率映射（Hz）
        freq_map = {
            "3.5GHz": 3.5e9,
            "28GHz": 28e9,
            "140GHz": 140e9,
            "300GHz": 300e9
        }
        carrier_freq = freq_map.get(self.params.carrier_freq, 3.5e9)
        
        # 使用 Sionna CDL-A 信道模型
        h = sample_cdl_a_channels_numpy(
            batch_size=self.params.num_users,
            num_rx_antennas=self.params.num_rx_antennas,
            num_tx_antennas=self.params.num_tx_antennas,
            seed=random.randint(0, 10000),
            carrier_frequency=carrier_freq
        )
        self.channel_matrix = h

    def _evaluate_method(self, method, snr_db):
        method = method.upper()
        num_users = self.params.num_users
        num_streams = self.params.num_streams_per_user
        total_snr_linear = 10 ** (float(snr_db) / 10.0)
        snr_per_stream = total_snr_linear / (num_users * num_streams)
        
        # 使用真正的 full_digital_mu 模块
        env = FullyDigitalMuMimoBicmEnvironment(
            num_users=num_users,
            num_tx_antennas=self.params.num_tx_antennas,
            num_rx_antennas=self.params.num_rx_antennas,
            num_streams_per_user=num_streams,
            channel_type="cdl-a"
        )
        
        user_channels = self.channel_matrix
        
        # 构建结构化预编码链
        chain = env.build_structured_chain(
            user_channels=user_channels,
            snr_per_stream=snr_per_stream,
            strategy=method.lower()
        )
        f = chain.f_bb
        r_chains = chain.r_chains
        q_chains = chain.q_chains
        
        # 评估性能
        bits_per_symbol = MOD_BITS.get(self.params.modulation, 4)
        
        # 创建 Monte Carlo 采样对象
        from classical.sic_sample_average import build_multiuser_sample_average
        sample_average = build_multiuser_sample_average(
            env=env,
            bits_per_symbol=bits_per_symbol,
            num_samples=100,
            num_repeats=16,
            base_seed=random.randint(0, 10000)
        )
        
        if method == "UCD":
            metrics = env.evaluate_ucd_precoder_current_receiver_average_b_chain(
                user_channels=user_channels,
                f=f,
                q_chains=q_chains,
                r_chains=r_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=bits_per_symbol,
                sample_average=sample_average
            )
        else:
            metrics = env.evaluate_precoder_current_receiver_average_fixed_chain(
                user_channels=user_channels,
                f=f,
                r_chains=r_chains,
                q_chains=q_chains,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=bits_per_symbol,
                sample_average=sample_average
            )
        
        # 提取每个用户的奇异值和速率
        singular_values = []
        stream_snrs_db = []
        user_gmi = []
        user_ber = []
        user_sinr_db = []
        
        for user_idx in range(num_users):
            # 从 r_chains 中获取实际的奇异值（对角矩阵的对角元素）
            r_local = r_chains[user_idx]
            s_eff = np.diag(r_local)[:num_streams]
            if len(s_eff) < num_streams:
                s_eff = np.pad(s_eff, (0, num_streams - len(s_eff)), constant_values=1e-12)
            
            singular_values.append(np.abs(s_eff).tolist())
            
            # 使用实际计算的速率
            user_rate = metrics.user_rates[user_idx]
            user_gmi.append(float(user_rate))
            user_ber.append(float(metrics.user_bit_error_rates[user_idx]))
            
            # 计算等效 SINR
            sinr_linear = (2 ** user_rate - 1) / num_streams
            user_sinr_db.append(float(10.0 * np.log10(max(sinr_linear, 1e-12))))
            
            # 流级 SNR（基于奇异值）
            stream_snr = 10.0 * np.log10(np.maximum(s_eff**2 * snr_per_stream, 1e-12))
            stream_snrs_db.append(stream_snr.tolist())
        
        snr_per_user_db = float(10.0 * np.log10(max(total_snr_linear / num_users, 1e-12)))
        return {
            "snr_per_user_db": snr_per_user_db,
            "sinr_per_user_db": user_sinr_db,
            "stream_snrs_db": stream_snrs_db,
            "singular_values": singular_values,
            "user_gmi": user_gmi,
            "user_ber": user_ber,
            "sum_rate": float(metrics.sum_rate),
        }

    def _init_users(self):
        self.user_states = []
        # 每次初始化用户时都重新生成信道
        self._init_channels()
        metrics = self._evaluate_method(self.params.precoding_method, self.params.snr_db)

        for i in range(self.params.num_users):
            angle = (2 * math.pi * i) / self.params.num_users - math.pi / 2
            radius = 260
            x = 700 + radius * math.cos(angle)
            y = 450 + radius * math.sin(angle)
            self.user_states.append(UserState(
                id=i + 1,
                x=x,
                y=y,
                distance=radius,
                snr_db=float(metrics["snr_per_user_db"]),
                sinr_db=float(metrics["sinr_per_user_db"][i]),
                stream_snrs_db=list(metrics["stream_snrs_db"][i]),
                singular_values=list(metrics["singular_values"][i]),
                gmi=float(metrics["user_gmi"][i]),
                ber=float(metrics["user_ber"][i]),
                color=USER_NODE_COLORS[i % len(USER_NODE_COLORS)]
            ))

    def _build_layout(self):
        root = ttk.Frame(self, style="App.TFrame", padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="6G MIMO Precoding Simulator", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="SVD/GMD/UCD Precoding Visualization", style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))

        control_frame = ttk.Frame(header, style="App.TFrame")
        control_frame.grid(row=0, column=2, rowspan=2, sticky="e")
        ttk.Button(control_frame, text="Run", command=self._run_simulation).pack(side="right", padx=(8, 0))
        ttk.Button(control_frame, text="Reset", command=self._reset_simulation).pack(side="right", padx=(8, 0))

        self.left_panel = ttk.Frame(root, style="Panel.TFrame", width=320)
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
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.right_panel = ttk.Frame(root, style="Panel.TFrame", width=360)
        self.right_panel.grid(row=1, column=2, sticky="ns", padx=(12, 0))
        self.right_panel.pack_propagate(False)
        self._build_info_panel(self.right_panel)

        self._sync_canvas()

    def _build_params_panel(self, parent):
        params_frame = ttk.LabelFrame(parent, text="System Parameters", padding=10)
        params_frame.pack(fill="x", pady=(0, 12))

        row = 0
        ttk.Label(params_frame, text="Users", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.users_combo = ttk.Combobox(params_frame, values=["1", "2", "3", "4", "5", "6", "7", "8"], state="readonly", width=8)
        self.users_combo.set(str(self.params.num_users))
        self.users_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Tx", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.nt_combo = ttk.Combobox(params_frame, values=["4", "8", "16", "32", "64", "256"], state="readonly", width=8)
        self.nt_combo.set(str(self.params.num_tx_antennas))
        self.nt_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Rx", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.nr_combo = ttk.Combobox(params_frame, values=["2", "4", "8", "16"], state="readonly", width=8)
        self.nr_combo.set(str(self.params.num_rx_antennas))
        self.nr_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Ns/User", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.streams_combo = ttk.Combobox(params_frame, values=["1", "2", "4", "8"], state="readonly", width=8)
        self.streams_combo.set(str(self.params.num_streams_per_user))
        self.streams_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Precoding", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.precoding_combo = ttk.Combobox(params_frame, values=["SVD", "GMD", "UCD"], state="readonly", width=8)
        self.precoding_combo.set(self.params.precoding_method)
        self.precoding_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="Frequency", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.freq_combo = ttk.Combobox(params_frame, 
            values=["3.5GHz (Sub-6G)", "28GHz (mmWave)", "140GHz (Sub-THz)"], state="readonly", width=18)
        self.freq_combo.set("3.5GHz (Sub-6G)")
        self.freq_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(params_frame, text="SNR (dB)", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.snr_slider = ttk.Scale(params_frame, from_=0, to=40, orient="horizontal", length=150)
        self.snr_slider.set(self.params.snr_db)
        self.snr_slider.grid(row=row, column=1, sticky="w", padx=(8, 0))

        self.snr_value_label = ttk.Label(params_frame, text=f"{self.params.snr_db:.1f} dB", foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self.snr_value_label.grid(row=row, column=2, sticky="w", padx=(4, 0))
        self.snr_slider.bind("<ButtonRelease-1>", lambda e: self.snr_value_label.config(text=f"{self.snr_slider.get():.1f} dB"))

        apply_btn = ttk.Button(params_frame, text="Apply", command=self._apply_params)
        apply_btn.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def _build_user_panel(self, parent):
        user_frame = ttk.LabelFrame(parent, text="User Metrics", padding=12)
        user_frame.pack(fill="both", expand=True)

        self.user_scrollbar = tk.Scrollbar(user_frame, orient="vertical", bg="#1e2d3d", activebackground="#2d4a6a", 
                                           troughcolor="#1a2332", borderwidth=1, relief="flat", width=16)
        self.user_scrollbar.pack(side="right", fill="y")

        self.user_canvas = tk.Canvas(user_frame, bg="#1a2332", highlightthickness=0, bd=0, yscrollcommand=self.user_scrollbar.set)
        self.user_canvas.pack(side="left", fill="both", expand=True)
        self.user_scrollbar.config(command=self.user_canvas.yview)

        self.user_scrollable = tk.Frame(self.user_canvas, bg="#1a2332")
        self.user_canvas.create_window((0, 0), window=self.user_scrollable, anchor="nw")

        self.user_scrollable.bind("<Configure>", lambda e: self.user_canvas.configure(scrollregion=self.user_canvas.bbox("all")))
        self._build_user_rows()

    def _build_user_rows(self):
        for widget in self.user_scrollable.winfo_children():
            widget.destroy()

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

            tk.Label(info_frame, text=f"SNR: {user.snr_db:.1f} dB | SINR: {user.sinr_db:.1f} dB", fg="#7dd3fc", 
                     bg="#223044", font=("Segoe UI", 9)).pack(anchor="w")

            stream_text = "Streams: " + ", ".join([f"{s:.1f} dB" for s in user.stream_snrs_db])
            tk.Label(info_frame, text=stream_text, fg="#b9cbe0", bg="#223044", font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

    def _build_info_panel(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Decomposition Info", padding=10)
        info_frame.pack(fill="both", expand=True)

        self.info_scrollbar = tk.Scrollbar(info_frame, orient="vertical", bg="#1e2d3d", activebackground="#2d4a6a", 
                                           troughcolor="#1a2332", borderwidth=1, relief="flat", width=16)
        self.info_scrollbar.pack(side="right", fill="y")

        self.info_canvas = tk.Canvas(info_frame, bg=PANEL, highlightthickness=0, bd=0, yscrollcommand=self.info_scrollbar.set)
        self.info_canvas.pack(side="left", fill="both", expand=True)
        self.info_scrollbar.config(command=self.info_canvas.yview)

        self.info_scrollable = tk.Frame(self.info_canvas, bg="#1a2332")
        self.info_canvas.create_window((0, 0), window=self.info_scrollable, anchor="nw")
        self.info_scrollable.bind("<Configure>", lambda e: self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all")))
        self._build_info_rows()

    def _build_info_rows(self):
        for widget in self.info_scrollable.winfo_children():
            widget.destroy()

        summary_frame = tk.Frame(self.info_scrollable, bg="#151c25", highlightbackground="#1e2d3d", highlightthickness=1, padx=10, pady=10)
        summary_frame.pack(fill="x", pady=(0, 8))

        topology_text = f"K={self.params.num_users} | Nt={self.params.num_tx_antennas} | Nr={self.params.num_rx_antennas} | Ns={self.params.num_streams_per_user}"
        tk.Label(summary_frame, text=topology_text, fg="#ffffff", bg="#151c25", font=("Segoe UI", 10)).pack(anchor="w")

        sum_rate = sum(u.gmi for u in self.user_states)
        tk.Label(summary_frame, text=f"Total Rate: {sum_rate:.2f} bits/symbol", fg=ACCENT, bg="#151c25", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 0))
        tk.Label(summary_frame, text=f"Precoding: {self.params.precoding_method} | Modulation: {self.params.modulation}", fg=MUTED, bg="#151c25", font=("Segoe UI", 9)).pack(anchor="w")

        for user in self.user_states:
            user_frame = tk.Frame(self.info_scrollable, bg="#151c25", highlightbackground="#1e2d3d", highlightthickness=1, padx=8, pady=8)
            user_frame.pack(fill="x", pady=(0, 6))

            color_box = tk.Label(user_frame, bg=user.color, width=2, height=1)
            color_box.pack(side="left", padx=(0, 8))

            info_frame = tk.Frame(user_frame, bg="#151c25")
            info_frame.pack(side="left", fill="both", expand=True)

            tk.Label(info_frame, text=f"User {user.id}", fg="#ffffff", bg="#151c25", font=("Segoe UI", 10, "bold")).pack(anchor="w")

            sv_text = "Singular Values: " + ", ".join([f"{sv:.3f}" for sv in user.singular_values])
            tk.Label(info_frame, text=sv_text, fg="#00d4ff", bg="#151c25", font=("Segoe UI", 9)).pack(anchor="w")

            avg_snr_db = np.mean(user.stream_snrs_db) if user.stream_snrs_db else 0
            tk.Label(info_frame, text=f"Avg Stream SNR: {avg_snr_db:.1f} dB", fg="#ff9f43", bg="#151c25", font=("Segoe UI", 9)).pack(anchor="w")

    def _sync_canvas(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width() or 1400
        height = self.canvas.winfo_height() or 800
        center_x, center_y = width // 2, height // 2

        # === Simulink风格基站图标 ===
        # 基站塔主体（梯形底座）
        bs_base_w = 60
        bs_tower_h = 80
        self.canvas.create_polygon(
            center_x - bs_base_w/2, center_y + bs_tower_h/2,
            center_x + bs_base_w/2, center_y + bs_tower_h/2,
            center_x + bs_base_w/4, center_y - bs_tower_h/2,
            center_x - bs_base_w/4, center_y - bs_tower_h/2,
            fill="#2563eb", outline="#60a5fa", width=2
        )
        
        # 塔顶天线阵列（真正的天线图标 - 偶极子天线）
        # 天线位置
        ant_base = center_y - bs_tower_h/2 - 10
        
        # 左侧天线
        self._draw_dipole_antenna(center_x - 20, ant_base)
        # 中间天线
        self._draw_dipole_antenna(center_x, ant_base)
        # 右侧天线
        self._draw_dipole_antenna(center_x + 20, ant_base)
        
        # 天线支架
        self.canvas.create_line(center_x - 25, ant_base + 15, center_x + 25, ant_base + 15, 
                               fill="#64748b", width=2)
        self.canvas.create_line(center_x, ant_base + 15, center_x, center_y - bs_tower_h/2, 
                               fill="#64748b", width=2)
        
        # 基站标签
        self.canvas.create_text(center_x, center_y + bs_tower_h/2 + 30, text="BS", 
                               fill="#ffffff", font=("Segoe UI", 14, "bold"))
        self.canvas.create_text(center_x, center_y + bs_tower_h/2 + 48, text="Base Station", 
                               fill="#94a3b8", font=("Segoe UI", 9))

        # === Simulink风格用户图标 ===
        for user in self.user_states:
            angle = (2 * math.pi * (user.id - 1)) / self.params.num_users - math.pi / 2
            radius = 220 + (user.id - 1) * 20
            user.x = center_x + radius * math.cos(angle)
            user.y = center_y + radius * math.sin(angle)

            # 绘制信道连线
            self.canvas.create_line(center_x, center_y, user.x, user.y, fill="#475569", width=2, dash=(4, 4))
            
            # === 用户设备图标（手机形状）===
            # 手机主体
            phone_w = 36
            phone_h = 60
            self.canvas.create_rectangle(
                user.x - phone_w/2, user.y - phone_h/2,
                user.x + phone_w/2, user.y + phone_h/2,
                fill=user.color, outline="#334155", width=2
            )
            # 手机圆角
            self.canvas.create_oval(
                user.x - phone_w/2, user.y - phone_h/2,
                user.x - phone_w/2 + 8, user.y - phone_h/2 + 8,
                fill=user.color, outline="#334155"
            )
            self.canvas.create_oval(
                user.x + phone_w/2 - 8, user.y - phone_h/2,
                user.x + phone_w/2, user.y - phone_h/2 + 8,
                fill=user.color, outline="#334155"
            )
            # 屏幕
            screen_w = 28
            screen_h = 40
            self.canvas.create_rectangle(
                user.x - screen_w/2, user.y - screen_h/2 + 5,
                user.x + screen_w/2, user.y + screen_h/2 + 5,
                fill="#1e293b", outline="#64748b", width=1
            )
            # 用户编号
            self.canvas.create_text(user.x, user.y + 5, text=str(user.id), 
                                   fill="#f1f5f9", font=("Segoe UI", 12, "bold"))
            
            # 用户标签
            self.canvas.create_text(user.x, user.y + phone_h/2 + 25, text=f"User {user.id}", 
                                   fill="#f1f5f9", font=("Segoe UI", 10))

    def _draw_dipole_antenna(self, x, y):
        """绘制偶极子天线图标"""
        # 天线臂长度
        arm_len = 12
        
        # 上半部分天线臂
        self.canvas.create_line(x, y - arm_len, x, y - 3, fill="#ef4444", width=2)
        # 下半部分天线臂
        self.canvas.create_line(x, y + 3, x, y + arm_len, fill="#ef4444", width=2)
        
        # 天线中间的绝缘子/馈电点
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#fbbf24", outline="#f59e0b")
        
        # 天线支架底座
        self.canvas.create_line(x - 5, y + arm_len, x + 5, y + arm_len, fill="#64748b", width=2)
        self.canvas.create_line(x, y + arm_len, x, y + arm_len + 8, fill="#64748b", width=2)

    def _on_canvas_resize(self, event):
        self._sync_canvas()

    def _apply_params(self):
        self.params.num_users = int(self.users_combo.get())
        self.params.num_tx_antennas = int(self.nt_combo.get())
        self.params.num_rx_antennas = int(self.nr_combo.get())
        self.params.num_streams_per_user = int(self.streams_combo.get())
        self.params.precoding_method = self.precoding_combo.get()
        self.params.snr_db = self.snr_slider.get()

        freq_text = self.freq_combo.get()
        if "3.5GHz" in freq_text:
            self.params.carrier_freq = 3.5e9
        elif "28GHz" in freq_text:
            self.params.carrier_freq = 28e9
        elif "140GHz" in freq_text:
            self.params.carrier_freq = 140e9

        self._init_channels()
        self._init_users()
        self._build_user_rows()
        self._build_info_rows()
        self._sync_canvas()

    def _run_simulation(self):
        self._apply_params()
        messagebox.showinfo("Simulation", "Simulation completed!")

    def _reset_simulation(self):
        self.params = SimulationParams()
        self.users_combo.set("2")
        self.nt_combo.set("16")
        self.nr_combo.set("4")
        self.streams_combo.set("4")
        self.precoding_combo.set("SVD")
        self.freq_combo.set("3.5GHz (Sub-6G)")
        self.snr_slider.set(10)
        self.snr_value_label.config(text="10.0 dB")
        
        self._init_channels()
        self._init_users()
        self._build_user_rows()
        self._build_info_rows()
        self._sync_canvas()


def main():
    app = MIMOSimulator()
    app.mainloop()

if __name__ == "__main__":
    main()
