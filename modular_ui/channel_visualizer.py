from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from .config_6g import band_from_carrier_frequency, is_xl_mimo
from .node_graph_core import TensorPayload


@dataclass(slots=True)
class BeamSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    color: str
    width: float
    alpha: float
    is_los: bool = True


@dataclass(slots=True)
class ScatterPath:
    via: tuple[float, float]
    user_index: int
    color: str
    alpha: float


@dataclass(slots=True)
class VisualSceneState:
    title: str
    is_sub_thz: bool
    is_xl_mimo: bool
    num_users: int
    num_tx_antennas: int
    base_station: tuple[float, float] = (0.0, 0.0)
    user_positions: list[tuple[float, float]] = field(default_factory=list)
    beams: list[BeamSegment] = field(default_factory=list)
    scatter_paths: list[ScatterPath] = field(default_factory=list)
    metrics: dict[str, float | str] = field(default_factory=dict)


class ChannelVisualizer:
    """
    6G-oriented scene visualizer for the node-based workbench.

    The renderer emphasizes:
    - wide-angle multipath for baseline cases
    - pencil-beam LoS dominance for Sub-THz cases
    - compact metadata overlays that reflect the active profile
    """

    USER_COLORS = ("#52b7ff", "#63e6be", "#ffd166", "#ff7b72", "#d2a8ff", "#ff9f43")

    def __init__(self, fig=None, ax=None):
        self.fig = fig or plt.figure(figsize=(10.5, 7.8))
        self.ax = ax or self.fig.add_subplot(111)
        self.scene_state = VisualSceneState(
            title="6G Channel Visualizer",
            is_sub_thz=False,
            is_xl_mimo=False,
            num_users=2,
            num_tx_antennas=16,
        )
        self._animated_artists: list = []
        self.animation: FuncAnimation | None = None
        self._phase = 0.0
        self._init_axes()

    def _init_axes(self) -> None:
        self.ax.set_facecolor("#06111f")
        self.fig.patch.set_facecolor("#06111f")
        self.ax.set_xlim(-9.5, 9.5)
        self.ax.set_ylim(-7.5, 7.5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("Azimuth Footprint", color="#9db7d7")
        self.ax.set_ylabel("Elevation Footprint", color="#9db7d7")
        self.ax.tick_params(colors="#6f89a8")
        for spine in self.ax.spines.values():
            spine.set_color("#18314d")
        self.ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.15, color="#3b597a")

    def update_from_payload(self, payload: TensorPayload) -> VisualSceneState:
        band = band_from_carrier_frequency(payload.carrier_freq)
        scene = self.build_scene(
            carrier_freq=payload.carrier_freq,
            num_users=payload.num_users,
            num_tx_antennas=payload.num_tx_antennas,
            snr=payload.snr,
            metadata=payload.metadata,
            stream_id=payload.stream_id,
        )
        scene.metrics["band"] = band.value
        scene.metrics["snr_db"] = float(payload.snr)
        scene.metrics["stream_id"] = payload.stream_id
        self.scene_state = scene
        return scene

    def build_scene(
        self,
        *,
        carrier_freq: float,
        num_users: int,
        num_tx_antennas: int,
        snr: float,
        metadata: dict | None = None,
        stream_id: str = "default",
    ) -> VisualSceneState:
        metadata = dict(metadata or {})
        band = band_from_carrier_frequency(carrier_freq)
        sub_thz = bool(metadata.get("is_sub_thz", band.is_sub_thz))
        xl_mode = bool(metadata.get("is_xl_mimo", is_xl_mimo(num_tx_antennas)))
        positions = self._generate_user_positions(num_users=num_users, sub_thz=sub_thz, xl_mode=xl_mode)
        beams = self._generate_beams(
            positions=positions,
            sub_thz=sub_thz,
            xl_mode=xl_mode,
            num_tx_antennas=num_tx_antennas,
        )
        scatters = self._generate_scatter_paths(positions=positions, enabled=not (sub_thz or xl_mode))

        title_flags: list[str] = []
        if sub_thz:
            title_flags.append("Sub-THz Pencil Beams")
        suffix = f" | {' + '.join(title_flags)}" if title_flags else ""
        return VisualSceneState(
            title=f"6G Channel Visualizer{suffix}",
            is_sub_thz=sub_thz,
            is_xl_mimo=xl_mode,
            num_users=num_users,
            num_tx_antennas=num_tx_antennas,
            user_positions=positions,
            beams=beams,
            scatter_paths=scatters,
            metrics={
                "carrier_freq_hz": float(carrier_freq),
                "snr_db": float(snr),
                "stream_id": stream_id,
                "num_users": int(num_users),
                "num_tx_antennas": int(num_tx_antennas),
            },
        )

    def _generate_user_positions(
        self,
        *,
        num_users: int,
        sub_thz: bool,
        xl_mode: bool,
    ) -> list[tuple[float, float]]:
        radius = 6.6 if sub_thz else 5.9
        if xl_mode:
            radius = 7.0
        positions: list[tuple[float, float]] = []
        for idx in range(max(num_users, 1)):
            angle = (2.0 * math.pi * idx / max(num_users, 1)) - math.pi / 2.0
            wobble = 0.35 * math.sin(idx * 1.4) if not sub_thz else 0.1 * math.cos(idx)
            positions.append((radius * math.cos(angle), radius * math.sin(angle) + wobble))
        return positions

    def _generate_beams(
        self,
        *,
        positions: list[tuple[float, float]],
        sub_thz: bool,
        xl_mode: bool,
        num_tx_antennas: int,
    ) -> list[BeamSegment]:
        beams: list[BeamSegment] = []
        for idx, (ux, uy) in enumerate(positions):
            color = self.USER_COLORS[idx % len(self.USER_COLORS)]
            base_width = 2.8 if sub_thz else 1.7
            if xl_mode:
                base_width += 0.6
            beams.append(
                BeamSegment(
                    start=(0.0, 0.0),
                    end=(ux, uy),
                    color=color,
                    width=base_width,
                    alpha=0.95 if sub_thz else 0.82,
                    is_los=True,
                )
            )
            if sub_thz or xl_mode:
                side_lobes = 2 if sub_thz else 1
                beam_length = math.hypot(ux, uy) * 0.96
                angle = math.atan2(uy, ux)
                spread = 0.055 if sub_thz else 0.09
                for lobe_idx in range(side_lobes):
                    offset = spread * (lobe_idx + 1)
                    for sign in (-1.0, 1.0):
                        theta = angle + sign * offset
                        beams.append(
                            BeamSegment(
                                start=(0.0, 0.0),
                                end=(beam_length * math.cos(theta), beam_length * math.sin(theta)),
                                color=color,
                                width=max(0.9, base_width * 0.35),
                                alpha=0.22 if sub_thz else 0.16,
                                is_los=False,
                            )
                        )
        return beams

    def _generate_scatter_paths(
        self,
        *,
        positions: list[tuple[float, float]],
        enabled: bool,
    ) -> list[ScatterPath]:
        if not enabled:
            return []
        base_points = [(-3.8, 2.4), (2.1, 4.6), (-4.7, -2.8), (3.7, -1.9), (0.8, 5.5), (-1.6, -4.9)]
        scatters: list[ScatterPath] = []
        for idx, _ in enumerate(positions):
            for offset in range(2):
                sx, sy = base_points[(idx + offset) % len(base_points)]
                scatters.append(
                    ScatterPath(
                        via=(sx + 0.15 * idx, sy - 0.1 * idx),
                        user_index=idx,
                        color=self.USER_COLORS[idx % len(self.USER_COLORS)],
                        alpha=0.28,
                    )
                )
        return scatters

    def draw(self) -> None:
        self.ax.clear()
        self._init_axes()
        self.ax.set_title(self.scene_state.title, color="#eef6ff", fontsize=14, weight="bold")
        self._draw_base_station()
        self._draw_users()
        self._draw_beams()
        self._draw_scatter_paths()
        self._draw_overlay()
        self.fig.canvas.draw_idle()

    def _draw_base_station(self) -> None:
        # Outer glow + BS ring to match the older, more explicit BS language.
        self.ax.add_patch(Circle((0.0, 0.0), 0.95, facecolor="#0b1522", edgecolor="#2a3f5f", linewidth=1.0, zorder=5))
        self.ax.add_patch(Circle((0.0, 0.0), 0.72, facecolor="#f59e0b", edgecolor="#92400e", linewidth=2.2, zorder=7))
        self.ax.add_patch(Circle((0.0, 0.0), 0.28, facecolor="#0b1522", edgecolor="#f8fafc", linewidth=1.2, zorder=8))
        self.ax.text(
            0.0,
            -1.2,
            "Base Station",
            color="#f8fafc",
            fontsize=11,
            ha="center",
            va="top",
            zorder=9,
            fontweight="bold",
        )
        if self.scene_state.is_xl_mimo:
            elements = min(self.scene_state.num_tx_antennas, 96)
            ring_radius = 0.82
            for idx in range(elements):
                angle = 2.0 * math.pi * idx / max(elements, 1)
                x = ring_radius * math.cos(angle)
                y = ring_radius * math.sin(angle)
                self.ax.add_patch(Circle((x, y), 0.05, facecolor="#c084fc", edgecolor="none", alpha=0.85, zorder=6))
        else:
            for row in range(4):
                for col in range(4):
                    x = -0.36 + col * 0.24
                    y = -0.36 + row * 0.24
                    self.ax.add_patch(Circle((x, y), 0.04, facecolor="#fca5a5", edgecolor="none", alpha=0.8, zorder=6))

    def _draw_users(self) -> None:
        for idx, (x, y) in enumerate(self.scene_state.user_positions):
            color = self.USER_COLORS[idx % len(self.USER_COLORS)]
            self.ax.add_patch(Circle((x, y), 0.52, facecolor="#0b1522", edgecolor="#2a3f5f", linewidth=1.0, zorder=6))
            self.ax.add_patch(Circle((x, y), 0.38, facecolor=color, edgecolor="#06111f", linewidth=1.8, zorder=8))
            self.ax.add_patch(Circle((x, y), 0.10, facecolor="#e2e8f0", edgecolor="none", zorder=9, alpha=0.9))
            self.ax.text(x, y + 0.66, f"User {idx + 1}", color=color, fontsize=10, ha="center", va="bottom", zorder=8, fontweight="bold")

    def _draw_beams(self) -> None:
        self._animated_artists = []
        for segment_idx, segment in enumerate(self.scene_state.beams):
            beam = FancyArrowPatch(
                segment.start,
                segment.end,
                arrowstyle="-|>",
                mutation_scale=14 if segment.is_los else 9,
                linewidth=segment.width,
                linestyle="-" if segment.is_los else "--",
                color=segment.color,
                alpha=segment.alpha,
                zorder=5 if segment.is_los else 4,
            )
            self.ax.add_patch(beam)
            self._animated_artists.append(beam)
            if segment_idx < max(1, min(2, self.scene_state.num_users)) and segment.is_los:
                mid_x = 0.58 * segment.end[0]
                mid_y = 0.58 * segment.end[1]
                self.ax.text(
                    mid_x,
                    mid_y,
                    "LoS Beam",
                    color=segment.color,
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "#08111d", "edgecolor": "none", "alpha": 0.55, "boxstyle": "round,pad=0.2"},
                    zorder=9,
                )

    def _draw_scatter_paths(self) -> None:
        for scatter in self.scene_state.scatter_paths:
            ux, uy = self.scene_state.user_positions[scatter.user_index]
            sx, sy = scatter.via
            self.ax.add_patch(Circle((sx, sy), 0.13, facecolor="#9ca3af", edgecolor="none", alpha=0.55, zorder=3))
            self.ax.plot((0.0, sx), (0.0, sy), linestyle="--", linewidth=1.1, color=scatter.color, alpha=scatter.alpha, zorder=2)
            self.ax.plot((sx, ux), (sy, uy), linestyle="--", linewidth=1.1, color=scatter.color, alpha=scatter.alpha, zorder=2)

    def _draw_overlay(self) -> None:
        band = self.scene_state.metrics.get("band")
        snr = self.scene_state.metrics.get("snr_db")
        nt = self.scene_state.metrics.get("num_tx_antennas")
        flags = []
        if self.scene_state.is_sub_thz:
            flags.append("LoS-Dominant")
            flags.append("Pencil Beam")
        overlay = [
            f"Band: {band}",
            f"SNR: {snr:.1f} dB" if isinstance(snr, (float, int)) else f"SNR: {snr}",
            f"Nt: {nt}",
            f"Flags: {', '.join(flags) if flags else 'Baseline Multipath'}",
        ]
        self.ax.text(
            0.02,
            0.98,
            "\n".join(overlay),
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#dbeafe",
            bbox={"facecolor": "#0a1626", "edgecolor": "#17304b", "boxstyle": "round,pad=0.45", "alpha": 0.9},
            zorder=10,
        )

    def animate(self, frames: int = 180, interval: int = 70) -> FuncAnimation:
        def update(frame: int):
            pulse = 0.82 + 0.18 * math.sin(frame * 0.22)
            for idx, artist in enumerate(self._animated_artists):
                if idx >= len(self.scene_state.beams):
                    continue
                segment = self.scene_state.beams[idx]
                alpha = segment.alpha
                if segment.is_los:
                    artist.set_alpha(min(1.0, alpha * (1.02 if self.scene_state.is_sub_thz else pulse)))
                    artist.set_linewidth(segment.width * (1.0 + 0.06 * math.sin(frame * 0.16 + idx)))
                else:
                    artist.set_alpha(alpha * (0.85 + 0.15 * math.cos(frame * 0.12 + idx)))
            return self._animated_artists

        self.animation = FuncAnimation(self.fig, update, frames=frames, interval=interval, blit=False)
        return self.animation

    def show(self) -> None:
        plt.show()


class ChannelVisualizerWidget:
    """
    Thin adapter used by the node workbench to consume TensorPayload directly.
    """

    def __init__(self, parent=None):
        del parent
        self.visualizer = ChannelVisualizer()

    def update(self, payload: TensorPayload) -> None:
        self.visualizer.update_from_payload(payload)
        self.visualizer.draw()
