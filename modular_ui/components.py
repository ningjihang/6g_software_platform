from __future__ import annotations

import tkinter as tk


BS_NODE_COLOR = "#f59e0b"
USER_NODE_COLORS = ["#5bb6ff", "#34d399", "#f472b6", "#a78bfa", "#fb923c", "#6ee7b7", "#fca5a5", "#c4b5fd"]


class NodeComponent:
    @staticmethod
    def draw(
        canvas: tk.Canvas,
        x: float,
        y: float,
        label: str,
        color: str,
        status_text: str = "",
        *,
        is_source: bool = False,
        is_selected: bool = False,
        size_factor: float = 1.0,
    ) -> None:
        base_size = 50 if is_source else 40
        size = base_size * size_factor
        glow_size = (base_size + 16 if is_source else base_size + 10) * size_factor
        
        if is_selected:
            canvas.create_oval(
                x - glow_size / 2, y - glow_size / 2,
                x + glow_size / 2, y + glow_size / 2,
                fill="", outline=color, width=2, dash=(4, 2)
            )
        
        canvas.create_oval(
            x - size / 2 - 6, y - size / 2 - 6,
            x + size / 2 + 6, y + size / 2 + 6,
            fill="#0f1722", outline="#2a3f5f", width=1
        )
        
        if is_source:
            gradient_start = color
            gradient_end = "#1f2937"
            for i in range(8):
                angle = (360 * i) / 8
                inner_radius = size / 2 - 4
                outer_radius = size / 2
                canvas.create_arc(
                    x - outer_radius, y - outer_radius,
                    x + outer_radius, y + outer_radius,
                    start=angle - 20, extent=40,
                    fill=gradient_start if i % 2 == 0 else gradient_end,
                    outline="", style="pieslice"
                )
            
            canvas.create_oval(
                x - size / 3, y - size / 3,
                x + size / 3, y + size / 3,
                fill="#0f1722", outline=color, width=2
            )
        else:
            canvas.create_oval(
                x - size / 2, y - size / 2,
                x + size / 2, y + size / 2,
                fill="#1a2636", outline=color, width=2
            )
            
            dot_size = 8 * size_factor
            canvas.create_oval(
                x - dot_size / 2, y - dot_size / 2,
                x + dot_size / 2, y + dot_size / 2,
                fill=color, outline=""
            )
            
            pulse_size = 12 * size_factor
            canvas.create_oval(
                x - pulse_size / 2, y - pulse_size / 2,
                x + pulse_size / 2, y + pulse_size / 2,
                fill=color, outline="", stipple="gray50"
            )
        
        label_y_offset = size / 2 + 16
        canvas.create_text(x, y + label_y_offset, text=label, fill="#eef5ff", font=("Segoe UI", 10, "bold"))
        
        if status_text:
            status_y_offset = size / 2 + 32
            canvas.create_text(x, y + status_y_offset, text=status_text, fill="#b9cbe0", font=("Segoe UI", 8))


class ChannelLink:
    @staticmethod
    def draw(
        canvas: tk.Canvas,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#5bb6ff",
        width: int = 2,
        opacity: float = 1.0,
        show_arrow: bool = True,
    ) -> None:
        draw_arrow_line(canvas, x1, y1, x2, y2, color, width, opacity)


def draw_arrow_line(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#5bb6ff",
    width: int = 2,
    opacity: float = 1.0,
) -> None:
    if opacity < 1.0:
        alpha = int(opacity * 255)
        color = _adjust_color_opacity(color, alpha)
    
    canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
    
    angle = _calculate_angle(x1, y1, x2, y2)
    arrow_size = 12
    
    arrow_x = x2 - arrow_size * _cos(angle)
    arrow_y = y2 - arrow_size * _sin(angle)
    
    canvas.create_line(
        x2, y2,
        arrow_x - arrow_size * _sin(angle - 30),
        arrow_y + arrow_size * _cos(angle - 30),
        fill=color, width=width
    )
    canvas.create_line(
        x2, y2,
        arrow_x + arrow_size * _sin(angle + 30),
        arrow_y - arrow_size * _cos(angle + 30),
        fill=color, width=width
    )


def draw_grid_background(canvas: tk.Canvas, width: int, height: int) -> None:
    grid_size = 50
    
    for x in range(0, width, grid_size):
        canvas.create_line(x, 0, x, height, fill="#16212f", width=1)
    
    for y in range(0, height, grid_size):
        canvas.create_line(0, y, width, y, fill="#16212f", width=1)
    
    center_x = width // 2
    center_y = height // 2
    
    canvas.create_line(center_x, 0, center_x, height, fill="#2a3f5f", width=1, dash=(4, 4))
    canvas.create_line(0, center_y, width, center_y, fill="#2a3f5f", width=1, dash=(4, 4))


def _adjust_color_opacity(color: str, alpha: int) -> str:
    if color.startswith("#"):
        color = color[1:]
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"#{r:02x}{g:02x}{b:02x}"


def _calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    import math
    return math.atan2(y2 - y1, x2 - x1)


def _cos(angle: float) -> float:
    import math
    return math.cos(angle)


def _sin(angle: float) -> float:
    import math
    return math.sin(angle)