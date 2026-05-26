"""
Step 3: 铅笔波束可视化器 (Tkinter 兼容版本)
"""
import math
import random
import tkinter as tk
from typing import List, Tuple, Optional
import numpy as np


class PencilBeamVisualizer:
    """
    6G MIMO 信道可视化器
    - 支持常规多径散射
    - 支持太赫兹铅笔波束
    - 支持 XL-MIMO 模式
    """
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.width = 800
        self.height = 600
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.base_radius = 30
        self.user_radius = 20
        self.user_dist = 180
        
        self.is_subthz = False
        self.is_xl_mimo = False
        self.num_users = 2
        self.num_tx_antennas = 16
        
        self.beams = []
        self.scatter_paths = []
        self.user_positions = []
        self.animated = False
        self._frame_count = 0
    
    def set_parameters(self, is_subthz: bool, is_xl_mimo: bool,
                       num_users: int, num_tx_antennas: int):
        """设置可视化参数"""
        self.is_subthz = is_subthz
        self.is_xl_mimo = is_xl_mimo
        self.num_users = num_users
        self.num_tx_antennas = num_tx_antennas
    
    def calculate_user_positions(self) -> List[Tuple[float, float]]:
        """计算用户位置（圆周均匀分布）"""
        positions = []
        for i in range(self.num_users):
            angle = 2 * math.pi * i / self.num_users
            x = self.center_x + self.user_dist * math.cos(angle)
            y = self.center_y + self.user_dist * math.sin(angle)
            positions.append((x, y))
        return positions
    
    def draw_base_station(self):
        """绘制基站与天线阵列"""
        # 基站主体
        self.canvas.create_oval(
            self.center_x - self.base_radius,
            self.center_y - self.base_radius,
            self.center_x + self.base_radius,
            self.center_y + self.base_radius,
            fill="#ef4444",
            outline="#b91c1c",
            width=2
        )
        
        # 标签
        self.canvas.create_text(
            self.center_x, self.center_y - self.base_radius - 15,
            text="Base Station",
            fill="#e5e7eb",
            font=("Arial", 10, "bold")
        )
        
        # 绘制天线阵列
        if self.is_xl_mimo:
            self._draw_xl_array()
        else:
            self._draw_normal_array()
    
    def _draw_normal_array(self):
        """常规天线阵列（4x4）"""
        for i in range(4):
            for j in range(4):
                x = self.center_x - 12 + i * 8
                y = self.center_y - 12 + j * 8
                self.canvas.create_oval(x-2, y-2, x+2, y+2,
                                       fill="#7f1d1d", outline="#450a0a")
    
    def _draw_xl_array(self):
        """XL-MIMO 密集天线阵列（圆形排列）"""
        num_ant = min(self.num_tx_antennas, 64)
        radius = 22
        for i in range(num_ant):
            angle = 2 * math.pi * i / num_ant
            x = self.center_x + radius * math.cos(angle)
            y = self.center_y + radius * math.sin(angle)
            self.canvas.create_oval(x-2, y-2, x+2, y+2,
                                   fill="#a855f7", outline="#581c87")
    
    def draw_users(self):
        """绘制用户设备"""
        colors = ["#60a5fa", "#4ade80", "#f472b6", "#fbbf24"]
        self.user_positions = self.calculate_user_positions()
        
        for idx, (x, y) in enumerate(self.user_positions):
            color = colors[idx % len(colors)]
            self.canvas.create_oval(
                x - self.user_radius, y - self.user_radius,
                x + self.user_radius, y + self.user_radius,
                fill=color,
                outline=self._darken_color(color),
                width=2
            )
            self.canvas.create_text(
                x, y - self.user_radius - 12,
                text=f"User {idx + 1}",
                fill=color,
                font=("Arial", 10, "bold")
            )
    
    def draw_pencil_beams(self):
        """绘制铅笔波束（太赫兹模式）"""
        self.beams = []
        colors = ["#38bdf8", "#4ade80", "#f472b6", "#fbbf24"]
        
        for idx, (user_x, user_y) in enumerate(self.user_positions):
            color = colors[idx % len(colors)]
            # 主波束
            self._draw_single_pencil_beam(self.center_x, self.center_y, user_x, user_y, color)
            # 旁瓣
            if idx % 2 == 0:
                angle = math.atan2(user_y - self.center_y, user_x - self.center_x)
                for offset in [-0.03, 0.03]:
                    sideline_end_x = self.center_x + (self.user_dist + 20) * math.cos(angle + offset)
                    sideline_end_y = self.center_y + (self.user_dist + 20) * math.sin(angle + offset)
                    self._draw_single_pencil_beam(
                        self.center_x, self.center_y, sideline_end_x, sideline_end_y,
                        color, alpha=0.3
                    )
    
    def _draw_single_pencil_beam(self, x1, y1, x2, y2, color, alpha=1.0):
        """绘制单条铅笔波束"""
        line = self.canvas.create_line(x1, y1, x2, y2,
                                       fill=color, width=3 if alpha == 1 else 1,
                                       arrow=tk.LAST, arrowshape=(8, 10, 4))
        self.beams.append(line)
    
    def draw_scatter_paths(self):
        """绘制散射多径（常规模式）"""
        self.scatter_paths = []
        scatter_points = []
        colors = ["#60a5fa", "#4ade80", "#f472b6", "#fbbf24"]
        
        # 生成散射点
        for _ in range(8):
            cx = random.randint(self.center_x - 120, self.center_x + 120)
            cy = random.randint(self.center_y - 100, self.center_y + 100)
            scatter_points.append((cx, cy))
        
        for idx, (user_x, user_y) in enumerate(self.user_positions):
            # 每个用户 3 条散射路径
            selected_points = scatter_points[idx*2 : idx*2+3]
            color = colors[idx % len(colors)]
            for (sx, sy) in selected_points:
                self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4,
                                       fill="#9ca3af", outline="#6b7280")
                # 基站 -> 散射点
                path1 = self.canvas.create_line(
                    self.center_x, self.center_y, sx, sy,
                    dash=(4, 4), fill=color, width=1.5, state=tk.DISABLED if self.is_subthz else tk.NORMAL
                )
                self.scatter_paths.append(path1)
                # 散射点 -> 用户
                path2 = self.canvas.create_line(
                    sx, sy, user_x, user_y,
                    dash=(4, 4), fill=color, width=1.5, state=tk.DISABLED if self.is_subthz else tk.NORMAL
                )
                self.scatter_paths.append(path2)
    
    def draw(self):
        """绘制完整场景"""
        self.canvas.delete("all")
        
        # 绘制网格背景
        for x in range(0, self.width, 40):
            self.canvas.create_line(x, 0, x, self.height,
                                   fill="#1f2937", width=1)
        for y in range(0, self.height, 40):
            self.canvas.create_line(0, y, self.width, y,
                                   fill="#1f2937", width=1)
        
        # 绘制组件
        self.draw_base_station()
        self.draw_users()
        
        if self.is_subthz or self.is_xl_mimo:
            self.draw_pencil_beams()
        else:
            self.draw_pencil_beams()
            self.draw_scatter_paths()
        
        # 模式标签
        mode_text = "6G MIMO Channel"
        if self.is_subthz:
            mode_text += " (Sub-THz: Pencil Beams)"
        elif self.is_xl_mimo:
            mode_text += " (XL-MIMO Mode)"
        
        self.canvas.create_text(
            self.width // 2, 30,
            text=mode_text,
            fill="#e5e7eb",
            font=("Arial", 14, "bold")
        )
    
    def animate(self):
        """动画效果 - 让波束闪烁"""
        self._frame_count += 1
        if not self.beams:
            return
        
        for idx, beam in enumerate(self.beams):
            # 相位偏移
            phase = self._frame_count * 0.1 + idx * 0.5
            if self.is_subthz:
                # 铅笔波束闪烁效果
                brightness = 0.6 + 0.4 * math.sin(phase)
                # 这里只调整一下大小
            else:
                # 常规波束的轻微波动
                pass
        
        self.canvas.after(50, self.animate)
    
    @staticmethod
    def _darken_color(hex_color: str) -> str:
        """简单的颜色变暗函数"""
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        # 简单处理
        return "#4b5563"
