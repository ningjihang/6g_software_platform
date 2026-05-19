from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulink_viewer import SimulationParams


@dataclass
class Ray:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    delay: float = 0.0
    power: float = 1.0
    reflection: int = 0
    ray_type: str = "direct"


class ChannelAnalyzer:
    def __init__(self, params: "SimulationParams") -> None:
        self.params = params
        self.channel_type = params.channel_type
        self.num_rays = 4
    
    def generate_rays(self, bs_x: float, bs_y: float, user_x: float, user_y: float, user_id: int) -> list[Ray]:
        rays = []
        
        direct_ray = Ray(
            start_x=bs_x,
            start_y=bs_y,
            end_x=user_x,
            end_y=user_y,
            delay=0.0,
            power=1.0,
            reflection=0,
            ray_type="direct"
        )
        rays.append(direct_ray)
        
        for i in range(self.num_rays - 1):
            offset_angle = (i + 1) * 15 * (1 if user_id % 2 == 0 else -1)
            offset_distance = random.uniform(30, 80)
            
            angle = math.atan2(user_y - bs_y, user_x - bs_x)
            perp_angle = angle + math.pi / 2
            
            mid_x = (bs_x + user_x) / 2 + offset_distance * math.cos(perp_angle)
            mid_y = (bs_y + user_y) / 2 + offset_distance * math.sin(perp_angle)
            
            reflected_ray = Ray(
                start_x=bs_x,
                start_y=bs_y,
                end_x=user_x,
                end_y=user_y,
                delay=0.5 + i * 0.2,
                power=0.6 - i * 0.15,
                reflection=i + 1,
                ray_type="reflected" if i < 2 else "diffracted"
            )
            rays.append(reflected_ray)
        
        return rays
    
    def get_ray_color(self, ray: Ray) -> str:
        if ray.ray_type == "direct":
            return "#5bb6ff"
        elif ray.ray_type == "reflected":
            return "#34d399"
        else:
            return "#f472b6"
    
    def draw_power_distribution(self, canvas: tk.Canvas, center_x: int, center_y: int, radius: int) -> None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            power = random.uniform(0.3, 1.0)
            arc_length = power * radius * 0.6
            
            x1 = center_x + radius * 0.3 * math.cos(angle_rad)
            y1 = center_y + radius * 0.3 * math.sin(angle_rad)
            x2 = center_x + arc_length * math.cos(angle_rad)
            y2 = center_y + arc_length * math.sin(angle_rad)
            
            color = self._get_power_color(power)
            canvas.create_line(x1, y1, x2, y2, fill=color, width=3, smooth=True)
            
            for i in range(3):
                offset = random.uniform(-15, 15)
                offset_angle = angle_rad + math.radians(offset)
                small_power = power * random.uniform(0.4, 0.7)
                small_length = small_power * radius * 0.4
                
                sx = center_x + radius * 0.2 * math.cos(offset_angle)
                sy = center_y + radius * 0.2 * math.sin(offset_angle)
                ex = sx + small_length * math.cos(offset_angle)
                ey = sy + small_length * math.sin(offset_angle)
                
                canvas.create_line(sx, sy, ex, ey, fill="#4b617a", width=1)
        
        self._draw_scattering_points(canvas, center_x, center_y, radius)
    
    def _get_power_color(self, power: float) -> str:
        if power > 0.7:
            return "#5bb6ff"
        elif power > 0.4:
            return "#34d399"
        else:
            return "#f59e0b"
    
    def _draw_scattering_points(self, canvas: tk.Canvas, center_x: int, center_y: int, radius: int) -> None:
        num_points = 12
        
        for _ in range(num_points):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(radius * 0.4, radius * 0.9)
            
            x = center_x + dist * math.cos(angle)
            y = center_y + dist * math.sin(angle)
            intensity = random.uniform(0.3, 1.0)
            
            size = 3 + intensity * 4
            r = int(91 * intensity + 45 * (1 - intensity))
            g = int(182 * intensity + 63 * (1 - intensity))
            b = int(255 * intensity + 95 * (1 - intensity))
            color = f"#{r:02x}{g:02x}{b:02x}"
            
            canvas.create_oval(
                x - size / 2, y - size / 2,
                x + size / 2, y + size / 2,
                fill=color, outline=""
            )
    
    def calculate_snr_distribution(self) -> list[tuple[float, float]]:
        dist_points = []
        for i in range(10):
            distance = 100 + i * 50
            path_loss = 128.1 + 37.6 * math.log10(distance / 1000)
            snr = self.params.snr_range[1] - path_loss * 0.1
            dist_points.append((distance, max(snr, self.params.snr_range[0])))
        return dist_points
    
    def get_channel_info(self) -> dict[str, float]:
        return {
            "mean_delay": 0.5,
            "rms_delay_spread": 0.15,
            "doppler_spread": 10.0,
            "coherence_bandwidth": 1e6,
        }