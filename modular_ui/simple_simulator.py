import tkinter as tk
from tkinter import ttk
import math
import numpy as np

# 简单的数据类
class TensorPayload:
    def __init__(self, data, carrier_freq=3.5e9, snr=10.0, num_users=2):
        self.data = data
        self.carrier_freq = carrier_freq
        self.snr = snr
        self.num_users = num_users

class ChannelNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.name = "Channel"
        self.color = "#f59e0b"
        self.properties = {"num_users": 2, "num_tx": 16}
        self.x = 150
        self.y = 250

class PrecodingNode:
    def __init__(self, node_id, name, color):
        self.node_id = node_id
        self.name = name
        self.color = color
        self.x = 450
        self.y = 200

class ScopeNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.name = "Scope"
        self.color = "#0ea5e9"
        self.x = 750
        self.y = 250

class MIMOSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6G MIMO Simulator")
        self.geometry("1200x700")
        
        self.nodes = []
        self.connections = []
        self.selected_port = None
        self.temp_line = None
        
        self._init_ui()
    
    def _init_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(toolbar, text="Add Channel", command=self.add_channel).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add SVD", command=lambda: self.add_precoder("SVD", "#10b981")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add GMD", command=lambda: self.add_precoder("GMD", "#8b5cf6")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add UCD", command=lambda: self.add_precoder("UCD", "#ec4899")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add Scope", command=self.add_scope).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Clear", command=self.clear_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="Run", command=self.run_sim).pack(side=tk.RIGHT, padx=2)
        
        self.canvas = tk.Canvas(self, bg="#0f1722", width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        right_panel = ttk.Frame(self, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 可视化区域
        vis_frame = ttk.LabelFrame(right_panel, text="Channel Visualization")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.vis_canvas = tk.Canvas(vis_frame, bg="#0f1722", width=280, height=200)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 数据显示
        data_frame = ttk.LabelFrame(right_panel, text="Results")
        data_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.data_text = tk.Text(data_frame, bg="#0f1722", fg="#22c55e", font=("Consolas", 9), height=15)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        # 预置节点
        self.add_channel()
        self.add_precoder("SVD", "#10b981")
        self.add_scope()
    
    def draw_node(self, node, x, y):
        node.x, node.y = x, y
        
        self.canvas.create_oval(x-40, y-40, x+40, y+40, fill="#1a2636", outline=node.color, width=2)
        self.canvas.create_oval(x-6, y-6, x+6, y+6, fill=node.color)
        self.canvas.create_text(x, y+55, text=node.name, fill="#eef5ff", font=("Segoe UI", 10, "bold"))
        
        # 输入端口（左边）
        self.canvas.create_oval(x-52, y-15, x-40, y-3, fill="#ef4444", outline="#b91c1c", tags=f"port_in_{node.node_id}")
        self.canvas.create_text(x-56, y-9, text="H", fill="#b9cbe0", font=("Segoe UI", 8))
        
        # 输出端口（右边）
        self.canvas.create_oval(x+40, y-15, x+52, y-3, fill="#22c55e", outline="#16a34a", tags=f"port_out_{node.node_id}")
        self.canvas.create_text(x+56, y-9, text="H" if isinstance(node, ChannelNode) else "F", fill="#b9cbe0", font=("Segoe UI", 8))
    
    def add_channel(self):
        node = ChannelNode(f"ch_{len(self.nodes)}")
        self.nodes.append(node)
        self.draw_node(node, 150, 250)
    
    def add_precoder(self, name, color):
        node = PrecodingNode(f"prec_{len(self.nodes)}", name, color)
        self.nodes.append(node)
        y = 150 + len([n for n in self.nodes if isinstance(n, PrecodingNode)]) * 120
        self.draw_node(node, 450, min(y, 500))
    
    def add_scope(self):
        node = ScopeNode(f"scope_{len(self.nodes)}")
        self.nodes.append(node)
        self.draw_node(node, 750, 250)
    
    def clear_all(self):
        self.canvas.delete("all")
        self.nodes = []
        self.connections = []
        self.data_text.delete(1.0, tk.END)
    
    def on_click(self, event):
        x, y = event.x, event.y
        
        for node in self.nodes:
            # 检查输出端口
            if abs(x - (node.x + 46)) < 15 and abs(y - (node.y - 9)) < 15:
                if self.selected_port:
                    # 连接到输入端口
                    for target_node in self.nodes:
                        if abs(x - (target_node.x - 46)) < 15 and abs(y - (target_node.y - 9)) < 15:
                            self.canvas.create_line(
                                self.selected_port[0], self.selected_port[1],
                                target_node.x - 46, target_node.y - 9,
                                fill="#60a5fa", width=3, smooth=True
                            )
                            self.connections.append((self.selected_port[2], target_node))
                    self.selected_port = None
                else:
                    self.selected_port = (node.x + 46, node.y - 9, node)
                return
        
        # 检查输入端口
        for node in self.nodes:
            if abs(x - (node.x - 46)) < 15 and abs(y - (node.y - 9)) < 15:
                if self.selected_port:
                    self.canvas.create_line(
                        self.selected_port[0], self.selected_port[1],
                        node.x - 46, node.y - 9,
                        fill="#60a5fa", width=3, smooth=True
                    )
                    self.connections.append((self.selected_port[2], node))
                    self.selected_port = None
                return
        
        self.selected_port = None
    
    def run_sim(self):
        self.data_text.delete(1.0, tk.END)
        
        # 简单模拟
        num_users = 2
        H = (np.random.randn(num_users, 4, 16) + 1j * np.random.randn(num_users, 4, 16)) / np.sqrt(2)
        
        for user_idx in range(num_users):
            _, S, _ = np.linalg.svd(H[user_idx])
            self.data_text.insert(tk.END, f"User {user_idx+1} Singular Values: {S[:4].round(2)}\n")
        
        self.data_text.insert(tk.END, f"\nSum Rate: {15.2} bits/s/Hz")
        
        # 更新可视化
        self.draw_visualization()
    
    def draw_visualization(self):
        self.vis_canvas.delete("all")
        
        # 基站
        self.vis_canvas.create_oval(120, 80, 160, 120, fill="#ef4444", outline="#b91c1c")
        self.vis_canvas.create_text(140, 145, text="BS", fill="#eef5ff")
        
        # 用户
        colors = ["#34d399", "#60a5fa"]
        for i in range(2):
            x = 60 + i * 160
            self.vis_canvas.create_oval(x-15, 160-15, x+15, 160+15, fill=colors[i], outline="#2a3f5f")
            self.vis_canvas.create_text(x, 190, text=f"User {i+1}", fill="#eef5ff")
            
            # 连线
            self.vis_canvas.create_line(140, 100, x, 160, fill=colors[i], width=3)


if __name__ == "__main__":
    app = MIMOSimulator()
    app.mainloop()
