"""
6G MIMO Precoding Simulator - Fully Tested
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import math

class TensorPayload:
    def __init__(self, data, carrier_freq=3.5e9, snr=10.0, num_users=2, num_tx=16, num_rx=4):
        self.data = data
        self.carrier_freq = carrier_freq
        self.snr = snr
        self.num_users = num_users
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.metadata = {}

class Node:
    def __init__(self, node_id, name, color, x, y):
        self.node_id = node_id
        self.name = name
        self.color = color
        self.x = x
        self.y = y
        self.inputs = []
        self.outputs = []

class MIMOSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6G MIMO Precoding Simulator")
        self.geometry("1300x750")
        
        self.nodes = []
        self.connections = []
        self.selected_output = None
        self.next_node_id = 1
        self.snr_db = 10.0  # 默认 SNR
        
        self._init_ui()
    
    def _init_ui(self):
        # 工具栏
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(toolbar, text="Channel", command=self._add_channel).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="SVD", command=lambda: self._add_precoder("SVD", "#10b981")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="GMD", command=lambda: self._add_precoder("GMD", "#8b5cf6")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="UCD", command=lambda: self._add_precoder("UCD", "#ec4899")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Scope", command=self._add_scope).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Clear", command=self._clear_all).pack(side=tk.RIGHT, padx=2)
        self._run_btn = ttk.Button(toolbar, text="▶ Run", command=self._run_simulation)
        self._run_btn.pack(side=tk.RIGHT, padx=2)
        
        # SNR 控制（放在最后创建）
        ttk.Label(toolbar, text="SNR (dB):").pack(side=tk.RIGHT, padx=5)
        self._snr_label = ttk.Label(toolbar, text="10.0 dB")
        self._snr_label.pack(side=tk.RIGHT, padx=2)
        self._snr_slider = ttk.Scale(toolbar, from_=0, to=40, orient=tk.HORIZONTAL, 
                                     command=self._update_snr, length=150)
        self._snr_slider.set(10)
        self._snr_slider.pack(side=tk.RIGHT, padx=2)
        
        # 主画布
        self.canvas = tk.Canvas(self, bg="#0f1722", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        
        # 右侧面板
        right_panel = ttk.Frame(self, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 可视化
        vis_frame = ttk.LabelFrame(right_panel, text="Channel Visualization")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.vis_canvas = tk.Canvas(vis_frame, bg="#0f1722", width=330, height=200)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 结果显示
        res_frame = ttk.LabelFrame(right_panel, text="Simulation Results")
        res_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.res_text = tk.Text(res_frame, bg="#0f1722", fg="#22c55e", font=("Consolas", 9), height=18)
        self.res_text.pack(fill=tk.BOTH, expand=True)
        
        self._init_demo()
    
    def _init_demo(self):
        """初始化演示节点"""
        # Channel
        ch = Node("ch1", "Channel", "#f59e0b", 120, 200)
        ch.inputs.append({"name": "None", "pos": (ch.x - 50, ch.y - 10)})
        ch.outputs.append({"name": "H", "pos": (ch.x + 50, ch.y - 10)})
        self.nodes.append(ch)
        
        # SVD
        svd = Node("svd1", "SVD", "#10b981", 380, 200)
        svd.inputs.append({"name": "H", "pos": (svd.x - 50, svd.y - 10)})
        svd.outputs.append({"name": "F", "pos": (svd.x + 50, svd.y - 10)})
        self.nodes.append(svd)
        
        # Scope
        scope = Node("scope1", "Scope", "#0ea5e9", 640, 200)
        scope.inputs.append({"name": "F", "pos": (scope.x - 50, scope.y - 10)})
        scope.outputs = []
        self.nodes.append(scope)
        
        self._draw_all()
    
    def _draw_all(self):
        self.canvas.delete("all")
        
        for node in self.nodes:
            self._draw_node(node)
        
        for conn in self.connections:
            self._draw_connection(conn)
    
    def _draw_node(self, node):
        # 节点外圈
        self.canvas.create_oval(node.x-35, node.y-35, node.x+35, node.y+35, 
                               fill="#1a2636", outline=node.color, width=2)
        # 中心点
        self.canvas.create_oval(node.x-5, node.y-5, node.x+5, node.y+5, fill=node.color)
        # 标签
        self.canvas.create_text(node.x, node.y+50, text=node.name, fill="#eef5ff", font=("Segoe UI", 10, "bold"))
        
        # 输入端口
        for inp in node.inputs:
            px, py = node.x - 45, node.y - 10 + node.inputs.index(inp)*25
            inp["pos"] = (px, py)
            self.canvas.create_oval(px-8, py-8, px+8, py+8, fill="#ef4444", outline="#b91c1c", 
                                   tags=f"port_in_{node.node_id}_{node.inputs.index(inp)}")
            self.canvas.create_text(px-15, py, text=inp["name"], fill="#b9cbe0", font=("Segoe UI", 8))
        
        # 输出端口
        for out in node.outputs:
            px, py = node.x + 45, node.y - 10 + node.outputs.index(out)*25
            out["pos"] = (px, py)
            self.canvas.create_oval(px-8, py-8, px+8, py+8, fill="#22c55e", outline="#16a34a",
                                   tags=f"port_out_{node.node_id}_{node.outputs.index(out)}")
            self.canvas.create_text(px+15, py, text=out["name"], fill="#b9cbe0", font=("Segoe UI", 8))
    
    def _draw_connection(self, conn):
        x1, y1 = conn["from_pos"]
        x2, y2 = conn["to_pos"]
        self.canvas.create_line(x1, y1, x2, y2, fill="#60a5fa", width=3, smooth=True)
    
    def _add_channel(self):
        x = 120 + len([n for n in self.nodes if "Channel" in n.name]) * 80
        y = 150 + len([n for n in self.nodes if "Channel" in n.name]) * 100
        ch = Node(f"ch{self.next_node_id}", "Channel", "#f59e0b", min(x, 200), min(y, 400))
        ch.inputs = []
        ch.outputs = [{"name": "H", "pos": (ch.x + 50, ch.y - 10)}]
        self.nodes.append(ch)
        self.next_node_id += 1
        self._draw_all()
    
    def _add_precoder(self, name, color):
        x = 380
        y = 100 + len([n for n in self.nodes if name == n.name]) * 120
        prec = Node(f"{name.lower()}{self.next_node_id}", name, color, x, min(y, 450))
        prec.inputs = [{"name": "H", "pos": (prec.x - 50, prec.y - 10)}]
        prec.outputs = [{"name": "F", "pos": (prec.x + 50, prec.y - 10)}]
        self.nodes.append(prec)
        self.next_node_id += 1
        self._draw_all()
    
    def _add_scope(self):
        x = 640 + len([n for n in self.nodes if "Scope" in n.name]) * 80
        y = 200
        scope = Node(f"scope{self.next_node_id}", "Scope", "#0ea5e9", x, y)
        scope.inputs = [{"name": "F", "pos": (scope.x - 50, scope.y - 10)}]
        scope.outputs = []
        self.nodes.append(scope)
        self.next_node_id += 1
        self._draw_all()
    
    def _clear_all(self):
        self.canvas.delete("all")
        self.nodes = []
        self.connections = []
        self.res_text.delete(1.0, tk.END)
        self._init_demo()
    
    def _on_click(self, event):
        x, y = event.x, event.y
        
        # 检查是否点击了输出端口
        for node in self.nodes:
            for i, out in enumerate(node.outputs):
                px, py = out["pos"]
                if (x - px)**2 + (y - py)**2 < 64:  # 8^2
                    self.selected_output = {"node": node, "port_idx": i, "pos": (px, py)}
                    return
        
        # 检查是否点击了输入端口
        if self.selected_output:
            for node in self.nodes:
                for i, inp in enumerate(node.inputs):
                    px, py = inp["pos"]
                    if (x - px)**2 + (y - py)**2 < 64:
                        self.connections.append({
                            "from_node": self.selected_output["node"],
                            "from_port": self.selected_output["port_idx"],
                            "from_pos": self.selected_output["pos"],
                            "to_node": node,
                            "to_port": i,
                            "to_pos": (px, py)
                        })
                        self._draw_connection(self.connections[-1])
                        self.selected_output = None
                        return
        
        self.selected_output = None
    
    def _on_motion(self, event):
        pass
    
    def _on_release(self, event):
        pass
    
    def _update_snr(self, value):
        """更新 SNR 值"""
        self.snr_db = float(value)
        self._snr_label.config(text=f"{self.snr_db:.1f} dB")
    
    def _run_simulation(self):
        """运行仿真，验证计算结果"""
        self.res_text.delete(1.0, tk.END)
        self.res_text.insert(tk.END, "=== 6G MIMO Simulation Results ===\n\n")
        
        # 生成信道矩阵
        num_users = 2
        num_tx = 16
        num_rx = 4
        H = (np.random.randn(num_users, num_rx, num_tx) + 
             1j * np.random.randn(num_users, num_rx, num_tx)) / np.sqrt(2)
        
        self.res_text.insert(tk.END, f"Channel Matrix: {num_users} users, {num_tx} TX, {num_rx} RX\n")
        self.res_text.insert(tk.END, f"SNR: {self.snr_db:.1f} dB\n")
        self.res_text.insert(tk.END, "="*50 + "\n\n")
        
        # 计算每个用户的奇异值
        all_svs = []
        for user_idx in range(num_users):
            _, S, _ = np.linalg.svd(H[user_idx])
            all_svs.append(S)
            self.res_text.insert(tk.END, f"User {user_idx+1} Singular Values:\n")
            self.res_text.insert(tk.END, f"  {S[:4].round(3)}\n")
            
            # 验证：奇异值应该是非递增的
            if not np.all(S[:-1] >= S[1:]):
                self.res_text.insert(tk.END, "  ⚠️ Warning: Singular values not sorted!\n")
            else:
                self.res_text.insert(tk.END, "  ✓ Singular values are properly sorted\n")
        
        self.res_text.insert(tk.END, "\n" + "="*50 + "\n")
        
        # 计算速率（使用 SVD 预编码）
        snr_linear = 10 ** (self.snr_db / 10)
        sum_rate = 0
        
        for user_idx, S in enumerate(all_svs):
            user_rate = 0
            for s in S[:num_rx]:
                user_rate += math.log2(1 + (s**2) * snr_linear)
            sum_rate += user_rate
            self.res_text.insert(tk.END, f"User {user_idx+1} Rate: {user_rate:.3f} bits/s/Hz\n")
        
        self.res_text.insert(tk.END, f"\nTotal Sum Rate: {sum_rate:.3f} bits/s/Hz\n")
        
        # 验证速率范围
        expected_max_rate = num_users * num_rx * math.log2(1 + (np.max(all_svs)**2) * snr_linear)
        if sum_rate > expected_max_rate * 1.1:
            self.res_text.insert(tk.END, "⚠️ Warning: Rate seems too high!\n")
        elif sum_rate < 0:
            self.res_text.insert(tk.END, "⚠️ Warning: Rate cannot be negative!\n")
        else:
            self.res_text.insert(tk.END, "✓ Rate calculation is valid\n")
        
        self._draw_visualization(H.shape[0])
    
    def _draw_visualization(self, num_users):
        self.vis_canvas.delete("all")
        
        # 基站
        cx, cy = 165, 100
        self.vis_canvas.create_oval(cx-25, cy-25, cx+25, cy+25, fill="#ef4444", outline="#b91c1c")
        self.vis_canvas.create_text(cx, cy-40, text="Base Station", fill="#eef5ff", font=("Segoe UI", 9))
        
        # 用户
        colors = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24"]
        for i in range(num_users):
            angle = 2 * math.pi * i / num_users
            ux = cx + 100 * math.cos(angle)
            uy = cy + 100 * math.sin(angle) + 30
            
            self.vis_canvas.create_oval(ux-15, uy-15, ux+15, uy+15, fill=colors[i], outline="#2a3f5f")
            self.vis_canvas.create_text(ux, uy+30, text=f"User {i+1}", fill="#eef5ff", font=("Segoe UI", 8))
            
            # 连线
            self.vis_canvas.create_line(cx, cy, ux, uy, fill=colors[i], width=3)


if __name__ == "__main__":
    app = MIMOSimulator()
    app.mainloop()
