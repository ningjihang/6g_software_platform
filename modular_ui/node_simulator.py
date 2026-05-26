import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

# 导入我们的模块
from .node_graph_core import BaseNode, create_node, get_registered_node_types
from .business_nodes import ChannelNode, ScopeNode
from .visualizer import PencilBeamVisualizer
from .config_6g import FrequencyBand


class NodeCanvas(tk.Canvas):
    """
    Tkinter 实现的节点画布 - 圆形节点风格
    - 支持节点拖拽
    - 支持连线
    - 支持右键菜单
    """
    NODE_RADIUS = 50
    PORT_RADIUS = 10  # 增大端口，更容易点击
    
    def __init__(self, parent):
        super().__init__(parent, bg="#0f1722", highlightthickness=0)
        
        self.nodes: Dict[str, dict] = {}  # node_id -> {obj, x, y}
        self.connections: List[dict] = []
        self.next_node_id = 1
        
        self._dragging = None
        self._drag_start_pos = (0, 0)
        self._connecting = False
        self._connect_start = None
        self._connect_line = None
        
        self.grid(row=0, column=0, sticky="nsew")
        
        # 绑定事件
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._on_right_click)
        self.bind("<Double-1>", self._on_double_click)
        
    def _create_node_graphics(self, node: BaseNode, x, y):
        """创建节点图形（圆形风格）"""
        node_id = node.node_id
        color = getattr(node, "NODE_COLOR", "#6366f1")
        category = getattr(node, "NODE_CATEGORY", "General")
        
        # 外圈
        self.create_oval(
            x - self.NODE_RADIUS - 6, y - self.NODE_RADIUS - 6,
            x + self.NODE_RADIUS + 6, y + self.NODE_RADIUS + 6,
            fill="#0f1722", outline="#2a3f5f", width=1,
            tags=(f"node_{node_id}", "nodes")
        )
        
        # 主节点
        self.create_oval(
            x - self.NODE_RADIUS, y - self.NODE_RADIUS,
            x + self.NODE_RADIUS, y + self.NODE_RADIUS,
            fill="#1a2636", outline=color, width=2,
            tags=(f"node_{node_id}", "nodes")
        )
        
        # 中心点
        self.create_oval(
            x - 8, y - 8, x + 8, y + 8,
            fill=color, outline="",
            tags=(f"node_{node_id}", "nodes")
        )
        
        # 节点标题
        self.create_text(
            x, y + self.NODE_RADIUS + 15,
            text=node.name,
            fill="#eef5ff",
            font=("Segoe UI", 10, "bold"),
            tags=(f"node_{node_id}", "nodes")
        )
        
        # 分类标签
        self.create_text(
            x, y + self.NODE_RADIUS + 30,
            text=category,
            fill="#b9cbe0",
            font=("Segoe UI", 8),
            tags=(f"node_{node_id}", "nodes")
        )
        
        # 输入端口（左边）
        for i, port in enumerate(node.inputs):
            px = x - self.NODE_RADIUS * 0.8
            py = y + (i - (len(node.inputs)-1)/2) * 20
            self.create_oval(
                px - self.PORT_RADIUS, py - self.PORT_RADIUS,
                px + self.PORT_RADIUS, py + self.PORT_RADIUS,
                fill="#ef4444", outline="#b91c1c",
                tags=(f"node_{node_id}", f"port_{node_id}_in_{i}", "ports")
            )
            self.create_text(
                px - 12, py,
                text=port.name,
                fill="#b9cbe0",
                anchor="e",
                font=("Segoe UI", 8),
                tags=(f"node_{node_id}", "nodes")
            )
        
        # 输出端口（右边）
        for i, port in enumerate(node.outputs):
            px = x + self.NODE_RADIUS * 0.8
            py = y + (i - (len(node.outputs)-1)/2) * 20
            self.create_oval(
                px - self.PORT_RADIUS, py - self.PORT_RADIUS,
                px + self.PORT_RADIUS, py + self.PORT_RADIUS,
                fill="#22c55e", outline="#16a34a",
                tags=(f"node_{node_id}", f"port_{node_id}_out_{i}", "ports")
            )
            self.create_text(
                px + 12, py,
                text=port.name,
                fill="#b9cbe0",
                anchor="w",
                font=("Segoe UI", 8),
                tags=(f"node_{node_id}", "nodes")
            )
    
    def add_node(self, node_type: str, x: int = 200, y: int = 200):
        """添加节点到画布"""
        node_obj = create_node(node_type, str(self.next_node_id))
        self.next_node_id += 1
        
        node_obj.position = (x, y)
        self._create_node_graphics(node_obj, x, y)
        
        self.nodes[node_obj.node_id] = {
            "obj": node_obj,
            "x": x,
            "y": y
        }
        
        return node_obj
    
    def _on_click(self, event):
        clicked = self.find_closest(event.x, event.y)
        tags = self.gettags(clicked)
        
        for tag in tags:
            if tag.startswith("port_"):
                parts = tag.split("_")
                node_id, io_type, port_idx = parts[1], parts[2], int(parts[3])
                
                if io_type == "out":
                    node = self.nodes[node_id]
                    start_x = node["x"] + self.NODE_RADIUS * 0.8
                    start_y = node["y"] + (port_idx - (len(node["obj"].outputs)-1)/2) * 20
                    self._connecting = True
                    self._connect_start = (node_id, port_idx, start_x, start_y)
                    self._connect_line = self.create_line(
                        start_x, start_y, start_x, start_y,
                        fill="#60a5fa", width=2, dash=(4, 4)
                    )
                return
        
        for tag in tags:
            if tag.startswith("node_"):
                node_id = tag.split("_")[1]
                self._dragging = node_id
                self._drag_start_pos = (event.x, event.y)
                self.tag_raise(tag)
                return
    
    def _on_drag(self, event):
        if self._connecting and self._connect_line:
            start_x, start_y = self._connect_start[2], self._connect_start[3]
            self.coords(self._connect_line, start_x, start_y, event.x, event.y)
        
        elif self._dragging:
            dx = event.x - self._drag_start_pos[0]
            dy = event.y - self._drag_start_pos[1]
            
            node_id = self._dragging
            node = self.nodes[node_id]
            
            self.move(f"node_{node_id}", dx, dy)
            node["x"] += dx
            node["y"] += dy
            node["obj"].position = (node["x"], node["y"])
            
            self._drag_start_pos = (event.x, event.y)
            self._update_connections()
    
    def _on_release(self, event):
        if self._connecting:
            clicked = self.find_closest(event.x, event.y)
            tags = self.gettags(clicked)
            
            for tag in tags:
                if tag.startswith("port_"):
                    parts = tag.split("_")
                    node2_id, io_type2, port2_idx = parts[1], parts[2], int(parts[3])
                    
                    if io_type2 == "in" and node2_id != self._connect_start[0]:
                        self._connect_nodes(
                            self._connect_start[0], self._connect_start[1],
                            node2_id, port2_idx
                        )
            
            if self._connect_line:
                self.delete(self._connect_line)
                self._connect_line = None
            self._connecting = False
            self._connect_start = None
        
        self._dragging = None
    
    def _connect_nodes(self, node1_id, port1_idx, node2_id, port2_idx):
        x1, y1 = self._get_port_position(node1_id, "out", port1_idx)
        x2, y2 = self._get_port_position(node2_id, "in", port2_idx)
        
        line = self.create_line(
            x1, y1, x2, y2,
            fill="#60a5fa", width=3, smooth=True,
            tags=("connections", f"conn_{node1_id}_{port1_idx}_{node2_id}_{port2_idx}")
        )
        
        self.connections.append({
            "from": (node1_id, port1_idx),
            "to": (node2_id, port2_idx),
            "line": line
        })
        
        node1 = self.nodes[node1_id]["obj"]
        node2 = self.nodes[node2_id]["obj"]
        node1.outputs[port1_idx].connect(node2.inputs[port2_idx])
    
    def _get_port_position(self, node_id, io_type, port_idx):
        node = self.nodes.get(node_id)
        if not node:
            return (0, 0)
        
        x, y = node["x"], node["y"]
        if io_type == "out":
            px = x + self.NODE_RADIUS * 0.8
        else:
            px = x - self.NODE_RADIUS * 0.8
        
        num_ports = len(node["obj"].outputs) if io_type == "out" else len(node["obj"].inputs)
        py = y + (port_idx - (num_ports - 1)/2) * 20
        return (px, py)
    
    def _update_connections(self):
        for conn in self.connections:
            node1_id, port1_idx = conn["from"]
            node2_id, port2_idx = conn["to"]
            
            x1, y1 = self._get_port_position(node1_id, "out", port1_idx)
            x2, y2 = self._get_port_position(node2_id, "in", port2_idx)
            self.coords(conn["line"], x1, y1, x2, y2)
    
    def _on_right_click(self, event):
        menu = tk.Menu(self, tearoff=0)
        add_menu = tk.Menu(menu, tearoff=0)
        for node_type in get_registered_node_types():
            add_menu.add_command(
                label=node_type.replace("_", " "),
                command=lambda t=node_type, x=event.x, y=event.y: self.add_node(t, x, y)
            )
        menu.add_cascade(label="Add Node", menu=add_menu)
        menu.add_separator()
        menu.add_command(label="Clear All", command=self._clear_all)
        menu.post(event.x_root, event.y_root)
    
    def _on_double_click(self, event):
        clicked = self.find_closest(event.x, event.y)
        tags = self.gettags(clicked)
        for tag in tags:
            if tag.startswith("node_"):
                node_id = tag.split("_")[1]
                node_obj = self.nodes[node_id]["obj"]
                if hasattr(self.master, "show_property_panel"):
                    self.master.show_property_panel(node_obj)
                break
    
    def _clear_all(self):
        self.delete("all")
        self.nodes = {}
        self.connections = []
        self.next_node_id = 1
    
    def get_nodes(self):
        return [v["obj"] for v in self.nodes.values()]


class NodeBasedSimulator(tk.Tk):
    """
    6G MIMO 节点式仿真器
    """
    def __init__(self):
        super().__init__()
        self.title("6G MIMO Precoding Simulator")
        self.geometry("1400x900")
        self._running = False
        
        # 布局
        self._init_ui()
    
    def _init_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(toolbar, text="Add Channel",
                  command=lambda: self.canvas.add_node("ChannelGenerator", 150, 250)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add SNR Gen",
                  command=lambda: self.canvas.add_node("SNRGenerator", 150, 450)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add SVD",
                  command=lambda: self.canvas.add_node("Precoding_SVD", 450, 150)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add GMD",
                  command=lambda: self.canvas.add_node("Precoding_GMD", 450, 320)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add UCD",
                  command=lambda: self.canvas.add_node("Precoding_UCD", 450, 490)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add Scope",
                  command=lambda: self.canvas.add_node("Scope", 750, 300)).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        self._run_btn = ttk.Button(toolbar, text="▶  Run", command=self._run_simulation)
        self._run_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹  Stop", command=self._stop_simulation).pack(side=tk.LEFT, padx=2)
        
        main_frame = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas_frame = ttk.LabelFrame(main_frame, text="Node Canvas")
        main_frame.add(canvas_frame, weight=3)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = NodeCanvas(canvas_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        right_panel = ttk.Frame(main_frame)
        main_frame.add(right_panel, weight=1)
        
        vis_frame = ttk.LabelFrame(right_panel, text="Channel Visualization")
        vis_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.vis_canvas = tk.Canvas(vis_frame, bg="#0f1722")
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)
        self.visualizer = PencilBeamVisualizer(self.vis_canvas)
        
        data_frame = ttk.LabelFrame(right_panel, text="Real-time Results")
        data_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._data_text = tk.Text(data_frame, height=12, bg="#0f1722", fg="#22c55e",
                                 font=("Consolas", 9))
        self._data_text.pack(fill=tk.BOTH, expand=True)
        
        self._data_text.insert(tk.END, "=== 6G MIMO Simulation ===\n\n")
        self._data_text.insert(tk.END, "Usage:\n")
        self._data_text.insert(tk.END, "1. Add Channel (or SNR Gen) and connect to Precoders\n")
        self._data_text.insert(tk.END, "2. Connect Precoders to Scope\n")
        self._data_text.insert(tk.END, "3. Double-click Channel to set Sub-THz/XL-MIMO\n")
        self._data_text.insert(tk.END, "4. Click 'Run' to start\n\n")
        
        prop_frame = ttk.LabelFrame(right_panel, text="Properties")
        prop_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self._prop_frame = prop_frame
        self.show_property_panel(None)
    
    def show_property_panel(self, node: Optional[BaseNode]):
        for widget in self._prop_frame.winfo_children():
            widget.destroy()
        
        if not node:
            tk.Label(self._prop_frame, text="Select a node to view properties",
                    fg="#b9cbe0").pack(pady=10)
            return
        
        tk.Label(self._prop_frame, text=f"Node: {node.name}",
                font=("Segoe UI", 11, "bold"), fg="#eef5ff").pack(anchor=tk.W, padx=5, pady=2)
        
        if isinstance(node, ChannelNode):
            tk.Label(self._prop_frame, text="Frequency Band:", fg="#b9cbe0").pack(anchor=tk.W, padx=5)
            band_var = tk.StringVar(value=node.get_property("frequency_band", FrequencyBand.SUB6G.value))
            cb = ttk.Combobox(self._prop_frame, textvariable=band_var, state="readonly",
                            values=[b.value for b in FrequencyBand])
            cb.pack(fill=tk.X, padx=5, pady=2)
            
            def update_band():
                node.set_property("frequency_band", band_var.get())
                self._update_visualizer(node)
            
            cb.bind("<<ComboboxSelected>>", lambda e: update_band())
            
            tk.Label(self._prop_frame, text="Num Users:", fg="#b9cbe0").pack(anchor=tk.W, padx=5)
            user_var = tk.IntVar(value=node.get_property("num_users", 2))
            spin = ttk.Spinbox(self._prop_frame, from_=1, to=8, textvariable=user_var)
            spin.pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(self._prop_frame, text="TX Antennas:", fg="#b9cbe0").pack(anchor=tk.W, padx=5)
            tx_var = tk.IntVar(value=node.get_property("num_tx_antennas", 16))
            tx_spin = ttk.Spinbox(self._prop_frame, from_=4, to=1024, textvariable=tx_var)
            tx_spin.pack(fill=tk.X, padx=5, pady=2)
            
            def update_props():
                node.set_property("num_users", user_var.get())
                node.set_property("num_tx_antennas", tx_var.get())
                self._update_visualizer(node)
            
            spin.bind("<FocusOut>", lambda e: update_props())
            tx_spin.bind("<FocusOut>", lambda e: update_props())
        
        tk.Button(self._prop_frame, text="Update Visualization",
                 command=lambda: self._update_visualizer(node)).pack(pady=10, padx=5, fill=tk.X)
    
    def _update_visualizer(self, node):
        if isinstance(node, ChannelNode):
            band_name = node.get_property("frequency_band", FrequencyBand.SUB6G.value)
            is_subthz = "Sub-THz" in band_name
            is_xl = node.get_property("num_tx_antennas", 16) >= 256
            self.visualizer.set_parameters(
                is_subthz, is_xl,
                node.get_property("num_users", 2),
                node.get_property("num_tx_antennas", 16)
            )
            self.visualizer.draw()
    
    def _update_data_display(self, results):
        self._data_text.delete(1.0, tk.END)
        self._data_text.insert(tk.END, "=== 6G MIMO Simulation Results ===\n\n")
        
        if results and results["snrs"]:
            self._data_text.insert(tk.END, f"Last SNR: {results['snrs'][-1]} dB\n")
            
            if results["singular_values"]:
                last_svs = results["singular_values"][-1]
                self._data_text.insert(tk.END, f"\nSingular Values:\n")
                for user_idx, svs in enumerate(last_svs):
                    self._data_text.insert(tk.END, f"  User {user_idx+1}: {[f'{s:.2f}' for s in svs]}\n")
            
            if results["rates"]:
                self._data_text.insert(tk.END, f"\nSum Rate: {results['rates'][-1]:.2f} bits/s/Hz")
            
            self._data_text.see(tk.END)
    
    def _run_simulation(self):
        if self._running:
            return
        
        self._running = True
        self._run_btn.config(text="▶ Running...")
        
        nodes = self.canvas.get_nodes()
        
        scope_nodes = [n for n in nodes if isinstance(n, ScopeNode)]
        for scope in scope_nodes:
            scope.set_update_callback(self._update_data_display)
            scope.results = {"snrs": [], "rates": [], "bers": [], "singular_values": []}
        
        channel_nodes = [n for n in nodes if isinstance(n, ChannelNode)]
        for ch in channel_nodes:
            self._update_visualizer(ch)
        
        source_nodes = [n for n in nodes if not n.inputs]
        self._sim_step(source_nodes, scope_nodes, 0)
    
    def _sim_step(self, source_nodes, scope_nodes, step):
        if not self._running:
            return
        
        if step > 0 and step % 20 == 0:
            for n in source_nodes:
                n.run()
        
        if step < 100:
            self.after(50, lambda: self._sim_step(source_nodes, scope_nodes, step + 1))
        else:
            self._running = False
            self._run_btn.config(text="▶  Run")
    
    def _stop_simulation(self):
        self._running = False
        self._run_btn.config(text="▶  Run")


def main():
    app = NodeBasedSimulator()
    
    # 预置一些节点
    app.canvas.add_node("ChannelGenerator", 150, 250)
    app.canvas.add_node("SNRGenerator", 150, 450)
    app.canvas.add_node("Precoding_SVD", 450, 150)
    app.canvas.add_node("Precoding_GMD", 450, 320)
    app.canvas.add_node("Precoding_UCD", 450, 490)
    app.canvas.add_node("Scope", 750, 300)
    
    app.mainloop()


if __name__ == "__main__":
    main()
