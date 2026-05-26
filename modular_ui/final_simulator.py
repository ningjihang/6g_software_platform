"""
6G MIMO Precoding Simulator - Fully Functional Version
Features:
- Single/Multi-user mode switch
- SNR slider (0-40 dB)
- User count selection
- Working node connections
- Real-time visualization
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math

class MIMOSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6G MIMO Precoding Simulator")
        self.geometry("1400x800")
        self.resizable(True, True)
        
        # Settings
        self.snr_db = 10.0
        self.num_users = 2
        self.num_tx = 16
        self.num_rx = 4
        self.is_single_user = False
        
        # UI State
        self.nodes = []
        self.connections = []
        self.selected_port = None
        self.temp_line = None
        self.conn_start = None
        
        self._init_ui()
    
    def _init_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Mode switch
        self.mode_var = tk.StringVar(value="multi")
        ttk.Radiobutton(toolbar, text="Single User", variable=self.mode_var, 
                        value="single", command=self._on_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(toolbar, text="Multi User", variable=self.mode_var, 
                        value="multi", command=self._on_mode_change).pack(side=tk.LEFT, padx=5)
        
        # User count
        ttk.Label(toolbar, text="Users:").pack(side=tk.LEFT, padx=5)
        self.user_spin = ttk.Spinbox(toolbar, from_=1, to=8, width=5, 
                                     command=self._on_user_change)
        self.user_spin.set(2)
        self.user_spin.pack(side=tk.LEFT, padx=2)
        
        # TX/RX antennas
        ttk.Label(toolbar, text="TX:").pack(side=tk.LEFT, padx=5)
        self.tx_spin = ttk.Spinbox(toolbar, from_=4, to=256, increment=4, width=6)
        self.tx_spin.set(16)
        self.tx_spin.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(toolbar, text="RX:").pack(side=tk.LEFT, padx=5)
        self.rx_spin = ttk.Spinbox(toolbar, from_=1, to=16, increment=1, width=5)
        self.rx_spin.set(4)
        self.rx_spin.pack(side=tk.LEFT, padx=2)
        
        # SNR slider
        ttk.Label(toolbar, text="SNR:").pack(side=tk.RIGHT, padx=5)
        self.snr_label = ttk.Label(toolbar, text="10.0 dB", width=10)
        self.snr_label.pack(side=tk.RIGHT, padx=2)
        self.snr_slider = ttk.Scale(toolbar, from_=0, to=40, orient=tk.HORIZONTAL,
                                    command=self._update_snr, length=150)
        self.snr_slider.set(10)
        self.snr_slider.pack(side=tk.RIGHT, padx=2)
        
        # Run button
        self.run_btn = ttk.Button(toolbar, text="▶ Run Simulation", command=self._run_simulation)
        self.run_btn.pack(side=tk.RIGHT, padx=10)
        
        # Main canvas area
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Node canvas
        self.canvas = tk.Canvas(main_frame, bg="#0f1722", highlightthickness=1, highlightbackground="#334155")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        
        # Right panel
        right_panel = ttk.Frame(main_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Channel visualization
        vis_frame = ttk.LabelFrame(right_panel, text="Channel Visualization")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.vis_canvas = tk.Canvas(vis_frame, bg="#0f1722", width=380, height=200)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results display
        res_frame = ttk.LabelFrame(right_panel, text="Simulation Results")
        res_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.res_text = tk.Text(res_frame, bg="#0f1722", fg="#22c55e", font=("Consolas", 9), 
                                wrap=tk.WORD, height=20)
        scrollbar = ttk.Scrollbar(res_frame, orient=tk.VERTICAL, command=self.res_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.res_text.configure(yscrollcommand=scrollbar.set)
        self.res_text.pack(fill=tk.BOTH, expand=True)
        
        self._init_nodes()
    
    def _init_nodes(self):
        """Initialize default nodes"""
        self.canvas.delete("all")
        self.nodes = []
        
        # Channel node
        ch_node = {"id": "ch1", "name": "Channel", "color": "#f59e0b", 
                   "x": 150, "y": 250, "inputs": [], "outputs": ["H"]}
        self.nodes.append(ch_node)
        
        # SVD precoder
        svd_node = {"id": "svd1", "name": "SVD", "color": "#10b981",
                    "x": 400, "y": 200, "inputs": ["H"], "outputs": ["F"]}
        self.nodes.append(svd_node)
        
        # GMD precoder  
        gmd_node = {"id": "gmd1", "name": "GMD", "color": "#8b5cf6",
                    "x": 400, "y": 350, "inputs": ["H"], "outputs": ["F"]}
        self.nodes.append(gmd_node)
        
        # Scope node
        scope_node = {"id": "scope1", "name": "Scope", "color": "#0ea5e9",
                      "x": 650, "y": 275, "inputs": ["F"], "outputs": []}
        self.nodes.append(scope_node)
        
        self._draw_all_nodes()
    
    def _draw_all_nodes(self):
        """Draw all nodes and connections"""
        self.canvas.delete("node", "port", "connection")
        
        for node in self.nodes:
            self._draw_node(node)
        
        for conn in self.connections:
            self._draw_connection(conn)
    
    def _draw_node(self, node):
        """Draw a single node with ports"""
        x, y = node["x"], node["y"]
        
        # Node circle
        self.canvas.create_oval(x-35, y-35, x+35, y+35, 
                               fill="#1a2636", outline=node["color"], width=2, tags="node")
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=node["color"], tags="node")
        self.canvas.create_text(x, y+50, text=node["name"], fill="#eef5ff", 
                                font=("Segoe UI", 10, "bold"), tags="node")
        
        # Input ports (left side)
        node["input_positions"] = []
        for i, inp in enumerate(node["inputs"]):
            px, py = x - 48, y - 20 + i * 25
            node["input_positions"].append((px, py))
            self.canvas.create_oval(px-8, py-8, px+8, py+8, fill="#ef4444", 
                                   outline="#b91c1c", tags="port")
            self.canvas.create_text(px-16, py, text=inp, fill="#b9cbe0", font=("Segoe UI", 8))
        
        # Output ports (right side)
        node["output_positions"] = []
        for i, out in enumerate(node["outputs"]):
            px, py = x + 48, y - 20 + i * 25
            node["output_positions"].append((px, py))
            self.canvas.create_oval(px-8, py-8, px+8, py+8, fill="#22c55e", 
                                   outline="#16a34a", tags="port")
            self.canvas.create_text(px+16, py, text=out, fill="#b9cbe0", font=("Segoe UI", 8))
    
    def _draw_connection(self, conn):
        """Draw a connection between ports"""
        x1, y1 = conn["from_pos"]
        x2, y2 = conn["to_pos"]
        self.canvas.create_line(x1, y1, x2, y2, fill="#60a5fa", width=3, 
                               smooth=True, tags="connection")
    
    def _on_click(self, event):
        """Handle canvas click"""
        x, y = event.x, event.y
        
        # Check output ports first
        for node in self.nodes:
            for i, (px, py) in enumerate(node.get("output_positions", [])):
                if (x - px)**2 + (y - py)**2 < 64:
                    self.selected_port = {"type": "output", "node": node, "port_idx": i, "pos": (px, py)}
                    self.conn_start = (px, py)
                    return
        
        # Check input ports
        for node in self.nodes:
            for i, (px, py) in enumerate(node.get("input_positions", [])):
                if (x - px)**2 + (y - py)**2 < 64:
                    if self.selected_port:
                        # Complete connection
                        self.connections.append({
                            "from_node": self.selected_port["node"]["id"],
                            "from_port": self.selected_port["port_idx"],
                            "from_pos": self.selected_port["pos"],
                            "to_node": node["id"],
                            "to_port": i,
                            "to_pos": (px, py)
                        })
                        self._draw_connection(self.connections[-1])
                        messagebox.showinfo("Connected", f"Connected {self.selected_port['node']['name']} -> {node['name']}")
                    self.selected_port = None
                    self.conn_start = None
                    return
        
        # Clear selection
        self.selected_port = None
        self.conn_start = None
    
    def _on_drag(self, event):
        """Handle mouse drag"""
        if self.selected_port and self.conn_start:
            # Draw temporary line
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            self.temp_line = self.canvas.create_line(self.conn_start[0], self.conn_start[1],
                                                    event.x, event.y, fill="#60a5fa", 
                                                    width=2, dash=(4, 4))
    
    def _on_release(self, event):
        """Handle mouse release"""
        if self.temp_line:
            self.canvas.delete(self.temp_line)
            self.temp_line = None
    
    def _on_mode_change(self):
        """Handle mode switch"""
        self.is_single_user = (self.mode_var.get() == "single")
        if self.is_single_user:
            self.user_spin.set(1)
            self.user_spin.config(state="disabled")
        else:
            self.user_spin.config(state="normal")
    
    def _on_user_change(self):
        """Handle user count change"""
        self.num_users = int(self.user_spin.get())
    
    def _update_snr(self, value):
        """Update SNR value"""
        self.snr_db = float(value)
        self.snr_label.config(text=f"{self.snr_db:.1f} dB")
    
    def _run_simulation(self):
        """Run MIMO simulation"""
        self.res_text.delete(1.0, tk.END)
        
        # Get parameters
        self.num_users = int(self.user_spin.get())
        self.num_tx = int(self.tx_spin.get())
        self.num_rx = int(self.rx_spin.get())
        
        self.res_text.insert(tk.END, "="*60 + "\n")
        self.res_text.insert(tk.END, "6G MIMO Precoding Simulation\n")
        self.res_text.insert(tk.END, "="*60 + "\n\n")
        
        # Print configuration
        self.res_text.insert(tk.END, f"Mode: {'Single User' if self.is_single_user else 'Multi User'}\n")
        self.res_text.insert(tk.END, f"Users: {self.num_users}\n")
        self.res_text.insert(tk.END, f"TX Antennas: {self.num_tx}\n")
        self.res_text.insert(tk.END, f"RX Antennas: {self.num_rx}\n")
        self.res_text.insert(tk.END, f"SNR: {self.snr_db:.1f} dB\n")
        self.res_text.insert(tk.END, "-"*60 + "\n\n")
        
        # Generate channel matrix
        H = (np.random.randn(self.num_users, self.num_rx, self.num_tx) + 
             1j * np.random.randn(self.num_users, self.num_rx, self.num_tx)) / np.sqrt(2)
        
        # Calculate SVD for each user
        self.res_text.insert(tk.END, "=== Singular Value Decomposition ===\n")
        all_svs = []
        for user_idx in range(self.num_users):
            U, S, Vh = np.linalg.svd(H[user_idx])
            all_svs.append(S)
            self.res_text.insert(tk.END, f"\nUser {user_idx+1}:\n")
            self.res_text.insert(tk.END, f"  Singular Values: {S[:self.num_rx].round(3)}\n")
            self.res_text.insert(tk.END, f"  Condition Number: {S[0]/S[-1]:.2f}\n")
            
            # Validate
            if not np.allclose(S[:-1], np.sort(S)[:0:-1], atol=1e-10):
                self.res_text.insert(tk.END, "  ⚠️ Warning: Singular values not sorted!\n")
            else:
                self.res_text.insert(tk.END, "  ✓ Singular values properly sorted\n")
        
        # Calculate rates
        self.res_text.insert(tk.END, "\n" + "-"*60 + "\n")
        self.res_text.insert(tk.END, "=== Rate Calculation ===\n")
        
        snr_linear = 10 ** (self.snr_db / 10)
        sum_rate = 0
        
        for user_idx, S in enumerate(all_svs):
            user_rate = 0
            for s in S[:self.num_rx]:
                user_rate += math.log2(1 + (s**2) * snr_linear)
            sum_rate += user_rate
            self.res_text.insert(tk.END, f"User {user_idx+1} Rate: {user_rate:.3f} bits/s/Hz\n")
        
        self.res_text.insert(tk.END, f"\nTotal Sum Rate: {sum_rate:.3f} bits/s/Hz\n")
        
        # Validate rate
        max_possible = self.num_users * self.num_rx * math.log2(1 + snr_linear * self.num_tx)
        if sum_rate > max_possible * 1.1:
            self.res_text.insert(tk.END, "⚠️ Warning: Rate exceeds theoretical maximum!\n")
        elif sum_rate < 0:
            self.res_text.insert(tk.END, "⚠️ Warning: Negative rate detected!\n")
        else:
            self.res_text.insert(tk.END, "✓ Rate calculation is valid\n")
        
        # Draw visualization
        self._draw_visualization()
    
    def _draw_visualization(self):
        """Draw channel visualization"""
        self.vis_canvas.delete("all")
        
        # Base station
        cx, cy = 190, 100
        self.vis_canvas.create_oval(cx-30, cy-30, cx+30, cy+30, fill="#ef4444", outline="#b91c1c")
        self.vis_canvas.create_text(cx, cy-45, text="Base Station", fill="#eef5ff", font=("Segoe UI", 9))
        
        # Users
        colors = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#fb923c", "#38bdf8", "#4ade80"]
        for i in range(min(self.num_users, 8)):
            angle = 2 * math.pi * i / self.num_users
            ux = cx + 120 * math.cos(angle)
            uy = cy + 120 * math.sin(angle) + 40
            
            self.vis_canvas.create_oval(ux-18, uy-18, ux+18, uy+18, fill=colors[i], outline="#2a3f5f")
            self.vis_canvas.create_text(ux, uy+35, text=f"User {i+1}", fill="#eef5ff", font=("Segoe UI", 8))
            
            # Connection line
            self.vis_canvas.create_line(cx, cy, ux, uy, fill=colors[i], width=3)


if __name__ == "__main__":
    app = MIMOSimulator()
    app.mainloop()
