"""
6G MIMO Precoding Simulator - Enhanced Version
Based on original code structure with 6G features
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math
import random

class MIMOSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6G MIMO Precoding Simulator")
        self.geometry("1400x850")
        self.resizable(True, True)
        
        # System parameters
        self.params = {
            'snr_db': 10.0,
            'num_users': 2,
            'num_tx': 16,
            'num_rx': 4,
            'carrier_freq': 3.5e9,
            'is_single_user': False,
            'channel_model': 'CDL-A'
        }
        
        # UI State
        self.nodes = []
        self.connections = []
        self.selected_port = None
        self.temp_line = None
        self.user_positions = []
        
        self._init_ui()
    
    def _init_ui(self):
        # Main toolbar
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
        self.tx_spin = ttk.Spinbox(toolbar, from_=4, to=1024, increment=4, width=6)
        self.tx_spin.set(16)
        self.tx_spin.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(toolbar, text="RX:").pack(side=tk.LEFT, padx=5)
        self.rx_spin = ttk.Spinbox(toolbar, from_=1, to=16, increment=1, width=5)
        self.rx_spin.set(4)
        self.rx_spin.pack(side=tk.LEFT, padx=2)
        
        # 6G Frequency selector
        ttk.Label(toolbar, text="Frequency:").pack(side=tk.LEFT, padx=5)
        self.freq_var = tk.StringVar(value="3.5GHz")
        freq_options = ["3.5GHz (Sub-6G)", "28GHz (mmWave)", "140GHz (Sub-THz)", "300GHz (THz)"]
        self.freq_combo = ttk.Combobox(toolbar, textvariable=self.freq_var, values=freq_options, width=20)
        self.freq_combo.pack(side=tk.LEFT, padx=2)
        
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
        
        # Main content frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Node canvas
        self.canvas = tk.Canvas(main_frame, bg="#1e293b", highlightthickness=1, highlightbackground="#475569")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        
        # Right panel
        right_panel = ttk.Frame(main_frame, width=420)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Channel visualization
        vis_frame = ttk.LabelFrame(right_panel, text="Channel Visualization")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.vis_canvas = tk.Canvas(vis_frame, bg="#1e293b", width=400, height=250)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results display
        res_frame = ttk.LabelFrame(right_panel, text="Simulation Results")
        res_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.res_text = tk.Text(res_frame, bg="#1e293b", fg="#22c55e", font=("Consolas", 9),
                                wrap=tk.WORD, height=22)
        scrollbar = ttk.Scrollbar(res_frame, orient=tk.VERTICAL, command=self.res_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.res_text.configure(yscrollcommand=scrollbar.set)
        self.res_text.pack(fill=tk.BOTH, expand=True)
        
        self._init_nodes()
    
    def _init_nodes(self):
        """Initialize default node layout"""
        self.canvas.delete("all")
        self.nodes = []
        
        # Channel node
        ch_node = {
            'id': 'ch1',
            'name': 'Channel',
            'color': '#f59e0b',
            'x': 180,
            'y': 280,
            'inputs': [],
            'outputs': ['H']
        }
        self.nodes.append(ch_node)
        
        # SVD precoder
        svd_node = {
            'id': 'svd1',
            'name': 'SVD',
            'color': '#10b981',
            'x': 450,
            'y': 200,
            'inputs': ['H'],
            'outputs': ['F']
        }
        self.nodes.append(svd_node)
        
        # GMD precoder
        gmd_node = {
            'id': 'gmd1',
            'name': 'GMD',
            'color': '#8b5cf6',
            'x': 450,
            'y': 360,
            'inputs': ['H'],
            'outputs': ['F']
        }
        self.nodes.append(gmd_node)
        
        # UCD precoder
        ucd_node = {
            'id': 'ucd1',
            'name': 'UCD',
            'color': '#ec4899',
            'x': 450,
            'y': 520,
            'inputs': ['H'],
            'outputs': ['F']
        }
        self.nodes.append(ucd_node)
        
        # Scope node
        scope_node = {
            'id': 'scope1',
            'name': 'Scope',
            'color': '#0ea5e9',
            'x': 720,
            'y': 340,
            'inputs': ['F'],
            'outputs': []
        }
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
        x, y = node['x'], node['y']
        
        # Node body
        self.canvas.create_oval(x-40, y-40, x+40, y+40,
                               fill="#334155", outline=node['color'], width=2, tags="node")
        self.canvas.create_oval(x-6, y-6, x+6, y+6, fill=node['color'], tags="node")
        self.canvas.create_text(x, y+55, text=node['name'], fill="#f1f5f9",
                                font=("Segoe UI", 11, "bold"), tags="node")
        
        # Input ports
        node['input_positions'] = []
        for i, inp in enumerate(node['inputs']):
            px, py = x - 52, y - 25 + i * 28
            node['input_positions'].append((px, py))
            self.canvas.create_oval(px-9, py-9, px+9, py+9, fill="#ef4444",
                                   outline="#dc2626", tags="port")
            self.canvas.create_text(px-18, py, text=inp, fill="#cbd5e1", font=("Segoe UI", 8))
        
        # Output ports
        node['output_positions'] = []
        for i, out in enumerate(node['outputs']):
            px, py = x + 52, y - 25 + i * 28
            node['output_positions'].append((px, py))
            self.canvas.create_oval(px-9, py-9, px+9, py+9, fill="#22c55e",
                                   outline="#16a34a", tags="port")
            self.canvas.create_text(px+18, py, text=out, fill="#cbd5e1", font=("Segoe UI", 8))
    
    def _draw_connection(self, conn):
        """Draw connection between ports"""
        x1, y1 = conn['from_pos']
        x2, y2 = conn['to_pos']
        self.canvas.create_line(x1, y1, x2, y2, fill="#60a5fa", width=3,
                               smooth=True, tags="connection")
    
    def _on_click(self, event):
        """Handle canvas click"""
        x, y = event.x, event.y
        
        # Check output ports
        for node in self.nodes:
            for i, (px, py) in enumerate(node.get('output_positions', [])):
                if (x - px)**2 + (y - py)**2 < 81:
                    self.selected_port = {
                        'type': 'output',
                        'node': node,
                        'port_idx': i,
                        'pos': (px, py)
                    }
                    return
        
        # Check input ports
        for node in self.nodes:
            for i, (px, py) in enumerate(node.get('input_positions', [])):
                if (x - px)**2 + (y - py)**2 < 81:
                    if self.selected_port:
                        self.connections.append({
                            'from_node': self.selected_port['node']['id'],
                            'from_port': self.selected_port['port_idx'],
                            'from_pos': self.selected_port['pos'],
                            'to_node': node['id'],
                            'to_port': i,
                            'to_pos': (px, py)
                        })
                        self._draw_connection(self.connections[-1])
                        messagebox.showinfo("Connected", 
                            f"{self.selected_port['node']['name']} -> {node['name']}")
                    self.selected_port = None
                    return
        
        self.selected_port = None
    
    def _on_drag(self, event):
        """Handle mouse drag for temp line"""
        if self.selected_port:
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            self.temp_line = self.canvas.create_line(
                self.selected_port['pos'][0], self.selected_port['pos'][1],
                event.x, event.y, fill="#60a5fa", width=2, dash=(4, 4))
    
    def _on_release(self, event):
        """Handle mouse release"""
        if self.temp_line:
            self.canvas.delete(self.temp_line)
            self.temp_line = None
    
    def _on_mode_change(self):
        """Handle mode switch"""
        self.params['is_single_user'] = (self.mode_var.get() == 'single')
        if self.params['is_single_user']:
            self.user_spin.set(1)
            self.user_spin.config(state='disabled')
        else:
            self.user_spin.config(state='normal')
    
    def _on_user_change(self):
        """Handle user count change"""
        self.params['num_users'] = int(self.user_spin.get())
    
    def _update_snr(self, value):
        """Update SNR value"""
        self.params['snr_db'] = float(value)
        self.snr_label.config(text=f"{self.params['snr_db']:.1f} dB")
    
    def _parse_frequency(self):
        """Parse frequency from combobox"""
        freq_text = self.freq_var.get()
        if '3.5GHz' in freq_text:
            return 3.5e9
        elif '28GHz' in freq_text:
            return 28e9
        elif '140GHz' in freq_text:
            return 140e9
        elif '300GHz' in freq_text:
            return 300e9
        return 3.5e9
    
    def _generate_channel_with_distance(self):
        """Generate channel matrix with distance-dependent pathloss"""
        num_users = int(self.user_spin.get())
        num_tx = int(self.tx_spin.get())
        num_rx = int(self.rx_spin.get())
        carrier_freq = self._parse_frequency()
        
        # Generate user positions with random distances
        self.user_positions = []
        channels = []
        
        for i in range(num_users):
            # Random distance between 50m and 500m
            distance = 50 + random.uniform(0, 450)
            angle = 2 * math.pi * i / num_users
            self.user_positions.append({
                'distance': distance,
                'angle': angle,
                'x': math.cos(angle) * distance,
                'y': math.sin(angle) * distance
            })
            
            # Path loss calculation (Friis formula)
            wavelength = 3e8 / carrier_freq
            path_loss = (4 * math.pi * distance / wavelength) ** 2
            
            # Generate channel matrix with path loss
            H = (np.random.randn(num_rx, num_tx) + 
                 1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2)
            
            # Apply path loss and shadowing
            shadowing = 10 ** (np.random.normal(0, 8) / 20)
            H = H * np.sqrt(1 / (path_loss * shadowing))
            
            channels.append(H)
        
        return np.array(channels), carrier_freq
    
    def _run_simulation(self):
        """Run MIMO simulation with enhanced features"""
        self.res_text.delete(1.0, tk.END)
        
        # Get parameters
        self.params['num_users'] = int(self.user_spin.get())
        self.params['num_tx'] = int(self.tx_spin.get())
        self.params['num_rx'] = int(self.rx_spin.get())
        self.params['carrier_freq'] = self._parse_frequency()
        
        freq_label = self.freq_var.get()
        
        # Generate channel with distance
        H, carrier_freq = self._generate_channel_with_distance()
        
        # Write header
        self.res_text.insert(tk.END, "="*65 + "\n")
        self.res_text.insert(tk.END, "6G MIMO Precoding Simulation Results\n")
        self.res_text.insert(tk.END, "="*65 + "\n\n")
        
        # Configuration info
        self.res_text.insert(tk.END, f"Mode:           {'Single User' if self.params['is_single_user'] else 'Multi User'}\n")
        self.res_text.insert(tk.END, f"Users:          {self.params['num_users']}\n")
        self.res_text.insert(tk.END, f"TX Antennas:    {self.params['num_tx']}")
        if self.params['num_tx'] >= 256:
            self.res_text.insert(tk.END, "  [XL-MIMO Mode]")
        self.res_text.insert(tk.END, "\n")
        self.res_text.insert(tk.END, f"RX Antennas:    {self.params['num_rx']}\n")
        self.res_text.insert(tk.END, f"Frequency:      {freq_label}")
        if carrier_freq >= 100e9:
            self.res_text.insert(tk.END, "  [Sub-THz Band]")
        self.res_text.insert(tk.END, "\n")
        self.res_text.insert(tk.END, f"SNR:            {self.params['snr_db']:.1f} dB\n")
        self.res_text.insert(tk.END, "-"*65 + "\n\n")
        
        # User distances
        self.res_text.insert(tk.END, "=== User Distances & Channel Quality ===\n")
        for i, pos in enumerate(self.user_positions):
            self.res_text.insert(tk.END, f"User {i+1}: Distance = {pos['distance']:.1f} m")
            if pos['distance'] < 100:
                self.res_text.insert(tk.END, "  [Good Channel]")
            elif pos['distance'] > 300:
                self.res_text.insert(tk.END, "  [Poor Channel]")
            self.res_text.insert(tk.END, "\n")
        self.res_text.insert(tk.END, "\n")
        
        # SVD decomposition
        self.res_text.insert(tk.END, "=== Singular Value Decomposition ===\n")
        all_svs = []
        for user_idx in range(self.params['num_users']):
            _, S, _ = np.linalg.svd(H[user_idx])
            all_svs.append(S)
            self.res_text.insert(tk.END, f"\nUser {user_idx+1} (Dist: {self.user_positions[user_idx]['distance']:.0f}m):\n")
            self.res_text.insert(tk.END, f"  Singular Values: {S[:self.params['num_rx']].round(3)}\n")
            self.res_text.insert(tk.END, f"  Max Eigenvalue:  {S[0]**2:.3f}\n")
            self.res_text.insert(tk.END, f"  Min Eigenvalue:  {S[-1]**2:.3f}\n")
            
            if not np.allclose(S[:-1], np.sort(S)[:0:-1], atol=1e-10):
                self.res_text.insert(tk.END, "  ⚠️ Warning: Singular values not sorted!\n")
            else:
                self.res_text.insert(tk.END, "  ✓ Valid SVD decomposition\n")
        
        # Rate calculation
        self.res_text.insert(tk.END, "\n" + "-"*65 + "\n")
        self.res_text.insert(tk.END, "=== Rate Calculation ===\n")
        
        snr_linear = 10 ** (self.params['snr_db'] / 10)
        sum_rate = 0
        
        for user_idx, S in enumerate(all_svs):
            user_rate = 0
            for s in S[:self.params['num_rx']]:
                user_rate += math.log2(1 + (s**2) * snr_linear)
            sum_rate += user_rate
            self.res_text.insert(tk.END, f"User {user_idx+1} Rate: {user_rate:.3f} bits/s/Hz\n")
        
        self.res_text.insert(tk.END, f"\nTotal Sum Rate: {sum_rate:.3f} bits/s/Hz\n")
        
        # Validate
        max_rate = self.params['num_users'] * self.params['num_rx'] * math.log2(1 + snr_linear * self.params['num_tx'])
        if sum_rate > max_rate * 1.1:
            self.res_text.insert(tk.END, "⚠️ Warning: Rate exceeds theoretical maximum!\n")
        elif sum_rate < 0:
            self.res_text.insert(tk.END, "⚠️ Warning: Negative rate detected!\n")
        else:
            self.res_text.insert(tk.END, "✓ Rate calculation is valid\n")
        
        # Draw visualization
        self._draw_visualization(carrier_freq)
    
    def _draw_visualization(self, carrier_freq):
        """Draw enhanced channel visualization"""
        self.vis_canvas.delete("all")
        
        # Base station
        cx, cy = 200, 125
        self.vis_canvas.create_rectangle(cx-40, cy-30, cx+40, cy+30, 
                                        fill="#ef4444", outline="#dc2626")
        self.vis_canvas.create_text(cx, cy-50, text="Base Station", 
                                   fill="#f1f5f9", font=("Segoe UI", 10, "bold"))
        
        # Frequency indicator
        freq_text = self.freq_var.get()
        freq_color = "#22c55e" if carrier_freq < 100e9 else "#f59e0b" if carrier_freq < 200e9 else "#ef4444"
        self.vis_canvas.create_text(cx, cy+50, text=freq_text, 
                                   fill=freq_color, font=("Segoe UI", 9))
        
        # Users with distance-based visualization
        colors = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#fb923c", "#38bdf8", "#4ade80"]
        
        for i, pos in enumerate(self.user_positions):
            # Position calculation
            angle = pos['angle']
            distance = pos['distance']
            scale = 0.4  # Scale factor for visualization
            ux = cx + distance * scale * math.cos(angle)
            uy = cy + distance * scale * math.sin(angle) + 50
            
            # User circle size depends on channel quality (distance)
            radius = max(12, 25 - distance / 25)
            self.vis_canvas.create_oval(ux-radius, uy-radius, ux+radius, uy+radius,
                                       fill=colors[i], outline="#2a3f5f")
            self.vis_canvas.create_text(ux, uy+35, text=f"User {i+1}", 
                                       fill="#f1f5f9", font=("Segoe UI", 8))
            self.vis_canvas.create_text(ux, uy+50, text=f"{distance:.0f}m", 
                                       fill="#94a3b8", font=("Segoe UI", 7))
            
            # Connection line with quality indicator
            line_width = max(1, 4 - distance / 150)
            line_color = self._get_channel_color(distance, carrier_freq)
            
            # Pencil beam for Sub-THz frequencies
            if carrier_freq >= 100e9:
                # Draw pencil beam (narrow, focused)
                self.vis_canvas.create_line(cx, cy, ux, uy, fill=line_color, 
                                           width=line_width, arrow=tk.LAST, arrowshape=(10, 12, 6))
                # Add beam effect
                for offset in [-3, 0, 3]:
                    self.vis_canvas.create_line(cx, cy + offset, ux, uy, 
                                               fill=line_color, width=1, dash=(2, 2))
            else:
                # Regular multipath
                self.vis_canvas.create_line(cx, cy, ux, uy, fill=line_color, 
                                           width=line_width)
    
    def _get_channel_color(self, distance, carrier_freq):
        """Get channel quality color based on distance and frequency"""
        if distance < 100:
            return "#22c55e"  # Good - green
        elif distance < 200:
            return "#eab308"  # Medium - yellow
        elif distance < 350:
            return "#f97316"  # Poor - orange
        else:
            return "#ef4444"  # Very poor - red


if __name__ == "__main__":
    app = MIMOSimulator()
    app.mainloop()
