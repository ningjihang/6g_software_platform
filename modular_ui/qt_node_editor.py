from __future__ import annotations

from dataclasses import dataclass
import math
import sys

try:
    from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QAction, QColor, QPainter, QPainterPath, QPen, QBrush
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsPathItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_BINDING = "PyQt6"
    ANTIALIAS = QPainter.RenderHint.Antialiasing
    TEXT_ANTIALIAS = QPainter.RenderHint.TextAntialiasing
    NO_PEN = Qt.PenStyle.NoPen
    HORIZONTAL = Qt.Orientation.Horizontal
except Exception:
    from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QBrush
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsPathItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_BINDING = "PyQt5"
    ANTIALIAS = QPainter.Antialiasing
    TEXT_ANTIALIAS = QPainter.TextAntialiasing
    NO_PEN = Qt.NoPen
    HORIZONTAL = Qt.Horizontal

import modular_ui.business_nodes  # ensure node registry is populated
from .business_nodes import ChannelNode
from .channel_visualizer import ChannelVisualizer
from .config_6g import FrequencyBand, XL_MIMO_CONFIG, display_frequency_bands
from .node_graph_core import BaseNode, GraphCycleError, NodeGraph, NodePort, PortType, create_node, get_registered_node_types

if QT_BINDING == "PyQt6":
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


WORKBENCH_BG = "#07111f"
PANEL_BG = "#0f1b2d"
CARD_BG = "#13233a"
TEXT = "#e6f1ff"
MUTED = "#8da7c7"
GRID = "#10243b"
EDGE = "#6fd3ff"
EDGE_ACTIVE = "#f59e0b"
PORT_IN = "#f87171"
PORT_OUT = "#34d399"


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", port: NodePort, is_input: bool) -> None:
        self.node_item = node_item
        self.port = port
        self.is_input = is_input
        radius = 6.0
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0, node_item)
        self.setBrush(QBrush(QColor(PORT_IN if is_input else PORT_OUT)))
        self.setPen(QPen(QColor("#dbeafe"), 1.0))
        self.setZValue(3)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.node_item.editor.begin_connection_drag(self)
        event.accept()


class NodeItem(QGraphicsRectItem):
    WIDTH = 220
    HEADER_H = 34

    def __init__(self, editor: "NodeScene", node: BaseNode) -> None:
        super().__init__(0.0, 0.0, self.WIDTH, self._body_height(node))
        self.editor = editor
        self.node = node
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setBrush(QBrush(QColor(CARD_BG)))
        self.setPen(QPen(QColor(node.NODE_COLOR), 1.6))
        self.setZValue(1)

        header = QGraphicsRectItem(0.0, 0.0, self.WIDTH, self.HEADER_H, self)
        header.setBrush(QBrush(QColor(node.NODE_COLOR)))
        header.setPen(QPen(NO_PEN))

        title = QGraphicsSimpleTextItem(node.name, self)
        title.setBrush(QBrush(QColor("#08111d")))
        title.setPos(10.0, 8.0)

        subtitle = QGraphicsSimpleTextItem(node.NODE_CATEGORY, self)
        subtitle.setBrush(QBrush(QColor(MUTED)))
        subtitle.setPos(10.0, self.HEADER_H + 10.0)

        self.input_items: list[PortItem] = []
        self.output_items: list[PortItem] = []
        self._build_ports()

    @staticmethod
    def _body_height(node: BaseNode) -> float:
        port_rows = max(len(node.inputs), len(node.outputs), 1)
        return max(96.0, NodeItem.HEADER_H + 26.0 + port_rows * 24.0)

    def _build_ports(self) -> None:
        port_rows = max(len(self.node.inputs), len(self.node.outputs), 1)
        top = self.HEADER_H + 28.0
        step = 24.0 if port_rows > 1 else 0.0

        for idx, port in enumerate(self.node.inputs):
            item = PortItem(self, port, True)
            y = top + idx * step
            item.setPos(10.0, y)
            label = QGraphicsSimpleTextItem(port.name, self)
            label.setBrush(QBrush(QColor(MUTED)))
            label.setPos(22.0, y - 8.0)
            self.input_items.append(item)

        for idx, port in enumerate(self.node.outputs):
            item = PortItem(self, port, False)
            y = top + idx * step
            item.setPos(self.WIDTH - 10.0, y)
            label = QGraphicsSimpleTextItem(port.name, self)
            label.setBrush(QBrush(QColor(MUTED)))
            bounds = label.boundingRect()
            label.setPos(self.WIDTH - 22.0 - bounds.width(), y - 8.0)
            self.output_items.append(item)

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.ItemPositionHasChanged:
            pos = self.pos()
            self.node.position = (float(pos.x()), float(pos.y()))
            self.editor.refresh_edges()
        return super().itemChange(change, value)


class EdgeItem(QGraphicsPathItem):
    def __init__(self, source_item: PortItem, target_item: PortItem | None = None) -> None:
        super().__init__()
        self.source_item = source_item
        self.target_item = target_item
        self.dynamic_target: QPointF | None = None
        self.phase = 0.0
        self.setZValue(0)
        self.setPen(QPen(QColor(EDGE), 2.0))
        self.refresh_path()

    def refresh_path(self, dynamic_target: QPointF | None = None) -> None:
        if dynamic_target is not None:
            self.dynamic_target = dynamic_target
        source = self.source_item.scenePos()
        target = self.target_item.scenePos() if self.target_item is not None else self.dynamic_target or source
        dx = max(80.0, abs(target.x() - source.x()) * 0.5)
        path = QPainterPath(source)
        path.cubicTo(
            QPointF(source.x() + dx, source.y()),
            QPointF(target.x() - dx, target.y()),
            target,
        )
        self.setPath(path)

    def pulse(self, active: bool) -> None:
        self.phase = (self.phase + 1.0) % 12.0
        color = QColor(EDGE_ACTIVE if active else EDGE)
        pen = QPen(color, 2.2 if active else 2.0)
        pen.setDashPattern([6.0, 4.0])
        pen.setDashOffset(-self.phase)
        self.setPen(pen)


class NodeScene(QGraphicsScene):
    node_selected = pyqtSignal(object)

    def __init__(self, graph: NodeGraph, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.graph = graph
        self.setBackgroundBrush(QBrush(QColor(WORKBENCH_BG)))
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: dict[tuple[str, str, str, str], EdgeItem] = {}
        self.pending_edge: EdgeItem | None = None
        self.pending_source: PortItem | None = None
        self.highlight_active = False

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        painter.save()
        painter.setPen(QPen(QColor(GRID), 1.0))
        step = 28
        left = int(math.floor(rect.left() / step) * step)
        top = int(math.floor(rect.top() / step) * step)
        x = left
        while x < rect.right():
            painter.drawLine(
                QPointF(float(x), float(rect.top())),
                QPointF(float(x), float(rect.bottom())),
            )
            x += step
        y = top
        while y < rect.bottom():
            painter.drawLine(
                QPointF(float(rect.left()), float(y)),
                QPointF(float(rect.right()), float(y)),
            )
            y += step
        painter.restore()

    def add_runtime_node(self, node: BaseNode, position: QPointF) -> NodeItem:
        self.graph.add_node(node)
        item = NodeItem(self, node)
        item.setPos(position)
        node.position = (position.x(), position.y())
        self.addItem(item)
        self.node_items[node.node_id] = item
        return item

    def begin_connection_drag(self, source_item: PortItem) -> None:
        if source_item.port.port_type != PortType.OUTPUT:
            return
        self.pending_source = source_item
        self.pending_edge = EdgeItem(source_item)
        self.addItem(self.pending_edge)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.pending_edge is not None:
            self.pending_edge.refresh_path(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.pending_edge is not None and self.pending_source is not None:
            target_item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else None)
            if isinstance(target_item, PortItem) and target_item.port.port_type == PortType.INPUT:
                self._finalize_connection(self.pending_source, target_item)
            self.removeItem(self.pending_edge)
            self.pending_edge = None
            self.pending_source = None
        super().mouseReleaseEvent(event)
        selected = self.selectedItems()
        if selected and isinstance(selected[0], NodeItem):
            self.node_selected.emit(selected[0].node)

    def _finalize_connection(self, source_item: PortItem, target_item: PortItem) -> None:
        try:
            self.graph.connect(source_item.port, target_item.port)
        except GraphCycleError as exc:
            QMessageBox.warning(None, "Invalid Connection", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(None, "Connection Failed", str(exc))
            return

        key = (
            source_item.port.node.node_id,
            source_item.port.name,
            target_item.port.node.node_id,
            target_item.port.name,
        )
        self._create_edge_graphic(source_item.port, target_item.port)
        self.refresh_edges()

    def _create_edge_graphic(self, source_port: NodePort, target_port: NodePort) -> None:
        key = (
            source_port.node.node_id,
            source_port.name,
            target_port.node.node_id,
            target_port.name,
        )
        if key in self.edge_items:
            return
        source_item = self.node_items[source_port.node.node_id].output_items[source_port.index]
        target_item = self.node_items[target_port.node.node_id].input_items[target_port.index]
        edge = EdgeItem(source_item, target_item)
        self.addItem(edge)
        self.edge_items[key] = edge

    def refresh_edges(self) -> None:
        for edge in self.edge_items.values():
            edge.refresh_path()

    def animate_edges(self) -> None:
        self.highlight_active = not self.highlight_active
        for edge in self.edge_items.values():
            edge.pulse(self.highlight_active)


class PropertiesPane(QWidget):
    def __init__(self, graph: NodeGraph, on_change, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.graph = graph
        self.on_change = on_change
        self.node: BaseNode | None = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        self.title = QLabel("Node Properties")
        self.title.setStyleSheet("color: #e6f1ff; font-size: 16px; font-weight: 600;")
        self.layout.addWidget(self.title)

        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(8)
        self.layout.addWidget(self.form_host)
        self.layout.addStretch(1)
        self.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT};")

    def set_node(self, node: BaseNode | None) -> None:
        self.node = node
        while self.form.rowCount():
            self.form.removeRow(0)

        if node is None:
            self.title.setText("Node Properties")
            self.form.addRow(QLabel("Select a node on the graph."))
            return

        self.title.setText(node.name)
        self.form.addRow("Type", QLabel(node.NODE_TYPE))
        self.form.addRow("Category", QLabel(node.NODE_CATEGORY))

        if isinstance(node, ChannelNode):
            self._add_channel_fields(node)
        elif node.NODE_TYPE == "UserConfig":
            self._add_user_config_fields(node)
        elif node.NODE_TYPE == "SNRGenerator":
            self._add_snr_fields(node)
        else:
            for key, value in node.properties.items():
                self.form.addRow(str(key), QLabel(str(value)))

    def _add_channel_fields(self, node: ChannelNode) -> None:
        band_box = QComboBox()
        band_box.addItems(display_frequency_bands())
        band_box.setCurrentText(str(node.get_property("frequency_band", FrequencyBand.SUB6G.value)))
        band_box.currentTextChanged.connect(lambda value: self._set_property(node, "frequency_band", value))
        self.form.addRow("Band", band_box)

        tx_host = QWidget()
        tx_layout = QVBoxLayout(tx_host)
        tx_layout.setContentsMargins(0, 0, 0, 0)
        tx_layout.setSpacing(4)

        tx_spin = QSpinBox()
        tx_spin.setRange(4, 1024)
        tx_spin.setSingleStep(4)
        tx_spin.setValue(int(node.get_property("num_tx_antennas", 16)))
        tx_mode_label = QLabel()
        tx_mode_label.setStyleSheet("color:#9db7d7;")

        nrf_label = QLabel()
        nrf_label.setStyleSheet("color:#9db7d7;")

        def _update_tx_related(value: int) -> None:
            self._set_property(node, "num_tx_antennas", int(value))
            tx_mode_label.setText("Tx antenna count")
            nrf_label.setText(f"{value} (fixed = Nt in full-digital)")

        tx_spin.valueChanged.connect(_update_tx_related)
        _update_tx_related(int(node.get_property("num_tx_antennas", 16)))

        tx_layout.addWidget(tx_spin)
        tx_layout.addWidget(tx_mode_label)
        self.form.addRow("Nt", tx_host)

        rx_spin = QSpinBox()
        rx_spin.setRange(1, 64)
        rx_spin.setValue(int(node.get_property("num_rx_antennas", 4)))
        rx_spin.valueChanged.connect(lambda value: self._set_property(node, "num_rx_antennas", value))
        self.form.addRow("Nr", rx_spin)

        ns_spin = QSpinBox()
        ns_spin.setRange(1, 8)
        ns_spin.setValue(self._recommended_streams_for_channel(node))
        ns_spin.valueChanged.connect(lambda value: self._set_all_precoder_streams(int(value)))
        self.form.addRow("Ns / user", ns_spin)

        self.form.addRow("Nrf", nrf_label)

        snr_host = QWidget()
        snr_layout = QVBoxLayout(snr_host)
        snr_layout.setContentsMargins(0, 0, 0, 0)
        snr_layout.setSpacing(4)
        snr_slider = QSlider(Qt.Horizontal)
        snr_slider.setRange(0, 50)
        snr_slider.setValue(int(round(float(node.get_property("snr_db", 10.0)))))
        snr_value = QLabel(f"{float(node.get_property('snr_db', 10.0)):.1f} dB")
        snr_value.setStyleSheet("color:#7dd3fc; font-weight:600;")

        def _preview_snr_change(value: int) -> None:
            snr_value.setText(f"{float(value):.1f} dB")

        def _apply_snr_change() -> None:
            self._set_property(node, "snr_db", float(snr_slider.value()))

        snr_slider.valueChanged.connect(_preview_snr_change)
        snr_slider.sliderReleased.connect(_apply_snr_change)
        snr_layout.addWidget(snr_slider)
        snr_layout.addWidget(snr_value)
        self.form.addRow("Total SNR", snr_host)

    def _add_snr_fields(self, node: BaseNode) -> None:
        for key, label, lo, hi, step in (
            ("snr_start", "SNR Start", -40.0, 80.0, 1.0),
            ("snr_end", "SNR End", -40.0, 80.0, 1.0),
            ("snr_step", "SNR Step", 0.1, 20.0, 0.5),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(float(node.get_property(key, 0.0)))
            spin.valueChanged.connect(lambda value, k=key: self._set_property(node, k, float(value)))
            self.form.addRow(label, spin)

        band_box = QComboBox()
        band_box.addItems(display_frequency_bands())
        band_box.setCurrentText(str(node.get_property("frequency_band", FrequencyBand.SUB6G.value)))
        band_box.currentTextChanged.connect(lambda value: self._set_property(node, "frequency_band", value))
        self.form.addRow("Band", band_box)

    def _add_user_config_fields(self, node: BaseNode) -> None:
        user_spin = QSpinBox()
        user_spin.setRange(0, 15)
        user_spin.setValue(int(node.get_property("user_index", 0)))
        user_spin.valueChanged.connect(lambda value: self._set_property(node, "user_index", int(value)))
        self.form.addRow("User Index", user_spin)

    def _set_tx_from_label(self, node: BaseNode, label: str) -> None:
        for key, config in XL_MIMO_CONFIG.items():
            if config["label"] == label:
                self._set_property(node, "num_tx_antennas", int(key))
                return

    def _set_property(self, node: BaseNode, key: str, value) -> None:
        node.set_property(key, value)
        self.on_change()

    def _recommended_streams_for_channel(self, node: ChannelNode) -> int:
        num_rx = int(node.get_property("num_rx_antennas", 4))
        return max(1, min(4, num_rx))

    def _set_all_precoder_streams(self, value: int) -> None:
        for graph_node in self.graph.nodes.values():
            if graph_node.NODE_CATEGORY == "Precoding":
                graph_node.set_property("num_streams", int(value))
        self.on_change()


class ResultPane(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Scope / Runtime")
        title.setStyleSheet("color: #e6f1ff; font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        self.scope_title = QLabel("Current Result: --")
        self.scope_title.setStyleSheet("color:#7dd3fc; font-size: 15px; font-weight: 700;")
        layout.addWidget(self.scope_title)

        self.binding_label = QLabel(f"Qt Binding: {QT_BINDING}")
        self.binding_label.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(self.binding_label)

        self.figure = Figure(figsize=(4.8, 3.2), facecolor=WORKBENCH_BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet(f"background:{WORKBENCH_BG}; border:1px solid #17304b;")
        self.visualizer = ChannelVisualizer(fig=self.figure, ax=self.figure.add_subplot(111))
        self.visualizer.draw()
        layout.addWidget(self.canvas, 2)

        self.status = QLabel("Ready. Use the palette to add nodes and drag connections.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(self.status)

        self.metrics = QLabel("SNR: -- | Band: -- | Diag Gains: --")
        self.metrics.setWordWrap(True)
        self.metrics.setStyleSheet("color:#dbeafe; background:#0a1626; padding:8px; border:1px solid #17304b;")
        layout.addWidget(self.metrics)

        self.mapping_label = QLabel("User-Method Map: --")
        self.mapping_label.setWordWrap(True)
        self.mapping_label.setStyleSheet("color:#f8fafc; background:#0f172a; padding:8px; border:1px solid #17304b;")
        layout.addWidget(self.mapping_label)

        self.legend = QLabel(
            "Legend: red node = Base Station, colored nodes = Users, solid bright beams = dominant LoS, "
            "soft dashed beams = side lobes / residual paths."
        )
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet("color:#9db7d7; background:#08111d; padding:8px; border:1px solid #17304b;")
        layout.addWidget(self.legend)

        self.user_table = QTableWidget(0, 5)
        self.user_table.setHorizontalHeaderLabels(["User", "Method", "Diag Gains", "Power Weights", "User Rate"])
        self.user_table.horizontalHeader().setStretchLastSection(True)
        self.user_table.setStyleSheet(
            "QTableWidget { background:#08111d; color:#dbeafe; gridline-color:#17304b; border:1px solid #17304b; }"
            "QHeaderView::section { background:#10233a; color:#c9e6ff; padding:4px; border:1px solid #17304b; }"
        )
        layout.addWidget(self.user_table, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "background:#08111d; color:#bfe8ff; border:1px solid #17304b; font-family: Consolas;"
        )
        layout.addWidget(self.log, 1)
        self.scope_payloads: dict[str, object] = {}
        self.active_scope_name: str | None = None

    def append(self, text: str) -> None:
        self.log.appendPlainText(text)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def remember_scope_payload(self, payload) -> None:
        scope_name = payload.metadata.get("scope_name")
        if scope_name:
            self.scope_payloads[str(scope_name)] = payload

    def update_visual_scene(self, node: BaseNode | None = None, payload=None, resolved_num_users: int | None = None) -> None:
        if payload is not None:
            self.remember_scope_payload(payload)
            self.scope_title.setText(f"Current Result: {payload.metadata.get('scope_name', payload.metadata.get('method', '--'))}")
            self.visualizer.update_from_payload(payload)
            self._update_metrics_from_payload(payload)
        elif isinstance(node, ChannelNode):
            self.scope_title.setText("Current Result: Channel Preview")
            band_text = str(node.get_property("frequency_band", FrequencyBand.SUB6G.value))
            num_users = int(resolved_num_users if resolved_num_users is not None else node.get_property("num_users", 2))
            num_tx = int(node.get_property("num_tx_antennas", 16))
            snr_db = float(node.get_property("snr_db", 10.0))
            scene = self.visualizer.build_scene(
                carrier_freq=FrequencyBand.from_value(band_text).frequency_hz,
                num_users=num_users,
                num_tx_antennas=num_tx,
                snr=snr_db,
                metadata={
                    "is_sub_thz": FrequencyBand.from_value(band_text).is_sub_thz,
                    "is_xl_mimo": num_tx >= 256,
                },
                stream_id="editor-preview",
            )
            self.visualizer.scene_state = scene
            self.metrics.setText(
                f"SNR: {snr_db:.1f} dB | Band: {band_text} | Users: {num_users} | Nt={num_tx}"
            )
            self.mapping_label.setText("User-Method Map: preview only, no decomposition selected yet.")
            self._fill_user_table(
                singular_values=[[] for _ in range(num_users)],
                power_weights=[[] for _ in range(num_users)],
                user_rates=[0.0 for _ in range(num_users)],
                methods=["--" for _ in range(num_users)],
                rhos=[[] for _ in range(num_users)],
            )
        elif node is not None and node.NODE_TYPE == "SNRGenerator":
            self.scope_title.setText("Current Result: SNR Generator")
            self.metrics.setText(
                f"SNR Sweep: {float(node.get_property('snr_start', 0.0)):.1f} -> "
                f"{float(node.get_property('snr_end', 40.0)):.1f} dB, "
                f"step {float(node.get_property('snr_step', 2.0)):.1f} dB | "
                f"Band: {node.get_property('frequency_band', FrequencyBand.SUB6G.value)}"
            )
            self.mapping_label.setText("User-Method Map: SNR generator is not a decomposition branch.")
        elif node is not None and node.NODE_TYPE == "UserConfig":
            self.scope_title.setText("Current Result: User Configuration")
            self.metrics.setText(
                f"Users are controlled by the number of User modules on the canvas.\n"
                f"Current active user count K = {resolved_num_users if resolved_num_users is not None else 1}."
            )
            self.mapping_label.setText("User-Method Map: choose a Scope to see which decomposition applies to each user.")
            self._fill_user_table(
                singular_values=[[] for _ in range(resolved_num_users or 1)],
                power_weights=[[] for _ in range(resolved_num_users or 1)],
                user_rates=[0.0 for _ in range(resolved_num_users or 1)],
                methods=["--" for _ in range(resolved_num_users or 1)],
                rhos=[[] for _ in range(resolved_num_users or 1)],
            )
        elif node is not None and node.NODE_TYPE == "Scope":
            self.active_scope_name = str(node.name)
            self.scope_title.setText(f"Current Result: {self.active_scope_name}")
            cached = self.scope_payloads.get(self.active_scope_name)
            if cached is not None:
                self.visualizer.update_from_payload(cached)
                self._update_metrics_from_payload(cached)
            else:
                self.metrics.setText(
                    f"Selected {node.name}. Run the graph to populate this scope with its matrix decomposition result."
                )
                self.mapping_label.setText("User-Method Map: pending result.")
        else:
            return
        self.visualizer.draw()
        self.canvas.draw_idle()

    def _update_metrics_from_payload(self, payload) -> None:
        singular_values = payload.metadata.get("scope_results", {}).get("latest_singular_values")
        if singular_values is None:
            singular_values = payload.metadata.get("diag_gains")
        sv_text = f"{len(singular_values)} users" if singular_values else "--"
        band_text = payload.metadata.get("frequency_band", FrequencyBand.from_value(payload.carrier_freq).value)
        allocation = payload.metadata.get("allocation")
        if allocation:
            rho_by_user = allocation.get("rho_by_user", [])
            weight_by_user = allocation.get("power_weights", [])
            sv_by_user = allocation.get("diag_gains", singular_values or [])
            user_rates = allocation.get("user_rates", [])
            sum_rate = allocation.get("sum_rate", 0.0)
            method = allocation.get("method", payload.metadata.get("method", "--"))
            self.metrics.setText(
                f"Scope: {payload.metadata.get('scope_name', '--')} | Total SNR: {float(payload.snr):.1f} dB | Band: {band_text} | Method: {method} | "
                f"Users: {len(sv_by_user)} | Sum-Rate: {float(sum_rate):.2f} b/s/Hz\n"
                f"Detailed per-user diagonal gains / rho / rate are listed in the table below."
            )
            method_map = allocation.get("per_user_methods", [])
            if method_map:
                mapping_text = " ; ".join(
                    f"User {user_idx + 1} -> {str(user_method).upper()}"
                    for user_idx, user_method in enumerate(method_map)
                )
            else:
                mapping_text = "--"
            self.mapping_label.setText(f"User-Method Map: {mapping_text}")
            self._fill_user_table(
                singular_values=sv_by_user,
                power_weights=weight_by_user,
                user_rates=user_rates,
                methods=allocation.get("per_user_methods", []),
                rhos=rho_by_user,
            )
            return

        self.metrics.setText(
            f"Total SNR: {float(payload.snr):.1f} dB | Band: {band_text} | "
            f"Users: {int(payload.num_users)} | Nt={int(payload.num_tx_antennas)}\n"
            f"Detailed per-user diagonal gains are listed in the table below."
        )
        self.mapping_label.setText("User-Method Map: --")
        self._fill_user_table(
            singular_values=singular_values or [],
            power_weights=[],
            user_rates=[],
            methods=[],
            rhos=[],
        )

    def _fill_user_table(self, *, singular_values, power_weights, user_rates, methods, rhos) -> None:
        if self.user_table.columnCount() != 6:
            self.user_table.setColumnCount(6)
            self.user_table.setHorizontalHeaderLabels(["User", "Method", "Diag Gains", "Power Weights", "rho", "User Rate"])
        num_rows = max(len(singular_values), len(power_weights), len(user_rates), len(methods), len(rhos))
        self.user_table.setRowCount(num_rows)
        for row in range(num_rows):
            sv_text = ", ".join(f"{float(v):.2f}" for v in (singular_values[row][:4] if row < len(singular_values) else []))
            pw_text = ", ".join(f"{float(v):.2f}" for v in (power_weights[row][:4] if row < len(power_weights) else []))
            rho_text = ", ".join(f"{float(v):.2f}" for v in (rhos[row][:4] if row < len(rhos) else []))
            rate_text = f"{float(user_rates[row]):.2f}" if row < len(user_rates) else "--"
            method_text = str(methods[row]).upper() if row < len(methods) else "--"
            self.user_table.setItem(row, 0, QTableWidgetItem(f"User {row + 1}"))
            self.user_table.setItem(row, 1, QTableWidgetItem(method_text))
            self.user_table.setItem(row, 2, QTableWidgetItem(sv_text or "--"))
            self.user_table.setItem(row, 3, QTableWidgetItem(pw_text or "--"))
            self.user_table.setItem(row, 4, QTableWidgetItem(rho_text or "--"))
            self.user_table.setItem(row, 5, QTableWidgetItem(rate_text))


class NodePalette(QWidget):
    def __init__(self, on_add, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.on_add = on_add
        self.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Node Palette")
        title.setStyleSheet("color: #e6f1ff; font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background:#08111d; color:#dbeafe; border:1px solid #17304b; }"
            "QListWidget::item:selected { background:#12314f; }"
        )
        for node_type in [
            "UserConfig",
            "ChannelGenerator",
            "Precoding_SVD",
            "Precoding_GMD",
            "Precoding_UCD",
            "Scope",
        ]:
            QListWidgetItem(node_type, self.list_widget)
        layout.addWidget(self.list_widget, 1)

        add_button = QPushButton("Add Selected Node")
        add_button.clicked.connect(self._add_selected)
        layout.addWidget(add_button)

    def _add_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item is not None:
            self.on_add(item.text())


class NodeGraphWorkbench(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"6G Simulink-Style Node Workbench ({QT_BINDING})")
        self.resize(1560, 940)
        self.setStyleSheet(f"QMainWindow {{ background:{WORKBENCH_BG}; color:{TEXT}; }}")

        self.graph = NodeGraph()
        self.scene = NodeScene(self.graph, self)
        self.scene.node_selected.connect(self._on_node_selected)
        self._stop_requested = False
        self._pending_preview_node: BaseNode | None = None
        self._pending_resolved_users: int | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._flush_preview_update)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(ANTIALIAS | TEXT_ANTIALIAS)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        self.view.setStyleSheet(f"background:{WORKBENCH_BG};")

        self.result_pane = ResultPane()
        self.properties = PropertiesPane(self.graph, on_change=self._sync_selected_node)
        self.palette = NodePalette(self._add_node)

        center = QSplitter(HORIZONTAL)
        center.addWidget(self.palette)
        center.addWidget(self.view)
        center.addWidget(self.properties)
        center.addWidget(self.result_pane)
        center.setSizes([220, 860, 260, 280])
        self.setCentralWidget(center)

        self._build_actions()

        self.edge_timer = QTimer(self)
        self.edge_timer.timeout.connect(self.scene.animate_edges)
        self.edge_timer.start(140)

        self._install_runtime_hooks()
        self._bootstrap_default_graph()
        self._select_default_channel_node()

    def _build_actions(self) -> None:
        toolbar = self.addToolBar("Workbench")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT};")

        run_action = QAction("Run Graph", self)
        run_action.triggered.connect(self._run_graph)
        toolbar.addAction(run_action)

        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self._stop_graph)
        toolbar.addAction(stop_action)

        clear_action = QAction("Clear Graph", self)
        clear_action.triggered.connect(self._clear_graph)
        toolbar.addAction(clear_action)

        add_channel = QAction("Add Channel", self)
        add_channel.triggered.connect(lambda: self._add_node("ChannelGenerator"))
        toolbar.addAction(add_channel)

        add_user_cfg = QAction("Add User", self)
        add_user_cfg.triggered.connect(lambda: self._add_node("UserConfig"))
        toolbar.addAction(add_user_cfg)

        remove_user_cfg = QAction("Remove User", self)
        remove_user_cfg.triggered.connect(self._remove_user_node)
        toolbar.addAction(remove_user_cfg)

        add_svd = QAction("Add SVD", self)
        add_svd.triggered.connect(lambda: self._add_node("Precoding_SVD"))
        toolbar.addAction(add_svd)

        add_gmd = QAction("Add GMD", self)
        add_gmd.triggered.connect(lambda: self._add_node("Precoding_GMD"))
        toolbar.addAction(add_gmd)

        add_ucd = QAction("Add UCD", self)
        add_ucd.triggered.connect(lambda: self._add_node("Precoding_UCD"))
        toolbar.addAction(add_ucd)

        add_scope = QAction("Add Scope", self)
        add_scope.triggered.connect(lambda: self._add_node("Scope"))
        toolbar.addAction(add_scope)

    def _install_runtime_hooks(self) -> None:
        self.graph.event_bus.subscribe("node_started", self._handle_runtime_event)
        self.graph.event_bus.subscribe("node_completed", self._handle_runtime_event)
        self.graph.event_bus.subscribe("node_failed", self._handle_runtime_event)
        self.graph.event_bus.subscribe("graph_edge_added", self._handle_runtime_event)

    def _handle_runtime_event(self, event) -> None:
        if event.topic == "node_started":
            self.result_pane.set_status(f"Running {event.detail.get('name', event.node_id)}")
        elif event.topic == "node_completed":
            summary = event.payload.summary() if event.payload is not None else {}
            self.result_pane.append(f"[DONE] {event.node_id}: {summary}")
            self.result_pane.set_status(f"Completed {event.detail.get('name', event.node_id)}")
            if event.payload is not None:
                self.result_pane.remember_scope_payload(event.payload)
                scope_name = event.payload.metadata.get("scope_name")
                if scope_name == self.result_pane.active_scope_name:
                    self.result_pane.update_visual_scene(payload=event.payload)
        elif event.topic == "node_failed":
            self.result_pane.append(f"[ERROR] {event.node_id}: {event.detail.get('error')}")
            self.result_pane.set_status(f"Failed {event.detail.get('name', event.node_id)}")
        elif event.topic == "graph_edge_added":
            self.result_pane.append(f"[LINK] {event.detail.get('source')} -> {event.detail.get('target')}")

    def _bootstrap_default_graph(self) -> None:
        positions = {
            "User1": QPointF(20.0, 100.0),
            "User2": QPointF(20.0, 220.0),
            "User3": QPointF(20.0, 340.0),
            "ChannelSVD": QPointF(170.0, 70.0),
            "ChannelGMD": QPointF(170.0, 240.0),
            "ChannelUCD": QPointF(170.0, 410.0),
            "Precoding_SVD": QPointF(420.0, 70.0),
            "Precoding_GMD": QPointF(420.0, 240.0),
            "Precoding_UCD": QPointF(420.0, 410.0),
            "Scope_SVD": QPointF(790.0, 70.0),
            "Scope_GMD": QPointF(790.0, 240.0),
            "Scope_UCD": QPointF(790.0, 410.0),
        }
        self._add_node("UserConfig", positions["User1"])
        self._add_node("UserConfig", positions["User2"])
        self._add_node("UserConfig", positions["User3"])
        self._add_node("ChannelGenerator", positions["ChannelSVD"], custom_name="Channel SVD")
        self._add_node("ChannelGenerator", positions["ChannelGMD"], custom_name="Channel GMD")
        self._add_node("ChannelGenerator", positions["ChannelUCD"], custom_name="Channel UCD")
        self._add_node("Precoding_SVD", positions["Precoding_SVD"])
        self._add_node("Precoding_GMD", positions["Precoding_GMD"])
        self._add_node("Precoding_UCD", positions["Precoding_UCD"])
        self._add_node("Scope", positions["Scope_SVD"], custom_name="Scope SVD")
        self._add_node("Scope", positions["Scope_GMD"], custom_name="Scope GMD")
        self._add_node("Scope", positions["Scope_UCD"], custom_name="Scope UCD")
        self._auto_wire_default_graph()

    def _select_default_channel_node(self) -> None:
        channel_node = next((node for node in self.graph.nodes.values() if isinstance(node, ChannelNode)), None)
        if channel_node is None:
            return
        item = self.scene.node_items.get(channel_node.node_id)
        if item is None:
            return
        item.setSelected(True)
        self.scene.node_selected.emit(channel_node)

    def _add_node(self, node_type: str, position: QPointF | None = None, custom_name: str | None = None) -> None:
        node_id = f"node_{len(self.graph.nodes) + 1}"
        node = create_node(node_type, node_id, custom_name)
        if node_type == "UserConfig":
            user_nodes = [existing for existing in self.graph.nodes.values() if existing.NODE_TYPE == "UserConfig"]
            node.set_property("user_index", len(user_nodes))
            if not custom_name:
                node.name = f"User {len(user_nodes) + 1}"
        pos = position or QPointF(120.0 + len(self.graph.nodes) * 22.0, 120.0 + len(self.graph.nodes) * 18.0)
        self.scene.add_runtime_node(node, pos)
        self.result_pane.append(f"[ADD] {node.NODE_TYPE} @ ({pos.x():.0f}, {pos.y():.0f})")
        if node_type == "UserConfig":
            self._wire_branch_graph()
            channel_node = next((graph_node for graph_node in self.graph.nodes.values() if isinstance(graph_node, ChannelNode)), None)
            if channel_node is not None:
                resolved_num_users = self._resolved_num_users()
                self._schedule_preview_update(channel_node, resolved_num_users)

    def _remove_user_node(self) -> None:
        user_nodes = [node for node in self.graph.nodes.values() if node.NODE_TYPE == "UserConfig"]
        if len(user_nodes) <= 1:
            self.result_pane.append("[SKIP] At least one User module must remain.")
            self.result_pane.set_status("Cannot remove the last User module.")
            return
        user_to_remove = sorted(user_nodes, key=lambda node: int(node.get_property("user_index", 0)))[-1]
        self.scene.removeItem(self.scene.node_items[user_to_remove.node_id])
        self.scene.node_items.pop(user_to_remove.node_id, None)
        self.graph.remove_node(user_to_remove.node_id)
        self.result_pane.append(f"[REMOVE] {user_to_remove.name}")
        self.result_pane.set_status(f"Removed {user_to_remove.name}")
        self._renumber_user_nodes()
        self._wire_branch_graph()
        channel_node = next((node for node in self.graph.nodes.values() if isinstance(node, ChannelNode)), None)
        if channel_node is not None:
            resolved_num_users = self._resolved_num_users()
            self._schedule_preview_update(channel_node, resolved_num_users)

    def _renumber_user_nodes(self) -> None:
        user_nodes = sorted(
            [node for node in self.graph.nodes.values() if node.NODE_TYPE == "UserConfig"],
            key=lambda node: int(node.get_property("user_index", 0)),
        )
        for idx, node in enumerate(user_nodes):
            node.set_property("user_index", idx)
            node.name = f"User {idx + 1}"

    def _on_node_selected(self, node: BaseNode) -> None:
        self.properties.set_node(node)
        resolved_num_users = self._resolved_num_users()
        self._schedule_preview_update(node, resolved_num_users)

    def _sync_selected_node(self) -> None:
        node = self.properties.node
        if node is None:
            return
        if isinstance(node, ChannelNode):
            self._harmonize_precoding_nodes(node)
        self.result_pane.append(f"[EDIT] {node.name} -> {node.properties}")
        self.properties.set_node(node)
        resolved_num_users = self._resolved_num_users()
        self._schedule_preview_update(node, resolved_num_users)

    def _run_graph(self) -> None:
        try:
            self._stop_requested = False
            self._sync_user_into_channel()
            validation_error = self._validate_graph()
            if validation_error is not None:
                QMessageBox.warning(self, "Invalid Parameters", validation_error)
                self.result_pane.append(f"[INVALID] {validation_error}")
                self.result_pane.set_status("Parameter validation failed.")
                return
            self.graph.reset()
            self.scene.refresh_edges()
            if self._stop_requested:
                self.result_pane.set_status("Run stopped.")
                return
            branch_node_ids = self._selected_branch_node_ids()
            selected_name = self.properties.node.name if self.properties.node is not None else "all branches"
            self.result_pane.set_status(f"Running selected branch around: {selected_name}")
            for node in self.graph.topological_order():
                if node.node_id in branch_node_ids:
                    node.run()
        except Exception as exc:
            QMessageBox.critical(self, "Run Failed", str(exc))
            self.result_pane.append(f"[FATAL] {exc}")

    def _schedule_preview_update(self, node: BaseNode, resolved_num_users: int | None) -> None:
        self._pending_preview_node = node
        self._pending_resolved_users = resolved_num_users
        self._preview_timer.start(60)

    def _flush_preview_update(self) -> None:
        if self._pending_preview_node is None:
            return
        self.result_pane.update_visual_scene(
            node=self._pending_preview_node,
            resolved_num_users=self._pending_resolved_users,
        )
        self._pending_preview_node = None
        self._pending_resolved_users = None

    def _selected_branch_node_ids(self) -> set[str]:
        selected_node = self.properties.node
        if selected_node is None:
            return set(self.graph.nodes.keys())
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in self.graph.nodes}
        for edge in self.graph.edges:
            adjacency[edge.source_node_id].add(edge.target_node_id)
            adjacency[edge.target_node_id].add(edge.source_node_id)
        visited = {selected_node.node_id}
        queue = [selected_node.node_id]
        while queue:
            current = queue.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _stop_graph(self) -> None:
        self._stop_requested = True
        self.result_pane.set_status("Stop requested.")
        self.result_pane.append("[STOP] Execution stop requested.")

    def _clear_graph(self) -> None:
        self.scene.clear()
        self.graph = NodeGraph()
        self.scene.graph = self.graph
        self.scene.node_items.clear()
        self.scene.edge_items.clear()
        self.properties.set_node(None)
        self.result_pane.set_status("Graph cleared.")
        self.result_pane.append("[RESET] Cleared graph.")
        self._install_runtime_hooks()

    def _validate_graph(self) -> str | None:
        user_cfg_nodes = [node for node in self.graph.nodes.values() if node.NODE_TYPE == "UserConfig"]
        channel_node = next((node for node in self.graph.nodes.values() if isinstance(node, ChannelNode)), None)
        resolved_num_users = len(user_cfg_nodes) if user_cfg_nodes else None
        if channel_node is not None:
            num_users = resolved_num_users if resolved_num_users is not None else int(channel_node.get_property("num_users", 2))
            num_tx = int(channel_node.get_property("num_tx_antennas", 16))
            num_rx = int(channel_node.get_property("num_rx_antennas", 4))
            snr_db = float(channel_node.get_property("snr_db", 10.0))
            if num_users <= 0:
                return "Number of users must be positive."
            if num_tx <= 0 or num_rx <= 0:
                return "Antenna counts must be positive."
            if num_tx < num_users:
                return f"Nt={num_tx} is too small for K={num_users}; increase Nt or reduce users."
            if snr_db < 0.0 or snr_db > 50.0:
                return f"Total SNR={snr_db:.1f} dB must stay within 0 to 50 dB."
            if num_tx % num_users != 0:
                valid_values = [str(value) for value in range(1, num_tx + 1) if num_tx % value == 0]
                return (
                    f"Current full-digital implementation requires Nt % K == 0. "
                    f"Got Nt={num_tx}, K={num_users}. Valid K for Nt={num_tx}: {', '.join(valid_values)}."
                )

        for node in self.graph.nodes.values():
            if node.NODE_CATEGORY == "Precoding":
                num_streams = int(node.properties.get("num_streams", 4))
                if num_streams <= 0:
                    return f"{node.name}: number of streams must be positive."
                if channel_node is not None and num_streams > int(channel_node.get_property("num_rx_antennas", 4)):
                    return (
                        f"{node.name}: Ns/user={num_streams} exceeds Nr="
                        f"{int(channel_node.get_property('num_rx_antennas', 4))}. "
                        "Reduce Ns/user or increase Nr."
                    )
                if channel_node is not None and resolved_num_users is not None:
                    num_tx = int(channel_node.get_property("num_tx_antennas", 16))
                    total_streams = resolved_num_users * num_streams
                    rf_per_user = num_tx // max(resolved_num_users, 1)
                    if total_streams > num_tx:
                        return (
                            f"{node.name}: total streams K*Ns={total_streams} exceed Nt={num_tx}. "
                            "Reduce Users or Ns/user, or increase Nt."
                        )
                    if rf_per_user < num_streams:
                        return (
                            f"{node.name}: per-user RF/space budget Nt/K={rf_per_user} is smaller than Ns/user={num_streams}. "
                            "Increase Nt, reduce Users, or reduce Ns/user."
                        )

        for node in self.graph.nodes.values():
            if node.NODE_TYPE == "SNRGenerator":
                snr_start = float(node.get_property("snr_start", 0.0))
                snr_end = float(node.get_property("snr_end", 40.0))
                snr_step = float(node.get_property("snr_step", 2.0))
                if snr_step <= 0.0:
                    return "SNR step must be positive."
                if snr_end < snr_start:
                    return "SNR end must be greater than or equal to SNR start."
        return None

    def _harmonize_precoding_nodes(self, channel_node: ChannelNode) -> None:
        num_rx = int(channel_node.get_property("num_rx_antennas", 4))
        recommended_streams = max(1, min(4, num_rx))
        for node in self.graph.nodes.values():
            if node.NODE_CATEGORY == "Precoding":
                current = int(node.properties.get("num_streams", recommended_streams))
                if current <= 0 or current > num_rx:
                    node.set_property("num_streams", recommended_streams)

    def _resolved_num_users(self) -> int:
        user_cfg_nodes = [node for node in self.graph.nodes.values() if node.NODE_TYPE == "UserConfig"]
        if user_cfg_nodes:
            return len(user_cfg_nodes)
        channel_node = next((node for node in self.graph.nodes.values() if isinstance(node, ChannelNode)), None)
        if channel_node is not None:
            return int(channel_node.get_property("num_users", 2))
        return 2

    def _sync_user_into_channel(self) -> None:
        channels = [node for node in self.graph.nodes.values() if isinstance(node, ChannelNode)]
        for channel in channels:
            channel.set_property("num_users", 1)

    def _auto_wire_default_graph(self) -> None:
        self._wire_branch_graph()

    def _wire_branch_graph(self) -> None:
        try:
            user_cfg_nodes = sorted(
                [node for node in self.graph.nodes.values() if node.NODE_TYPE == "UserConfig"],
                key=lambda node: int(node.get_property("user_index", 0)),
            )
            channel_svd = next(node for node in self.graph.nodes.values() if node.name == "Channel SVD")
            channel_gmd = next(node for node in self.graph.nodes.values() if node.name == "Channel GMD")
            channel_ucd = next(node for node in self.graph.nodes.values() if node.name == "Channel UCD")
            svd = next(node for node in self.graph.nodes.values() if node.NODE_TYPE == "Precoding_SVD")
            gmd = next(node for node in self.graph.nodes.values() if node.NODE_TYPE == "Precoding_GMD")
            ucd = next(node for node in self.graph.nodes.values() if node.NODE_TYPE == "Precoding_UCD")
            scope_svd = next(node for node in self.graph.nodes.values() if node.name == "Scope SVD")
            scope_gmd = next(node for node in self.graph.nodes.values() if node.name == "Scope GMD")
            scope_ucd = next(node for node in self.graph.nodes.values() if node.name == "Scope UCD")

            branch_pairs = [
                (user_cfg_nodes[0], channel_svd, svd, scope_svd),
                (user_cfg_nodes[1], channel_gmd, gmd, scope_gmd),
                (user_cfg_nodes[2], channel_ucd, ucd, scope_ucd),
            ]

            for edge in list(self.graph.edges):
                source_node = self.graph.nodes.get(edge.source_node_id)
                target_node = self.graph.nodes.get(edge.target_node_id)
                if source_node is None or target_node is None:
                    continue
                source_port = source_node.get_output(edge.source_port_name)
                target_port = target_node.get_input(edge.target_port_name)
                if source_port is not None and target_port is not None:
                    self.graph.disconnect(source_port, target_port)
            for edge_item in list(self.scene.edge_items.values()):
                self.scene.removeItem(edge_item)
            self.scene.edge_items.clear()

            for user_cfg, channel, decomp, scope in branch_pairs:
                self.graph.connect(user_cfg.outputs[0], channel.inputs[0])
                self.graph.connect(channel.outputs[0], decomp.inputs[0])
                self.graph.connect(decomp.outputs[0], scope.inputs[0])
                self.scene._create_edge_graphic(user_cfg.outputs[0], channel.inputs[0])
                self.scene._create_edge_graphic(channel.outputs[0], decomp.inputs[0])
                self.scene._create_edge_graphic(decomp.outputs[0], scope.inputs[0])
            self.scene.refresh_edges()
        except Exception:
            pass


def run_qt_node_workbench() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = NodeGraphWorkbench()
    window.show()
    return app.exec()
