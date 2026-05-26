from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, DefaultDict, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence
import time

import numpy as np


class PortType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class NodeExecutionState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(slots=True)
class TensorPayload:
    """
    Runtime payload exchanged between DAG nodes.

    The payload intentionally carries both dense tensor data and 6G-oriented
    control metadata so the graph runtime can remain UI-agnostic.
    """

    data: np.ndarray | Sequence[float] | Sequence[complex] | float | int
    carrier_freq: float = 3.5e9
    snr: float = 10.0
    num_users: int = 2
    num_tx_antennas: int = 16
    num_rx_antennas: int = 4
    sample_rate: float | None = None
    bandwidth: float | None = None
    frame_index: int = 0
    stream_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def is_sub_thz(self) -> bool:
        return float(self.carrier_freq) >= 100e9

    @property
    def is_xl_mimo(self) -> bool:
        return int(self.num_tx_antennas) >= 256

    @property
    def is_sensing_enabled(self) -> bool:
        return bool(self.metadata.get("isac_enabled", False))

    def clone(self, **updates: Any) -> TensorPayload:
        copied_data = self.array.copy()
        copied_meta = dict(self.metadata)
        payload = replace(self, data=copied_data, metadata=copied_meta)
        return replace(payload, **updates) if updates else payload

    def with_data(self, data: Any, **metadata_updates: Any) -> TensorPayload:
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata_updates.pop("metadata", {}))
        return self.clone(data=data, metadata=merged_metadata, **metadata_updates)

    def attach_metadata(self, **metadata_updates: Any) -> TensorPayload:
        merged = dict(self.metadata)
        merged.update(metadata_updates)
        return self.clone(metadata=merged)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "carrier_freq": float(self.carrier_freq),
            "snr": float(self.snr),
            "num_users": int(self.num_users),
            "num_tx_antennas": int(self.num_tx_antennas),
            "num_rx_antennas": int(self.num_rx_antennas),
            "is_sub_thz": self.is_sub_thz,
            "is_xl_mimo": self.is_xl_mimo,
            "stream_id": self.stream_id,
            "frame_index": int(self.frame_index),
        }


@dataclass(slots=True, frozen=True)
class GraphEvent:
    topic: str
    node_id: str | None = None
    port_name: str | None = None
    payload: TensorPayload | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """
    Lightweight observer hub used by the runtime and, later, any Qt/Tk UI.
    """

    def __init__(self) -> None:
        self._listeners: DefaultDict[str, list[Callable[[GraphEvent], None]]] = defaultdict(list)

    def subscribe(self, topic: str, callback: Callable[[GraphEvent], None]) -> None:
        if callback not in self._listeners[topic]:
            self._listeners[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[GraphEvent], None]) -> None:
        listeners = self._listeners.get(topic)
        if not listeners:
            return
        if callback in listeners:
            listeners.remove(callback)
        if not listeners:
            self._listeners.pop(topic, None)

    def emit(self, topic: str, **event_kwargs: Any) -> GraphEvent:
        event = GraphEvent(topic=topic, **event_kwargs)
        callbacks = list(self._listeners.get(topic, []))
        callbacks.extend(self._listeners.get("*", []))
        for callback in callbacks:
            callback(event)
        return event


@dataclass(slots=True, frozen=True)
class Edge:
    source_node_id: str
    source_port_name: str
    target_node_id: str
    target_port_name: str


class GraphCycleError(RuntimeError):
    pass


class NodePort:
    def __init__(
        self,
        node: BaseNode,
        name: str,
        port_type: PortType,
        index: int,
        payload_type: type = TensorPayload,
    ) -> None:
        self.node = node
        self.name = name
        self.port_type = port_type
        self.index = index
        self.payload_type = payload_type
        self.connected_ports: list[NodePort] = []
        self.data: TensorPayload | None = None
        self.data_listeners: list[Callable[[TensorPayload], None]] = []

    @property
    def key(self) -> str:
        return f"{self.node.node_id}:{self.port_type.value}:{self.name}"

    def connect(self, other_port: NodePort) -> bool:
        if self.port_type != PortType.OUTPUT or other_port.port_type != PortType.INPUT:
            return False
        if other_port in self.connected_ports:
            return False
        self.node._graph_connect_ports(self, other_port)
        return True

    def disconnect(self, other_port: NodePort) -> None:
        self.node._graph_disconnect_ports(self, other_port)

    def clear(self) -> None:
        self.data = None

    def send(self, payload: TensorPayload) -> None:
        self.data = payload
        for listener in list(self.data_listeners):
            listener(payload)
        for port in list(self.connected_ports):
            port.receive(payload)
        self.node.event_bus.emit(
            "port_data_sent",
            node_id=self.node.node_id,
            port_name=self.name,
            payload=payload,
            detail={"port_type": self.port_type.value},
        )

    def receive(self, payload: TensorPayload) -> None:
        self.data = payload
        for listener in list(self.data_listeners):
            listener(payload)
        self.node.event_bus.emit(
            "port_data_received",
            node_id=self.node.node_id,
            port_name=self.name,
            payload=payload,
            detail={"port_type": self.port_type.value},
        )
        self.node.on_input_received(self.index, payload)


Port = NodePort


class BaseNode:
    NODE_TYPE = "BaseNode"
    NODE_CATEGORY = "General"
    NODE_COLOR = "#6366f1"

    def __init__(self, node_id: str, name: str | None = None) -> None:
        self.node_id = node_id
        self.name = name or f"{self.NODE_TYPE}_{node_id}"
        self.properties: dict[str, Any] = {}
        self.inputs: list[NodePort] = []
        self.outputs: list[NodePort] = []
        self.position: tuple[float, float] = (0.0, 0.0)
        self.state = NodeExecutionState.IDLE
        self.running = False
        self.error: str | None = None
        self.last_inputs: list[TensorPayload] = []
        self.last_output: TensorPayload | None = None
        self.graph: NodeGraph | None = None
        self.event_bus = EventBus()

    def add_input(self, name: str, payload_type: type = TensorPayload) -> NodePort:
        port = NodePort(self, name, PortType.INPUT, len(self.inputs), payload_type)
        self.inputs.append(port)
        return port

    def add_output(self, name: str, payload_type: type = TensorPayload) -> NodePort:
        port = NodePort(self, name, PortType.OUTPUT, len(self.outputs), payload_type)
        self.outputs.append(port)
        return port

    def get_input(self, name_or_idx: str | int) -> NodePort | None:
        if isinstance(name_or_idx, int):
            return self.inputs[name_or_idx] if 0 <= name_or_idx < len(self.inputs) else None
        for port in self.inputs:
            if port.name == name_or_idx:
                return port
        return None

    def get_output(self, name_or_idx: str | int) -> NodePort | None:
        if isinstance(name_or_idx, int):
            return self.outputs[name_or_idx] if 0 <= name_or_idx < len(self.outputs) else None
        for port in self.outputs:
            if port.name == name_or_idx:
                return port
        return None

    def input_payloads(self) -> list[TensorPayload]:
        return [port.data for port in self.inputs if port.data is not None]

    def ready(self) -> bool:
        return not self.inputs or all(port.data is not None for port in self.inputs)

    def clear_inputs(self) -> None:
        for port in self.inputs:
            port.clear()

    def clear_outputs(self) -> None:
        for port in self.outputs:
            port.clear()

    def reset(self) -> None:
        self.running = False
        self.error = None
        self.state = NodeExecutionState.IDLE
        self.last_inputs = []
        self.last_output = None
        self.clear_inputs()
        self.clear_outputs()

    def on_input_received(self, port_idx: int, payload: TensorPayload) -> None:
        self.state = NodeExecutionState.READY if self.ready() else NodeExecutionState.IDLE
        self.event_bus.emit(
            "node_input_updated",
            node_id=self.node_id,
            payload=payload,
            detail={"port_idx": port_idx, "ready": self.ready()},
        )
        if self.ready():
            self.run()

    def run(self) -> TensorPayload | None:
        if self.running:
            return None
        if not self.ready():
            return None

        self.running = True
        self.error = None
        self.state = NodeExecutionState.RUNNING
        self.last_inputs = self.input_payloads()
        self.event_bus.emit("node_started", node_id=self.node_id, detail={"name": self.name})

        try:
            result = self.process(self.last_inputs)
            if result is None:
                raise ValueError(f"{self.name} returned no TensorPayload.")
            self.last_output = result
            for port in self.outputs:
                port.send(result)
            self.state = NodeExecutionState.COMPLETED
            self.event_bus.emit("node_completed", node_id=self.node_id, payload=result, detail={"name": self.name})
            return result
        except Exception as exc:
            self.error = str(exc)
            self.state = NodeExecutionState.ERROR
            self.event_bus.emit(
                "node_failed",
                node_id=self.node_id,
                detail={"name": self.name, "error": self.error},
            )
            raise
        finally:
            self.running = False

    def process(self, inputs: list[TensorPayload]) -> TensorPayload:
        raise NotImplementedError("Subclasses must implement process().")

    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def iter_ports(self) -> Iterator[NodePort]:
        yield from self.inputs
        yield from self.outputs

    def _graph_connect_ports(self, source_port: NodePort, target_port: NodePort) -> None:
        if self.graph is not None:
            self.graph.connect(source_port, target_port)
            return
        if target_port not in source_port.connected_ports:
            source_port.connected_ports.append(target_port)
        if source_port not in target_port.connected_ports:
            target_port.connected_ports.append(source_port)

    def _graph_disconnect_ports(self, source_port: NodePort, target_port: NodePort) -> None:
        if self.graph is not None:
            self.graph.disconnect(source_port, target_port)
            return
        if target_port in source_port.connected_ports:
            source_port.connected_ports.remove(target_port)
        if source_port in target_port.connected_ports:
            target_port.connected_ports.remove(source_port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.NODE_TYPE,
            "name": self.name,
            "position": self.position,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BaseNode:
        node = cls(str(data["node_id"]), data.get("name"))
        node.position = tuple(data.get("position", (0.0, 0.0)))
        node.properties = dict(data.get("properties", {}))
        return node


class NodeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, BaseNode] = {}
        self.edges: list[Edge] = []
        self.event_bus = EventBus()

    def add_node(self, node: BaseNode) -> BaseNode:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id={node.node_id!r}.")
        node.graph = self
        node.event_bus = self.event_bus
        self.nodes[node.node_id] = node
        self.event_bus.emit("graph_node_added", node_id=node.node_id, detail={"node_type": node.NODE_TYPE})
        return node

    def remove_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id)
        self.edges = [
            edge
            for edge in self.edges
            if edge.source_node_id != node_id and edge.target_node_id != node_id
        ]
        for port in node.iter_ports():
            for connected in list(port.connected_ports):
                if port in connected.connected_ports:
                    connected.connected_ports.remove(port)
                if connected in port.connected_ports:
                    port.connected_ports.remove(connected)
        node.graph = None
        self.event_bus.emit("graph_node_removed", node_id=node_id)

    def connect(self, source_port: NodePort, target_port: NodePort) -> Edge:
        if source_port.port_type != PortType.OUTPUT or target_port.port_type != PortType.INPUT:
            raise ValueError("Graph connections must be OUTPUT -> INPUT.")
        if target_port.connected_ports:
            raise ValueError(
                f"Input port '{target_port.key}' already has a connection. "
                "Single-input monitor nodes like Scope accept only one upstream source."
            )
        edge = Edge(
            source_node_id=source_port.node.node_id,
            source_port_name=source_port.name,
            target_node_id=target_port.node.node_id,
            target_port_name=target_port.name,
        )
        if edge in self.edges:
            return edge
        self._validate_no_cycle(edge)
        self.edges.append(edge)
        if target_port not in source_port.connected_ports:
            source_port.connected_ports.append(target_port)
        if source_port not in target_port.connected_ports:
            target_port.connected_ports.append(source_port)
        self.event_bus.emit(
            "graph_edge_added",
            node_id=source_port.node.node_id,
            detail={
                "source": source_port.key,
                "target": target_port.key,
            },
        )
        return edge

    def disconnect(self, source_port: NodePort, target_port: NodePort) -> None:
        self.edges = [
            edge
            for edge in self.edges
            if not (
                edge.source_node_id == source_port.node.node_id
                and edge.source_port_name == source_port.name
                and edge.target_node_id == target_port.node.node_id
                and edge.target_port_name == target_port.name
            )
        ]
        if target_port in source_port.connected_ports:
            source_port.connected_ports.remove(target_port)
        if source_port in target_port.connected_ports:
            target_port.connected_ports.remove(source_port)
        self.event_bus.emit(
            "graph_edge_removed",
            node_id=source_port.node.node_id,
            detail={"source": source_port.key, "target": target_port.key},
        )

    def sources(self) -> list[BaseNode]:
        targets = {edge.target_node_id for edge in self.edges}
        return [node for node_id, node in self.nodes.items() if node_id not in targets]

    def topological_order(self) -> list[BaseNode]:
        incoming_count: DefaultDict[str, int] = defaultdict(int)
        outgoing: DefaultDict[str, list[str]] = defaultdict(list)
        for node_id in self.nodes:
            incoming_count[node_id] = 0
        for edge in self.edges:
            outgoing[edge.source_node_id].append(edge.target_node_id)
            incoming_count[edge.target_node_id] += 1

        queue = deque(sorted((node_id for node_id, count in incoming_count.items() if count == 0)))
        ordered: list[BaseNode] = []
        while queue:
            node_id = queue.popleft()
            ordered.append(self.nodes[node_id])
            for target_id in outgoing[node_id]:
                incoming_count[target_id] -= 1
                if incoming_count[target_id] == 0:
                    queue.append(target_id)

        if len(ordered) != len(self.nodes):
            raise GraphCycleError("The node graph contains a cycle.")
        return ordered

    def run(self) -> dict[str, TensorPayload | None]:
        results: dict[str, TensorPayload | None] = {}
        for node in self.topological_order():
            if not node.inputs:
                results[node.node_id] = node.run()
                continue
            if node.ready():
                results[node.node_id] = node.run()
            else:
                results[node.node_id] = None
        return results

    def reset(self) -> None:
        for node in self.nodes.values():
            node.reset()
        self.event_bus.emit("graph_reset")

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [
                {
                    "source_node_id": edge.source_node_id,
                    "source_port_name": edge.source_port_name,
                    "target_node_id": edge.target_node_id,
                    "target_port_name": edge.target_port_name,
                }
                for edge in self.edges
            ],
        }

    def _validate_no_cycle(self, candidate_edge: Edge) -> None:
        adjacency: DefaultDict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source_node_id].append(edge.target_node_id)
        adjacency[candidate_edge.source_node_id].append(candidate_edge.target_node_id)

        start = candidate_edge.target_node_id
        target = candidate_edge.source_node_id
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            if current == target:
                raise GraphCycleError(
                    f"Connecting {candidate_edge.source_node_id} -> {candidate_edge.target_node_id} would create a cycle."
                )
            for nxt in adjacency[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)


_NODE_REGISTRY: dict[str, type[BaseNode]] = {}


def register_node(node_class: type[BaseNode]) -> type[BaseNode]:
    node_type = getattr(node_class, "NODE_TYPE", None)
    if not node_type:
        raise ValueError("Registered node class must define NODE_TYPE.")
    _NODE_REGISTRY[str(node_type)] = node_class
    return node_class


def create_node(node_type: str, node_id: str, name: str | None = None) -> BaseNode:
    node_class = _NODE_REGISTRY.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown node type: {node_type}")
    return node_class(node_id, name)


def get_registered_node_types() -> list[str]:
    return list(_NODE_REGISTRY.keys())
