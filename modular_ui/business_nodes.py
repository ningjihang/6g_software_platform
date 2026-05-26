"""
Step 4: 业务节点类 (ChannelNode, PrecodingNode, ScopeNode)
"""
from .node_graph_core import (
    BaseNode, register_node, TensorPayload
)
from .config_6g import FrequencyBand, resolve_6g_channel_profile
import numpy as np
import sys
import os
from typing import Sequence

# 把父目录加到路径中，以便导入 classical / full_digital_mu
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLASSICAL_DIR = os.path.join(_REPO_ROOT, "classical")
_FULL_DIGITAL_DIR = os.path.join(_REPO_ROOT, "full_digital_mu")
for _path in (_REPO_ROOT, _CLASSICAL_DIR, _FULL_DIGITAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from classical.sic_sample_average import build_multiuser_sample_average
except ImportError:
    from sic_sample_average import build_multiuser_sample_average

try:
    from classical.channel_model import ChannelModel
except ImportError:
    from channel_model import ChannelModel

try:
    from full_digital_mu.fd_mu_environment import FullyDigitalMuMimoBicmEnvironment
    HAS_FULL_DIGITAL_ENV = True
except ImportError:
    try:
        from fd_mu_environment import FullyDigitalMuMimoBicmEnvironment
        HAS_FULL_DIGITAL_ENV = True
    except ImportError:
        FullyDigitalMuMimoBicmEnvironment = None
        HAS_FULL_DIGITAL_ENV = False


def _evaluate_full_digital_method(
    payload: TensorPayload,
    method: str | Sequence[str],
    num_streams_per_user: int,
) -> tuple[np.ndarray, dict]:
    H = np.asarray(payload.data, dtype=complex)
    num_users, num_rx, num_tx = H.shape
    resolved_streams = min(int(num_streams_per_user), num_rx)
    if resolved_streams <= 0:
        raise ValueError("num_streams_per_user must be positive.")

    if not HAS_FULL_DIGITAL_ENV:
        raise RuntimeError("full_digital_mu environment is unavailable in the current runtime.")

    env = FullyDigitalMuMimoBicmEnvironment(
        num_users=num_users,
        num_tx_antennas=num_tx,
        num_rx_antennas=num_rx,
        num_streams_per_user=resolved_streams,
        channel_type=str(payload.metadata.get("channel_model", "cdl-a")),
        digital_power_constraint=float(num_users * resolved_streams),
        ucd_waterfill=True,
        ucd_min_power_loading=0.0,
    )
    total_snr_linear = 10 ** (float(payload.snr) / 10.0)
    snr_per_stream = total_snr_linear / env.total_streams
    if isinstance(method, str):
        per_user_methods = [str(method).lower()] * num_users
        chain = env.build_structured_chain(
            user_channels=H,
            snr_per_stream=snr_per_stream,
            strategy=str(method).lower(),
        )
        evaluation_mode = "ucd_b_chain" if str(method).lower() == "ucd" else "fixed_chain"
    else:
        per_user_methods = [str(item).lower() for item in method]
        if len(per_user_methods) < num_users:
            per_user_methods.extend([per_user_methods[-1]] * (num_users - len(per_user_methods)))
        per_user_methods = per_user_methods[:num_users]
        chain = _build_mixed_structured_chain(
            env=env,
            user_channels=H,
            snr_per_stream=snr_per_stream,
            per_user_methods=per_user_methods,
        )
        evaluation_mode = "mixed_fixed_chain"
    sample_average = build_multiuser_sample_average(
        env=env,
        bits_per_symbol=6,
        num_samples=64,
        num_repeats=1,
        base_seed=20260526,
        labeling="nr_like",
    )

    if evaluation_mode == "ucd_b_chain":
        evaluation = env.evaluate_ucd_precoder_current_receiver_average_b_chain(
            user_channels=H,
            f=chain.f_bb,
            q_chains=chain.q_chains,
            r_chains=chain.r_chains,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=6,
            sample_average=sample_average,
            labeling="nr_like",
        )
    else:
        evaluation = env.evaluate_precoder_current_receiver_average_fixed_chain(
            user_channels=H,
            f=chain.f_bb,
            r_chains=chain.r_chains,
            q_chains=chain.q_chains,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=6,
            sample_average=sample_average,
            labeling="nr_like",
        )

    diag_gains_by_user = []
    for user_idx in range(num_users):
        if evaluation_mode == "ucd_b_chain":
            diagonal_values = np.abs(np.diag(chain.r_chains[user_idx]))[:resolved_streams]
        else:
            diagonal_values = np.abs(np.diag(chain.r_chains[user_idx]))[:resolved_streams]
        diag_gains_by_user.append(np.asarray(diagonal_values, dtype=float).tolist())

    power_weights_by_user = []
    for rho_values in evaluation.user_rho:
        rho_array = np.maximum(np.asarray(rho_values, dtype=float), 1e-12)
        rho_sum = max(float(np.sum(rho_array)), 1e-12)
        power_weights_by_user.append((rho_array / rho_sum).tolist())

    allocation = {
        "method": str(method).upper() if isinstance(method, str) else "MIXED",
        "per_user_methods": [item.upper() for item in per_user_methods],
        "total_snr_db": float(payload.snr),
        "snr_per_stream": float(snr_per_stream),
        "num_streams_per_user": int(resolved_streams),
        "diag_gains": diag_gains_by_user,
        "power_weights": power_weights_by_user,
        "rho_by_user": [np.asarray(values, dtype=float).tolist() for values in evaluation.user_rho],
        "user_rates": np.asarray(evaluation.user_rates, dtype=float).tolist(),
        "user_bers": np.asarray(evaluation.user_bit_error_rates, dtype=float).tolist(),
        "sum_rate": float(evaluation.sum_rate),
        "bit_error_rate": float(evaluation.bit_error_rate),
        "offdiag_to_desired": float(evaluation.offdiag_to_desired),
        "evaluation_mode": evaluation_mode,
    }
    return np.asarray(chain.f_bb, dtype=complex), allocation


def _build_mixed_structured_chain(
    env: FullyDigitalMuMimoBicmEnvironment,
    user_channels: np.ndarray,
    snr_per_stream: float,
    per_user_methods: Sequence[str],
):
    core = env.core
    user_channels = np.asarray(user_channels, dtype=complex)
    f_rf = env.identity_precoder()
    effective_channels = core.build_effective_channels(user_channels, f_rf)
    bd_digital_bases = []
    user_blocks = []
    f_blocks = []
    r_chains = []
    q_chains = []
    f_rf_blocks = core.split_rf_blocks(f_rf)

    for user_idx in range(env.num_users):
        strategy = str(per_user_methods[user_idx]).lower()
        n_k = core.build_bd_digital_basis(effective_channels, user_idx)
        reduced_channel = effective_channels[user_idx] @ n_k
        u_eff, singular_values, vh_eff = np.linalg.svd(reduced_channel, full_matrices=False)
        singular_values = singular_values[: env.num_streams_per_user]
        u_eff = u_eff[:, : env.num_streams_per_user]
        v_eff = vh_eff.conj().T[:, : env.num_streams_per_user]

        if strategy == "svd":
            f_k_local = v_eff
            q_local = u_eff
            r_local = np.diag(singular_values)
        elif strategy == "gmd":
            gmd_block = core.digital_precoder.design_gmd_transceiver_from_svd(
                u_eff=u_eff,
                v_eff=v_eff,
                singular_values=singular_values,
            )
            f_k_local = gmd_block.p_gmd
            q_local = gmd_block.q_gmd
            r_local = gmd_block.r_gmd
        elif strategy == "ucd":
            ucd_block = core.digital_precoder.design_ucd_transceiver_from_svd(
                u_eff=u_eff,
                v_eff=v_eff,
                singular_values=singular_values,
                snr_per_stream=snr_per_stream,
                alpha_override=1.0 / max(float(snr_per_stream) * env.num_users, 1e-12),
            )
            f_k_local = ucd_block.p_ucd
            q_local = ucd_block.w_ucd
            r_local = ucd_block.r_aug_ucd
        else:
            raise ValueError(f"Unsupported per-user strategy: {strategy}")

        user_blocks.append(n_k @ f_k_local)
        f_blocks.append(np.asarray(f_k_local, dtype=complex))
        bd_digital_bases.append(n_k)
        r_chains.append(np.asarray(r_local, dtype=complex))
        q_chains.append(np.asarray(q_local, dtype=complex))

    f_bb = np.hstack(user_blocks)
    full_precoder = f_rf @ f_bb
    power = float(np.linalg.norm(full_precoder, "fro") ** 2)
    if power > 1e-12:
        correction_scale = np.sqrt(env.digital_power_constraint / power)
        f_bb = f_bb * correction_scale
        r_chains = [correction_scale * r_local for r_local in r_chains]

    class _Chain:
        pass

    chain = _Chain()
    chain.f_bb = f_bb
    chain.effective_channels = effective_channels
    chain.bd_digital_bases = bd_digital_bases
    chain.f_blocks = f_blocks
    chain.q_chains = q_chains
    chain.r_chains = r_chains
    chain.f_rf_blocks = f_rf_blocks
    return chain


@register_node
class UserConfigNode(BaseNode):
    """独立用户实体模块：一个模块代表一个用户"""
    NODE_TYPE = "UserConfig"
    NODE_CATEGORY = "Config"
    NODE_COLOR = "#60a5fa"

    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "User")
        self.properties = {
            "user_index": 0,
        }
        self.add_output("UserCfg")

    def process(self, inputs: list) -> TensorPayload:
        del inputs
        user_index = int(self.get_property("user_index", 0))
        return TensorPayload(
            data=np.array([user_index], dtype=int),
            num_users=1,
            metadata={
                "user_index": user_index,
                "topology_kind": "user-entity",
            },
        )


@register_node
class ChannelNode(BaseNode):
    """信道生成节点 - 支持 6G 特性"""
    NODE_TYPE = "ChannelGenerator"
    NODE_CATEGORY = "Source"
    NODE_COLOR = "#f59e0b"
    
    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "Channel")
        
        # 默认属性
        self.properties = {
            "num_tx_antennas": 16,
            "num_rx_antennas": 4,
            "frequency_band": FrequencyBand.SUB6G.value,
            "channel_model": "cdl-a",
            "snr_db": 10.0,
        }
        
        self.add_input("UserCfg")
        self.add_output("H")

    def ready(self) -> bool:
        user_port = self.inputs[0]
        return (not user_port.connected_ports) or user_port.data is not None
    
    def process(self, inputs: list) -> TensorPayload:
        user_payload = inputs[0] if inputs else None
        user_index = int(user_payload.metadata.get("user_index", 0)) if user_payload is not None else 0
        num_users = 1
        num_tx = self.get_property("num_tx_antennas", 16)
        num_rx = self.get_property("num_rx_antennas", 4)
        snr_db = float(self.get_property("snr_db", 10.0))
        
        band_name = self.get_property("frequency_band", FrequencyBand.SUB6G.value)
        channel_model_name = str(self.get_property("channel_model", "cdl-a"))
        band_profile = resolve_6g_channel_profile(band_name)
        carrier_freq = float(band_profile["carrier_frequency_hz"])

        channel_model = ChannelModel(
            num_tx_antennas=int(num_tx),
            num_rx_antennas=int(num_rx),
            channel_type=channel_model_name,
            frequency_band=band_name,
            carrier_frequency=float(band_profile["carrier_frequency_hz"]),
            delay_spread=float(band_profile["delay_spread_s"]),
            sampling_frequency=float(band_profile["sampling_frequency_hz"]),
        )
        H_single = np.asarray(channel_model.generate_channel(), dtype=complex)
        H = H_single[None, :, :]

        singular_values = []
        for user_idx in range(num_users):
            singular_values.append(np.linalg.svd(H[user_idx], compute_uv=False).tolist())
        
        return TensorPayload(
            data=H,
            carrier_freq=carrier_freq,
            snr=snr_db,
            num_users=num_users,
            num_tx_antennas=num_tx,
            num_rx_antennas=num_rx,
            metadata={
                "is_sub_thz": carrier_freq >= 100e9,
                "is_xl_mimo": num_tx >= 256,
                "channel_model": channel_model_name,
                "frequency_band": band_name,
                "delay_spread_s": float(band_profile["delay_spread_s"]),
                "sampling_frequency_hz": float(band_profile["sampling_frequency_hz"]),
                "singular_values": singular_values,
                "topology_kind": "bs-ue-branch",
                "user_index": user_index,
            }
        )


@register_node
class PrecodingNodeSVD(BaseNode):
    """SVD 预编码节点"""
    NODE_TYPE = "Precoding_SVD"
    NODE_CATEGORY = "Precoding"
    NODE_COLOR = "#10b981"
    
    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "SVD")
        self.properties = {"num_streams": 4}
        self.add_input("H")
        self.add_output("F")
    
    def process(self, inputs: list) -> TensorPayload:
        if not inputs or inputs[0].data.size == 0:
            return TensorPayload(np.array([]), 0, 0)
        
        payload = inputs[0]
        requested_streams = int(self.get_property("num_streams", 4))
        F_total, allocation = _evaluate_full_digital_method(
            payload=payload,
            method="svd",
            num_streams_per_user=requested_streams,
        )
        
        return TensorPayload(
            data=F_total,
            carrier_freq=payload.carrier_freq,
            snr=payload.snr,
            num_users=payload.num_users,
            num_tx_antennas=payload.num_tx_antennas,
            num_rx_antennas=payload.num_rx_antennas,
            metadata={
                **payload.metadata,
                "method": "SVD",
                "allocation": allocation,
            }
        )


@register_node
class PrecodingNodeGMD(BaseNode):
    """GMD 预编码节点"""
    NODE_TYPE = "Precoding_GMD"
    NODE_CATEGORY = "Precoding"
    NODE_COLOR = "#8b5cf6"
    
    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "GMD")
        self.properties = {"num_streams": 4}
        self.add_input("H")
        self.add_output("F")
    
    def process(self, inputs: list) -> TensorPayload:
        if not inputs or inputs[0].data.size == 0:
            return TensorPayload(np.array([]), 0, 0)
        
        payload = inputs[0]
        requested_streams = int(self.get_property("num_streams", 4))
        F_total, allocation = _evaluate_full_digital_method(
            payload=payload,
            method="gmd",
            num_streams_per_user=requested_streams,
        )
        
        return TensorPayload(
            data=F_total,
            carrier_freq=payload.carrier_freq,
            snr=payload.snr,
            num_users=payload.num_users,
            num_tx_antennas=payload.num_tx_antennas,
            num_rx_antennas=payload.num_rx_antennas,
            metadata={
                **payload.metadata,
                "method": "GMD",
                "allocation": allocation,
            }
        )


@register_node
class PrecodingNodeUCD(BaseNode):
    """UCD 预编码节点"""
    NODE_TYPE = "Precoding_UCD"
    NODE_CATEGORY = "Precoding"
    NODE_COLOR = "#ec4899"
    
    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "UCD")
        self.properties = {"num_streams": 4}
        self.add_input("H")
        self.add_output("F")
    
    def process(self, inputs: list) -> TensorPayload:
        if not inputs or inputs[0].data.size == 0:
            return TensorPayload(np.array([]), 0, 0)
        
        payload = inputs[0]
        requested_streams = int(self.get_property("num_streams", 4))
        F_total, allocation = _evaluate_full_digital_method(
            payload=payload,
            method="ucd",
            num_streams_per_user=requested_streams,
        )
        
        return TensorPayload(
            data=F_total,
            carrier_freq=payload.carrier_freq,
            snr=payload.snr,
            num_users=payload.num_users,
            num_tx_antennas=payload.num_tx_antennas,
            num_rx_antennas=payload.num_rx_antennas,
            metadata={
                **payload.metadata,
                "method": "UCD",
                "allocation": allocation,
            }
        )


@register_node
class MixedPrecodingNode(BaseNode):
    """每用户可选不同矩阵分解方式的混合预编码节点"""
    NODE_TYPE = "Precoding_Mixed"
    NODE_CATEGORY = "Precoding"
    NODE_COLOR = "#f59e0b"

    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "Mixed")
        self.properties = {
            "num_streams": 4,
            "user_1_method": "svd",
            "user_2_method": "gmd",
            "user_3_method": "ucd",
            "user_4_method": "svd",
            "user_5_method": "gmd",
            "user_6_method": "ucd",
            "user_7_method": "svd",
            "user_8_method": "gmd",
        }
        self.add_input("H")
        self.add_output("F")

    def process(self, inputs: list) -> TensorPayload:
        if not inputs or inputs[0].data.size == 0:
            return TensorPayload(np.array([]), 0, 0)

        payload = inputs[0]
        requested_streams = int(self.get_property("num_streams", 4))
        per_user_methods = [
            str(self.get_property(f"user_{user_idx + 1}_method", "svd")).lower()
            for user_idx in range(int(payload.num_users))
        ]
        F_total, allocation = _evaluate_full_digital_method(
            payload=payload,
            method=per_user_methods,
            num_streams_per_user=requested_streams,
        )
        return TensorPayload(
            data=F_total,
            carrier_freq=payload.carrier_freq,
            snr=payload.snr,
            num_users=payload.num_users,
            num_tx_antennas=payload.num_tx_antennas,
            num_rx_antennas=payload.num_rx_antennas,
            metadata={
                **payload.metadata,
                "method": "MIXED",
                "allocation": allocation,
            },
        )


@register_node
class UserBranchNode(BaseNode):
    """单用户分支节点：每个用户独立选择矩阵分解方式"""
    NODE_TYPE = "UserBranch"
    NODE_CATEGORY = "Matrix Decomp"
    NODE_COLOR = "#38bdf8"

    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "User Decomp")
        self.properties = {
            "user_index": 0,
            "method": "svd",
            "num_streams": 4,
        }
        self.add_input("H")
        self.add_output("UserResult")

    def process(self, inputs: list) -> TensorPayload:
        if not inputs or inputs[0].data.size == 0:
            return TensorPayload(np.array([]), 0, 0)

        payload = inputs[0]
        user_index = int(self.get_property("user_index", 0))
        num_users = int(payload.num_users)
        if user_index < 0 or user_index >= num_users:
            raise ValueError(f"user_index={user_index} is out of range for K={num_users}.")

        method = str(self.get_property("method", "svd")).lower()
        requested_streams = int(self.get_property("num_streams", 4))
        per_user_methods = ["svd"] * num_users
        per_user_methods[user_index] = method
        _, allocation = _evaluate_full_digital_method(
            payload=payload,
            method=per_user_methods,
            num_streams_per_user=requested_streams,
        )

        user_payload = payload.clone(
            data=np.asarray(payload.data[user_index : user_index + 1], dtype=complex),
            num_users=1,
        )
        user_payload.metadata.update(
            {
                "user_index": user_index,
                "method": method.upper(),
                "allocation": allocation,
                "diag_gains": [allocation["diag_gains"][user_index]],
                "power_weights": [allocation["power_weights"][user_index]],
                "rho_by_user": [allocation["rho_by_user"][user_index]],
                "user_rates": [allocation["user_rates"][user_index]],
                "user_bers": [allocation["user_bers"][user_index]],
                "topology_kind": "single-user-branch",
            }
        )
        return user_payload


@register_node
class MultiUserScopeNode(BaseNode):
    """多用户聚合监视器，汇总所有用户分支结果"""
    NODE_TYPE = "MultiUserScope"
    NODE_CATEGORY = "Sink"
    NODE_COLOR = "#f59e0b"

    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "Multi-User Scope")
        self.properties = {"expected_users": 2}
        for user_idx in range(8):
            self.add_input(f"UE{user_idx + 1}")
        self.add_output("Summary")

    def process(self, inputs: list) -> TensorPayload:
        active_inputs = [payload for payload in inputs if payload is not None and np.asarray(payload.data).size > 0]
        if not active_inputs:
            return TensorPayload(np.array([]), 0, 0)

        active_inputs.sort(key=lambda payload: int(payload.metadata.get("user_index", 0)))
        summary_rows = []
        for payload in active_inputs:
            user_index = int(payload.metadata.get("user_index", 0))
            summary_rows.append(
                {
                    "user_index": user_index,
                    "method": payload.metadata.get("method", "--"),
                    "diag_gains": payload.metadata.get("diag_gains", [[]])[0],
                    "power_weights": payload.metadata.get("power_weights", [[]])[0],
                    "rho": payload.metadata.get("rho_by_user", [[]])[0],
                    "user_rate": payload.metadata.get("user_rates", [0.0])[0],
                    "user_ber": payload.metadata.get("user_bers", [0.0])[0],
                }
            )

        merged = active_inputs[0].clone()
        merged.metadata["multi_user_scope"] = {
            "rows": summary_rows,
            "num_users": len(summary_rows),
            "sum_rate": float(sum(float(row["user_rate"]) for row in summary_rows)),
            "avg_ber": float(np.mean([float(row["user_ber"]) for row in summary_rows])),
        }
        return merged


@register_node
class SNRGeneratorNode(BaseNode):
    """SNR 扫描节点 - 生成 SNR 序列"""
    NODE_TYPE = "SNRGenerator"
    NODE_CATEGORY = "Source"
    NODE_COLOR = "#f97316"
    
    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "SNR Gen")
        self.properties = {
            "snr_start": 0,
            "snr_end": 40,
            "snr_step": 2,
            "frequency_band": FrequencyBand.SUB6G.value,
            "num_users": 2,
            "num_tx_antennas": 16,
            "num_rx_antennas": 4,
        }
        self._current_idx = 0
        self.add_output("SNR")
    
    def get_snr_sequence(self):
        start = self.get_property("snr_start", 0)
        end = self.get_property("snr_end", 40)
        step = self.get_property("snr_step", 2)
        return list(range(start, end + 1, step))
    
    def process(self, inputs: list) -> TensorPayload:
        seq = self.get_snr_sequence()
        if self._current_idx >= len(seq):
            self._current_idx = 0
        
        snr = seq[self._current_idx]
        self._current_idx += 1
        
        return TensorPayload(
            data=np.array([snr]),
            carrier_freq=FrequencyBand.from_value(self.get_property("frequency_band", FrequencyBand.SUB6G.value)).frequency_hz,
            snr=snr,
            num_users=int(self.get_property("num_users", 2)),
            num_tx_antennas=int(self.get_property("num_tx_antennas", 16)),
            num_rx_antennas=int(self.get_property("num_rx_antennas", 4)),
            metadata={
                "frequency_band": self.get_property("frequency_band", FrequencyBand.SUB6G.value),
                "is_sub_thz": FrequencyBand.from_value(self.get_property("frequency_band", FrequencyBand.SUB6G.value)).is_sub_thz,
                "is_xl_mimo": int(self.get_property("num_tx_antennas", 16)) >= 256,
            },
        )


@register_node
class ScopeNode(BaseNode):
    """示波器节点 - 动态绘图与显示"""
    NODE_TYPE = "Scope"
    NODE_CATEGORY = "Sink"
    NODE_COLOR = "#0ea5e9"
    
    def __init__(self, node_id: str, name: str = None):
        super().__init__(node_id, name or "Scope")
        self.properties = {"plot_type": "rate_ber"}
        self.results = {"snrs": [], "rates": [], "bers": [], "singular_values": []}
        self._update_callback = None
        self.add_input("Precoder")
    
    def set_update_callback(self, callback):
        self._update_callback = callback
    
    def process(self, inputs: list) -> TensorPayload:
        if not inputs or inputs[0].data.size == 0:
            return TensorPayload(np.array([]), 0, 0)
        
        payload = inputs[0]
        allocation = payload.metadata.get("allocation")
        if allocation:
            sv_per_user = allocation.get("diag_gains", [])
            sum_rate = float(allocation.get("sum_rate", 0.0))
            ber_value = float(allocation.get("bit_error_rate", 0.0))
        else:
            H = payload.data
            sv_per_user = []
            if H.ndim == 3:
                for user_idx in range(H.shape[0]):
                    _, S, _ = np.linalg.svd(H[user_idx])
                    sv_per_user.append(list(S[:4]))
            else:
                _, S, _ = np.linalg.svd(H)
                sv_per_user = [list(S[:4])]

            sum_rate = 0.0
            snr = payload.snr
            for user_sv in sv_per_user:
                for s in user_sv:
                    snr_linear = 10 ** (snr / 10)
                    rate = np.log2(1 + (s ** 2) * snr_linear)
                    sum_rate += rate
            ber_value = float(10 ** (-payload.snr / 10))
        
        # 记录数据
        self.results["snrs"].append(payload.snr)
        self.results["rates"].append(sum_rate)
        self.results["bers"].append(ber_value)
        self.results["singular_values"].append(sv_per_user)
        
        # 回调更新 UI
        if self._update_callback:
            self._update_callback(self.results)
        
        payload.metadata["scope_results"] = {
            "latest_sum_rate": sum_rate,
            "latest_ber": ber_value,
            "latest_singular_values": sv_per_user,
        }
        payload.metadata["scope_name"] = self.name
        return payload
