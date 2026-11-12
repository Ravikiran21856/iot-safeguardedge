"""
Communication and Bandwidth Profiler for Federated Edge Networks.
Tracks payload bytes, communication rounds, and compares FL network efficiency
against centralized raw sensor streaming.
"""

from typing import Dict, Any, List
import io
import torch


class CommunicationTracker:
    """Tracks uplink and downlink data volumes across federated rounds."""

    def __init__(self):
        self.total_uplink_bytes = 0    # Edge Client -> Central Server
        self.total_downlink_bytes = 0  # Central Server -> Edge Client
        self.rounds_log: List[Dict[str, Any]] = []

    def measure_state_dict_bytes(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Serializes PyTorch model parameters into an in-memory buffer to measure exact byte size."""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getbuffer().nbytes

    def record_round(
        self,
        round_idx: int,
        num_participating_clients: int,
        client_upload_bytes: int,
        server_broadcast_bytes: int,
    ):
        """Records bandwidth statistics for a completed federated round."""
        uplink_this_round = client_upload_bytes * num_participating_clients
        downlink_this_round = server_broadcast_bytes * num_participating_clients

        self.total_uplink_bytes += uplink_this_round
        self.total_downlink_bytes += downlink_this_round

        self.rounds_log.append({
            "round": round_idx,
            "participating_clients": num_participating_clients,
            "uplink_bytes": uplink_this_round,
            "downlink_bytes": downlink_this_round,
            "total_bytes_round": uplink_this_round + downlink_this_round,
            "cumulative_total_mb": round((self.total_uplink_bytes + self.total_downlink_bytes) / (1024 * 1024), 4),
        })

    def compute_bandwidth_savings(
        self,
        total_sensor_samples: int,
        num_sensors: int = 8,
        bytes_per_sample: int = 4,  # Float32
    ) -> Dict[str, Any]:
        """
        Compares FL communication overhead against transmitting raw continuous sensor telemetry to cloud.
        """
        # Centralized approach: All raw sensor readings continuously streamed to cloud
        raw_telemetry_bytes = total_sensor_samples * num_sensors * bytes_per_sample
        fl_total_bytes = self.total_uplink_bytes + self.total_downlink_bytes

        savings_pct = max(0.0, (raw_telemetry_bytes - fl_total_bytes) / max(raw_telemetry_bytes, 1) * 100.0)

        return {
            "fl_total_uplink_kb": round(self.total_uplink_bytes / 1024.0, 2),
            "fl_total_downlink_kb": round(self.total_downlink_bytes / 1024.0, 2),
            "fl_total_comm_kb": round(fl_total_bytes / 1024.0, 2),
            "raw_telemetry_stream_kb": round(raw_telemetry_bytes / 1024.0, 2),
            "bandwidth_reduction_pct": round(savings_pct, 2),
        }

