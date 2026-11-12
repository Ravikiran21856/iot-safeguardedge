"""
Visualization Suite for IoT Federated Anomaly Detection.
Generates publication-quality figures:
- Federated Training Convergence Curves (Loss, AUROC, F1)
- Temporal Sensor Telemetry with Anomaly Ground Truth and Reconstruction Error Spikes
- ROC and Precision-Recall Curves
- Multi-Sensor Root-Cause Attribution Heatmap / Bar Charts
- Edge Hardware Footprint (FP32 vs INT8 size, Latency, Bandwidth Savings)
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless edge/server execution
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


class Visualizer:
    """Generates and exports diagnostic plots to results directory."""

    def __init__(self, output_dir: str = "./results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Set clean aesthetic style
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_training_convergence(
        self,
        history: List[Dict[str, Any]],
        filename: str = "fl_training_convergence.png",
    ) -> str:
        """Plots Training Loss, AUROC, and F1-Score over communication rounds."""
        rounds = [h["round"] for h in history]
        losses = [h["avg_local_loss"] for h in history]
        aurocs = [h["avg_auroc"] for h in history]
        f1_scores = [h["avg_f1_score"] for h in history]
        fars = [h["avg_false_alarm_rate"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Convergence Loss
        ax1.plot(rounds, losses, marker="o", color="#1f77b4", linewidth=2.0, label="Reconstruction Loss")
        ax1.set_title("Federated Aggregation: Convergence Loss", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Federated Communication Round", fontsize=11)
        ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(frameon=True)

        # Subplot 2: Detection Metrics (AUROC & F1 & False Alarm Rate)
        ax2.plot(rounds, aurocs, marker="s", color="#2ca02c", linewidth=2.0, label="Mean AUROC")
        ax2.plot(rounds, f1_scores, marker="^", color="#ff7f0e", linewidth=2.0, label="Mean F1-Score")
        ax2.plot(rounds, fars, marker="x", color="#d62728", linestyle="--", linewidth=1.5, label="False Alarm Rate")
        ax2.set_title("Edge Anomaly Detection Performance", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Federated Communication Round", fontsize=11)
        ax2.set_ylabel("Metric Score", fontsize=11)
        ax2.set_ylim(-0.02, 1.05)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(frameon=True, loc="center right")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_anomaly_detection_timeline(
        self,
        raw_sensor_data: np.ndarray,
        sensor_names: List[str],
        reconstruction_errors: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float,
        client_name: str,
        filename: str = "anomaly_timeline.png",
        max_samples: int = 500,
    ) -> str:
        """
        Plots a temporal window of continuous sensor channels, reconstruction error,
        threshold, and ground-truth anomaly bands.
        """
        limit = min(len(reconstruction_errors), max_samples)
        t = np.arange(limit)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

        # 1. Primary sensor signals (Vibration & Temperature & Pressure)
        ax1.plot(t, raw_sensor_data[:limit, 0], label=f"{sensor_names[0]} (Bearing Vib)", color="#3498db")
        ax1.plot(t, raw_sensor_data[:limit, 1], label=f"{sensor_names[1]} (Thermal)", color="#e74c3c")
        ax1.plot(t, raw_sensor_data[:limit, 4], label=f"{sensor_names[4]} (Hydraulic)", color="#9b59b6")
        ax1.set_title(f"Multivariate IoT Telemetry Stream: {client_name}", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Raw Sensor Value", fontsize=10)
        ax1.legend(loc="upper right", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. On-Device Reconstruction Error with dynamic threshold
        ax2.plot(t, reconstruction_errors[:limit], color="#2c3e50", linewidth=1.5, label="Reconstruction Error (MSE)")
        ax2.axhline(y=threshold, color="#e74c3c", linestyle="--", linewidth=2.0, label=f"Calibrated Threshold ({threshold:.4f})")
        
        # Shade alarm areas
        alarm_mask = reconstruction_errors[:limit] > threshold
        ax2.fill_between(t, 0, reconstruction_errors[:limit], where=alarm_mask, color="#e74c3c", alpha=0.35, label="Active Alarm")
        ax2.set_ylabel("Reconstruction Error", fontsize=10)
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Ground Truth Anomaly Intervals
        ax3.fill_between(t, 0, ground_truth[:limit], step="mid", color="#d35400", alpha=0.6, label="Ground Truth Anomaly")
        ax3.set_ylabel("Ground Truth", fontsize=10)
        ax3.set_xlabel("Time Step (Seconds / Index)", fontsize=11)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["Normal", "Anomaly"])
        ax3.legend(loc="upper right", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_benchmark_comparison(
        self,
        benchmarks: Dict[str, Dict[str, float]],
        filename: str = "benchmark_comparison.png",
    ) -> str:
        """
        Compares Isolated Local vs Federated Learning vs Centralized Upper Bound
        across AUROC, F1-Score, and Communication Bandwidth Overhead.
        """
        models = list(benchmarks.keys())
        metrics = ["auroc", "f1_score", "precision", "recall"]

        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#95a5a6", "#2980b9", "#27ae60"]

        for i, (name, stats) in enumerate(benchmarks.items()):
            values = [stats.get(m, 0.0) for m in metrics]
            ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)], edgecolor="black", alpha=0.85)

        ax.set_title("Paradigm Benchmark: Isolated Edge vs. Federated vs. Centralized", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(["AUROC", "F1-Score", "Precision", "Recall"], fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=True, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_edge_efficiency(
        self,
        quant_info: Dict[str, Any],
        comm_info: Dict[str, Any],
        filename: str = "edge_efficiency_metrics.png",
    ) -> str:
        """
        Visualizes TinyML Footprint:
        - FP32 vs INT8 Memory Size (KB)
        - On-Device Inference Latency (ms)
        - Network Bandwidth Savings (% and KB)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

        # Subplot 1: Model Memory Footprint
        labels_mem = ["FP32 Model", "INT8 (TinyML)"]
        sizes = [quant_info["fp32_size_kb"], quant_info["int8_size_kb"]]
        bars1 = ax1.bar(labels_mem, sizes, color=["#34495e", "#1abc9c"], edgecolor="black", width=0.5)
        ax1.set_title("Model Memory Size (KB)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Kilobytes (KB)", fontsize=11)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f} KB", ha="center", va="bottom", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: Single-Window Inference Latency
        labels_lat = ["FP32 Inference", "INT8 Quantized"]
        latencies = [quant_info["fp32_latency_ms"], quant_info["int8_latency_ms"]]
        bars2 = ax2.bar(labels_lat, latencies, color=["#e67e22", "#27ae60"], edgecolor="black", width=0.5)
        ax2.set_title("Edge Inference Latency (ms)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Milliseconds (ms/window)", fontsize=11)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f} ms", ha="center", va="bottom", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Subplot 3: Network Bandwidth Overhead
        labels_bw = ["Raw Telemetry", "Lightweight FL"]
        bandwidths = [comm_info["raw_telemetry_stream_kb"], comm_info["fl_total_comm_kb"]]
        bars3 = ax3.bar(labels_bw, bandwidths, color=["#c0392b", "#2980b9"], edgecolor="black", width=0.5)
        ax3.set_title(f"Bandwidth Savings: {comm_info['bandwidth_reduction_pct']}%", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Total Kilobytes (KB)", fontsize=11)
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, yval + 5, f"{yval:.0f} KB", ha="center", va="bottom", fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_sensor_attributions(
        self,
        sensor_attributions: Dict[str, float],
        client_name: str,
        filename: str = "sensor_root_cause.png",
    ) -> str:
        """Horizontal bar chart indicating root-cause sensor contribution to anomaly alarms."""
        fig, ax = plt.subplots(figsize=(9, 5))
        sensors = list(sensor_attributions.keys())
        pcts = list(sensor_attributions.values())

        y_pos = np.arange(len(sensors))
        bars = ax.barh(y_pos, pcts, color="#8e44ad", edgecolor="black", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sensors, fontsize=10)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel("Reconstruction Error Contribution (%)", fontsize=11)
        ax.set_title(f"Root Cause Anomaly Attribution: {client_name}", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + 0.5, bar.get_y() + bar.get_height() / 2, f"{xval:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

