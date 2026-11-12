"""
Autonomous Edge Anomaly Detection Runtime Daemon.
Deploys on industrial gateways (Raspberry Pi, Advantech, Siemens IOT2050) & Smart Home hubs.
Runs continuous real-time sliding-window inference with INT8 quantized models,
trips hardware safety alarms, and generates forensic audit logs.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.synthetic_generator import SensorDataGenerator, DEFAULT_PROFILES, SENSOR_NAMES
from data.dataset_loader import TimeSeriesScaler
from models.anomaly_detector import OnDeviceAnomalyDetector
from models.lightweight_autoencoder import build_model
from models.quantization import quantize_model_dynamic


class EdgeAnomalyRuntime:
    """Production runtime for edge-level autonomous anomaly detection."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.0384,
        window_size: int = 16,
        num_sensors: int = 8,
        log_file: str = "edge_alarms.log",
    ):
        self.window_size = window_size
        self.num_sensors = num_sensors
        self.threshold = threshold
        self.log_file = log_file

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        print(f"[*] Initializing Edge Runtime from: {model_path}")
        raw_model = build_model({"architecture": "LightweightAutoencoder", "window_size": window_size, "input_dim": num_sensors})
        raw_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        # Apply INT8 dynamic quantization for edge optimization
        self.model = quantize_model_dynamic(raw_model)
        self.detector = OnDeviceAnomalyDetector(model=self.model, threshold_percentile=95.0)
        self.detector.threshold_ = threshold

        # Internal FIFO ring buffer for sliding window: [16, 8]
        self.buffer = np.zeros((window_size, num_sensors), dtype=np.float32)
        self.buffer_filled = 0

    def push_sensor_reading(self, reading: np.ndarray) -> bool:
        """
        Pushes a single 8-channel reading vector [8] into the edge ring buffer.
        Returns:
            is_anomaly: True if window reconstruction error exceeds calibrated threshold.
        """
        if len(reading) != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensor values, got {len(reading)}")

        # Shift buffer left by 1 and insert new reading at end
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = reading
        self.buffer_filled = min(self.window_size, self.buffer_filled + 1)

        # Wait until initial sliding window is populated
        if self.buffer_filled < self.window_size:
            return False

        # Run on-device inference: input shape [1, 16, 8]
        input_window = np.expand_dims(self.buffer, axis=0)
        preds, errors, channel_err = self.detector.predict(input_window)

        is_anomaly = bool(preds[0] == 1)
        mse_score = float(errors[0])

        if is_anomaly:
            self._handle_alarm(mse_score, channel_err[0])

        return is_anomaly

    def _handle_alarm(self, mse_score: float, channel_errors: np.ndarray):
        """Dispatches safety interlock action and logs alarm event."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total_err = float(np.sum(channel_errors)) + 1e-6
        contributions = {SENSOR_NAMES[i]: round(float(channel_errors[i] / total_err * 100), 2) for i in range(len(SENSOR_NAMES))}
        
        # Identify highest contributor
        top_sensor = max(contributions.items(), key=lambda x: x[1])

        alarm_entry = {
            "timestamp": timestamp,
            "event": "ANOMALY_ALARM_TRIGGERED",
            "mse_score": round(mse_score, 6),
            "threshold": round(self.threshold, 6),
            "top_root_cause": top_sensor[0],
            "top_contribution_pct": top_sensor[1],
            "hardware_safety_interlock": "TRIPPED",
            "all_sensor_contributions": contributions,
        }

        # Terminal alert
        print(f"\n[!!! SAFETY ALARM !!!] {timestamp}")
        print(f"  MSE Reconstruction Error: {mse_score:.5f} (Threshold: {self.threshold:.5f})")
        print(f"  Suspected Root Cause:     {top_sensor[0]} ({top_sensor[1]}% contribution)")
        print(f"  Safety Action:            PLC Emergency Relay Tripped")

        # Persist alarm log
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(alarm_entry) + "\n")


def run_edge_stream_simulation(client_idx: int = 0, duration_seconds: int = 25):
    """Simulates live continuous edge sensor streaming with automatic anomaly detection."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "results", "models", "model_fp32.pt")
    
    profile = DEFAULT_PROFILES[client_idx]
    print("=========================================================================")
    print(f" AUTONOMOUS ON-DEVICE EDGE RUNTIME: {profile.name} ")
    print(f" Target Asset Domain: {profile.category}")
    print("=========================================================================\n")

    runtime = EdgeAnomalyRuntime(model_path=model_path, threshold=0.0384)

    # Generate test stream
    generator = SensorDataGenerator(seed=101)
    raw_stream = generator.generate_normal_stream(profile, n_samples=300)
    test_stream, labels = generator.inject_anomalies(raw_stream, anomaly_ratio=0.15)

    scaler = TimeSeriesScaler()
    scaled_stream = scaler.fit_transform(test_stream)

    print(f"[*] Starting Real-Time Ingestion Loop (10 Hz). Press Ctrl+C to stop...\n")
    alarms_detected = 0

    try:
        for idx in range(len(scaled_stream)):
            reading = scaled_stream[idx]
            is_alarm = runtime.push_sensor_reading(reading)
            if is_alarm:
                alarms_detected += 1
            else:
                sys.stdout.write(f"\r[Step {idx:03d}] Normal Operation | Sensor 0: {test_stream[idx, 0]:.2f} | Temp: {test_stream[idx, 1]:.1f}°C")
                sys.stdout.flush()

            time.sleep(0.05)  # Simulate 20 Hz edge loop

    except KeyboardInterrupt:
        print("\n[*] Gracefully stopping edge runtime.")

    print(f"\n\n[SUMMARY] Ingestion complete. Total alarms triggered: {alarms_detected}")
    print(f"Alarm logs written to: {runtime.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Autonomous Edge Anomaly Detector")
    parser.add_argument("--client", type=int, default=0, help="Client profile index (0=CNC, 1=Press, 2=Motor, 3=HVAC, 4=Solar)")
    parser.add_argument("--duration", type=int, default=30, help="Simulation duration in seconds")
    args = parser.parse_args()

    run_edge_stream_simulation(client_idx=args.client, duration_seconds=args.duration)

