"""
Time-Series Dataset Processing and Windowing for IoT Edge Devices.
Includes MinMax/Z-Score Normalization, Temporal Sliding Windows,
and PyTorch DataLoader wrappers for Federated Clients.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import os
import csv


class TimeSeriesScaler:
    """Feature-wise min-max scaler fit on local client normal baseline."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-7):
        self.feature_range = feature_range
        self.eps = eps
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "TimeSeriesScaler":
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        diff = self.data_max_ - self.data_min_
        diff[diff < self.eps] = 1.0  # Avoid zero division
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / diff
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        scaled = (X - self.data_min_) * self.scale_ + self.feature_range[0]
        return np.clip(scaled, self.feature_range[0] - 1.0, self.feature_range[1] + 1.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 16,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms continuous time-series [N, D] into sliding windows [num_windows, window_size, D].
    Window anomaly label is 1 if any point within the window is anomalous, else 0.
    """
    n_samples, n_features = data.shape
    if n_samples < window_size:
        raise ValueError(f"Data length ({n_samples}) must be >= window_size ({window_size})")

    num_windows = (n_samples - window_size) // stride + 1
    windows = np.empty((num_windows, window_size, n_features), dtype=np.float32)
    window_labels = np.empty((num_windows,), dtype=np.int32)

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[start:end]
        # Any anomaly within the window marks the window as anomalous
        window_labels[i] = 1 if np.any(labels[start:end] == 1) else 0

    return windows, window_labels


class ClientDatasetBundle:
    """Holds processed sliding window datasets and scalers for a single federated client."""

    def __init__(
        self,
        client_id: str,
        category: str,
        scaler: TimeSeriesScaler,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        raw_test_data: np.ndarray,
        raw_test_labels: np.ndarray,
    ):
        self.client_id = client_id
        self.category = category
        self.scaler = scaler
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.raw_test_data = raw_test_data
        self.raw_test_labels = raw_test_labels


def prepare_federated_client_datasets(
    federated_raw: Dict[str, Dict[str, np.ndarray]],
    window_size: int = 16,
    stride: int = 1,
) -> Dict[str, ClientDatasetBundle]:
    """
    Preprocesses all client partitions:
    - Fits scaler strictly on local train set (prevents test data leakage)
    - Normalizes train, val, and test splits
    - Creates temporal sliding windows for unsupervised model training
    """
    processed = {}
    for client_id, payload in federated_raw.items():
        profile = payload["profile"]
        scaler = TimeSeriesScaler(feature_range=(0.0, 1.0))
        
        # Fit scaler on normal training baseline
        train_norm = scaler.fit_transform(payload["train_data"])
        val_norm = scaler.transform(payload["val_data"])
        test_norm = scaler.transform(payload["test_data"])

        x_train, y_train = create_sliding_windows(train_norm, payload["train_labels"], window_size, stride)
        x_val, y_val = create_sliding_windows(val_norm, payload["val_labels"], window_size, stride)
        x_test, y_test = create_sliding_windows(test_norm, payload["test_labels"], window_size, stride)

        processed[client_id] = ClientDatasetBundle(
            client_id=client_id,
            category=profile.category,
            scaler=scaler,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            raw_test_data=payload["test_data"],
            raw_test_labels=payload["test_labels"],
        )
    return processed


def export_client_data_to_csv(client_bundle: ClientDatasetBundle, out_dir: str) -> str:
    """Exports raw test data with labels to CSV for inspection or external benchmarks."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{client_bundle.client_id}_test_stream.csv")
    
    sensor_headers = [
        "vibration_m_s2",
        "temperature_c",
        "rotational_speed_rpm",
        "current_amperes",
        "pressure_bar",
        "acoustic_emission_db",
        "humidity_pct",
        "voltage_volts",
        "anomaly_ground_truth",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(sensor_headers)
        data = client_bundle.raw_test_data
        labels = client_bundle.raw_test_labels
        for i in range(len(data)):
            row = list(data[i]) + [int(labels[i])]
            writer.writerow(row)

    return csv_path

