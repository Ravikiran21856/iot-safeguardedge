"""
On-Device Anomaly Scoring and Thresholding Engine.
Computes multi-channel reconstruction error, determines dynamic thresholds,
and performs root-cause sensor contribution attribution for industrial alarms.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score


class OnDeviceAnomalyDetector:
    """
    Evaluates streaming sensor windows on-device using reconstruction loss
    and triggers alarms when error exceeds dynamically calibrated thresholds.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        threshold_method: str = "percentile",
        threshold_percentile: float = 95.0,
        std_multiplier: float = 3.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold_method = threshold_method
        self.threshold_percentile = threshold_percentile
        self.std_multiplier = std_multiplier
        self.threshold_: Optional[float] = None
        self.val_errors_: Optional[np.ndarray] = None

    def compute_reconstruction_errors(
        self,
        windows: np.ndarray,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes reconstruction errors for sliding windows.
        Returns:
            mean_errors: np.ndarray of shape [N] (MSE per window)
            channel_errors: np.ndarray of shape [N, num_sensors] (Mean error per sensor channel)
        """
        self.model.eval()
        n_samples = len(windows)
        mean_errors = []
        channel_errors = []

        tensor_data = torch.from_numpy(windows).float()

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_x = tensor_data[i : i + batch_size].to(self.device)
                reconstructed = self.model(batch_x)
                
                # Squared difference: (B, W, D)
                diff_sq = (batch_x - reconstructed) ** 2
                
                # Window MSE: (B,)
                batch_mse = diff_sq.mean(dim=(1, 2)).cpu().numpy()
                mean_errors.extend(batch_mse)

                # Channel-wise MSE across the temporal window: (B, D)
                batch_channel_mse = diff_sq.mean(dim=1).cpu().numpy()
                channel_errors.extend(batch_channel_mse)

        return np.array(mean_errors, dtype=np.float32), np.array(channel_errors, dtype=np.float32)

    def calibrate_threshold(self, val_windows: np.ndarray) -> float:
        """Calibrates baseline threshold strictly on normal validation windows."""
        val_errors, _ = self.compute_reconstruction_errors(val_windows)
        self.val_errors_ = val_errors

        if self.threshold_method == "percentile":
            self.threshold_ = float(np.percentile(val_errors, self.threshold_percentile))
        elif self.threshold_method == "std_dev":
            mean_err = float(np.mean(val_errors))
            std_err = float(np.std(val_errors))
            self.threshold_ = mean_err + self.std_multiplier * std_err
        elif self.threshold_method == "extreme_value":
            # Simple Extreme Value Theory (EVT) peak-over-threshold approximation
            u = float(np.percentile(val_errors, 90.0))
            excesses = val_errors[val_errors > u] - u
            if len(excesses) > 0:
                scale = float(np.mean(excesses))
                self.threshold_ = u + scale * np.log(len(val_errors) / 10.0)
            else:
                self.threshold_ = float(np.percentile(val_errors, 98.0))
        else:
            self.threshold_ = float(np.percentile(val_errors, 95.0))

        return self.threshold_

    def predict(
        self,
        test_windows: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classifies test windows as normal (0) or anomaly (1).
        Returns:
            binary_predictions: [N]
            reconstruction_errors: [N]
            channel_contributions: [N, D]
        """
        th = threshold if threshold is not None else self.threshold_
        if th is None:
            raise ValueError("Threshold is not calibrated. Call calibrate_threshold first.")

        errors, channel_err = self.compute_reconstruction_errors(test_windows)
        preds = (errors > th).astype(np.int32)
        return preds, errors, channel_err

    def evaluate(
        self,
        test_windows: np.ndarray,
        ground_truth: np.ndarray,
        threshold: Optional[float] = None,
        sensor_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive edge anomaly detection evaluation.
        Computes AUROC, AUPRC, Precision, Recall, F1, False Alarm Rate,
        and root-cause sensor contribution breakdowns.
        """
        preds, errors, channel_err = self.predict(test_windows, threshold)
        y_true = ground_truth.astype(np.int32)

        # Precision, Recall, F1-Score
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )

        # AUROC & AUPRC
        try:
            auroc = float(roc_auc_score(y_true, errors))
        except Exception:
            auroc = 0.5
        try:
            auprc = float(average_precision_score(y_true, errors))
        except Exception:
            auprc = 0.0

        # Confusion Matrix elements
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False Alarm Rate
        accuracy = float((tp + tn) / len(y_true))

        # Root-cause analysis: which sensors contributed most to detected anomalies
        anomaly_indices = np.where(preds == 1)[0]
        if len(anomaly_indices) > 0:
            avg_anomaly_channel_err = channel_err[anomaly_indices].mean(axis=0)
            total_err = float(np.sum(avg_anomaly_channel_err)) + 1e-7
            contributions = (avg_anomaly_channel_err / total_err).tolist()
        else:
            contributions = [0.0] * channel_err.shape[1]

        sensor_attr = {}
        if sensor_names and len(sensor_names) == len(contributions):
            for sname, pct in zip(sensor_names, contributions):
                sensor_attr[sname] = round(pct * 100, 2)

        return {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "accuracy": round(accuracy, 4),
            "false_alarm_rate": round(far, 4),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "threshold": round(float(self.threshold_), 6),
            "sensor_contributions_pct": sensor_attr,
            "reconstruction_errors": errors,
            "predictions": preds,
        }

