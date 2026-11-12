"""
Automated Test Suite for Lightweight Federated Learning IoT Pipeline.
Verifies Data Generation, Preprocessing, Model Architectures, Quantization,
and Federated Aggregation Logic.
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.synthetic_generator import SensorDataGenerator, DEFAULT_PROFILES
from data.dataset_loader import TimeSeriesScaler, create_sliding_windows
from models.lightweight_autoencoder import LightweightAutoencoder, Conv1DAutoencoder, count_parameters, get_model_size_kb
from models.quantization import quantize_model_dynamic
from models.anomaly_detector import OnDeviceAnomalyDetector
from federated.server import FederatedServer


class TestIoTPipeline(unittest.TestCase):

    def setUp(self):
        self.generator = SensorDataGenerator(seed=123)
        self.profile = DEFAULT_PROFILES[0]  # CNC Milling

    def test_synthetic_data_generation(self):
        """Test continuous sensor stream generation and anomaly injection."""
        n_samples = 500
        stream = self.generator.generate_normal_stream(self.profile, n_samples=n_samples)
        self.assertEqual(stream.shape, (n_samples, 8))
        self.assertFalse(np.isnan(stream).any())

        corrupted, labels = self.generator.inject_anomalies(stream, anomaly_ratio=0.10)
        self.assertEqual(corrupted.shape, (n_samples, 8))
        self.assertEqual(labels.shape, (n_samples,))
        self.assertGreater(labels.sum(), 0)

    def test_scaler_and_sliding_windows(self):
        """Test TimeSeriesScaler normalization and sliding window extraction."""
        data = np.random.randn(200, 8).astype(np.float32) * 50 + 100
        labels = np.zeros(200, dtype=np.int32)
        labels[50:60] = 1

        scaler = TimeSeriesScaler()
        scaled = scaler.fit_transform(data)
        self.assertTrue((scaled >= -0.1).all() and (scaled <= 1.1).all())

        window_size = 16
        windows, w_labels = create_sliding_windows(scaled, labels, window_size=window_size, stride=1)
        expected_windows = 200 - window_size + 1
        self.assertEqual(windows.shape, (expected_windows, window_size, 8))
        self.assertEqual(w_labels.shape, (expected_windows,))
        self.assertEqual(w_labels[45], 1)  # Region touching anomaly

    def test_lightweight_autoencoder_architecture(self):
        """Test lightweight MLP autoencoder dimensions, parameter count, and forward pass."""
        model = LightweightAutoencoder(window_size=16, input_dim=8, hidden_dims=(32, 16), latent_dim=8)
        param_count = count_parameters(model)
        size_kb = get_model_size_kb(model)

        # Confirm edge suitability: < 25,000 parameters and < 100 KB
        self.assertLess(param_count, 25000)
        self.assertLess(size_kb, 100.0)

        # Forward pass test
        dummy_input = torch.randn(4, 16, 8)
        reconstructed = model(dummy_input)
        self.assertEqual(reconstructed.shape, (4, 16, 8))

    def test_conv1d_autoencoder_architecture(self):
        """Test 1D-CNN autoencoder forward pass."""
        model = Conv1DAutoencoder(window_size=16, input_dim=8, latent_dim=8)
        dummy_input = torch.randn(4, 16, 8)
        reconstructed = model(dummy_input)
        self.assertEqual(reconstructed.shape, (4, 16, 8))

    def test_quantization(self):
        """Test Dynamic INT8 Quantization."""
        model = LightweightAutoencoder(window_size=16, input_dim=8, hidden_dims=(32, 16), latent_dim=8)
        quant_model = quantize_model_dynamic(model)
        dummy_input = torch.randn(2, 16, 8)
        out = quant_model(dummy_input)
        self.assertEqual(out.shape, (2, 16, 8))

    def test_anomaly_scoring_and_threshold(self):
        """Test threshold calibration and prediction output."""
        model = LightweightAutoencoder(window_size=16, input_dim=8)
        detector = OnDeviceAnomalyDetector(model=model, threshold_percentile=95.0)

        normal_val = np.random.randn(50, 16, 8).astype(np.float32) * 0.1
        threshold = detector.calibrate_threshold(normal_val)
        self.assertGreater(threshold, 0.0)

        test_windows = np.random.randn(30, 16, 8).astype(np.float32)
        test_labels = np.random.randint(0, 2, size=30)
        eval_metrics = detector.evaluate(test_windows, test_labels)

        self.assertIn("auroc", eval_metrics)
        self.assertIn("f1_score", eval_metrics)
        self.assertIn("confusion_matrix", eval_metrics)

    def test_federated_server_and_client_round(self):
        """Test end-to-end single federated round execution."""
        from federated.client import FederatedEdgeClient
        from data.dataset_loader import prepare_federated_client_datasets

        federated_raw = self.generator.generate_federated_dataset(n_samples_per_client=300)
        bundles = prepare_federated_client_datasets(federated_raw, window_size=16)

        config = {
            "model": {"architecture": "LightweightAutoencoder", "window_size": 16, "input_dim": 8, "hidden_dims": [16, 8], "latent_dim": 4},
            "local_training": {"batch_size": 16, "local_epochs": 1, "learning_rate": 0.01, "weight_decay": 1e-5},
            "federated": {"fraction_fit": 1.0, "fedprox_mu": 0.01, "differential_privacy": {"enabled": True, "clip_norm": 1.0, "noise_multiplier": 0.001}},
            "anomaly_detection": {"threshold_method": "percentile", "threshold_percentile": 95.0, "std_multiplier": 3.0},
        }

        clients = [FederatedEdgeClient(client_bundle=b, config=config) for b in bundles.values()]
        server = FederatedServer(config=config, clients=clients)
        summary = server.run_round(1)
        self.assertEqual(summary["round"], 1)
        self.assertIn("avg_auroc", summary)
        self.assertIn("avg_f1_score", summary)


if __name__ == "__main__":
    unittest.main()
