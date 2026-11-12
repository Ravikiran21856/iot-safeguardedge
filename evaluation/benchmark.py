"""
Scientific Benchmarking Suite:
Compares Isolated Local Edge Models vs Federated Learning vs Centralized Upper Bound.
Quantifies AUROC, F1-Score, False Alarm Rate, Network Bandwidth Overhead, and Data Privacy.
"""

import os
import sys
from typing import Dict, List, Any, Tuple
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lightweight_autoencoder import build_model
from models.anomaly_detector import OnDeviceAnomalyDetector
from federated.client import FederatedEdgeClient
from federated.server import FederatedServer
from data.dataset_loader import ClientDatasetBundle, prepare_federated_client_datasets
from data.synthetic_generator import SensorDataGenerator


class BenchmarkRunner:
    """Executes comparative evaluation across Isolated, Centralized, and Federated paradigms."""

    def __init__(
        self,
        config: Dict[str, Any],
        client_bundles: Dict[str, ClientDatasetBundle],
        device: str = "cpu",
    ):
        self.config = config
        self.bundles = client_bundles
        self.device = device

    def run_isolated_baseline(self) -> Dict[str, float]:
        """
        Baseline 1: Isolated Edge Training.
        Each edge device trains strictly in isolation on its own data without any collaborative sharing.
        """
        print("\n--- Running Isolated Local Training Baseline ---")
        client_results = []
        total_epochs = self.config["federated"]["num_rounds"] * self.config["local_training"]["local_epochs"]
        lr = self.config["local_training"]["learning_rate"]
        batch_size = self.config["local_training"]["batch_size"]

        for client_id, bundle in self.bundles.items():
            model = build_model(self.config["model"]).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            x_train = torch.from_numpy(bundle.x_train).float()
            loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)

            model.train()
            for _ in range(total_epochs):
                for (bx,) in loader:
                    bx = bx.to(self.device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, bx)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            detector = OnDeviceAnomalyDetector(
                model=model,
                device=self.device,
                threshold_method=self.config["anomaly_detection"]["threshold_method"],
                threshold_percentile=self.config["anomaly_detection"]["threshold_percentile"],
                std_multiplier=self.config["anomaly_detection"]["std_multiplier"],
            )
            detector.calibrate_threshold(bundle.x_val)
            eval_res = detector.evaluate(bundle.x_test, bundle.y_test)
            client_results.append(eval_res)

        return {
            "auroc": round(float(np.mean([r["auroc"] for r in client_results])), 4),
            "f1_score": round(float(np.mean([r["f1_score"] for r in client_results])), 4),
            "precision": round(float(np.mean([r["precision"] for r in client_results])), 4),
            "recall": round(float(np.mean([r["recall"] for r in client_results])), 4),
            "false_alarm_rate": round(float(np.mean([r["false_alarm_rate"] for r in client_results])), 4),
        }

    def run_centralized_baseline(self) -> Dict[str, float]:
        """
        Baseline 2: Centralized Cloud Training (Privacy-Violating Upper Bound).
        All edge raw telemetry is pooled onto a central cloud server.
        """
        print("\n--- Running Centralized Cloud Training Baseline ---")
        # Pool all training data
        all_train_x = np.concatenate([b.x_train for b in self.bundles.values()], axis=0)
        
        model = build_model(self.config["model"]).to(self.device)
        total_epochs = self.config["federated"]["num_rounds"] * self.config["local_training"]["local_epochs"]
        lr = self.config["local_training"]["learning_rate"]
        batch_size = self.config["local_training"]["batch_size"]

        x_train = torch.from_numpy(all_train_x).float()
        loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(total_epochs):
            for (bx,) in loader:
                bx = bx.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, bx)
                loss.backward()
                optimizer.step()

        # Evaluate on each client
        client_results = []
        for client_id, bundle in self.bundles.items():
            detector = OnDeviceAnomalyDetector(
                model=model,
                device=self.device,
                threshold_method=self.config["anomaly_detection"]["threshold_method"],
                threshold_percentile=self.config["anomaly_detection"]["threshold_percentile"],
                std_multiplier=self.config["anomaly_detection"]["std_multiplier"],
            )
            detector.calibrate_threshold(bundle.x_val)
            eval_res = detector.evaluate(bundle.x_test, bundle.y_test)
            client_results.append(eval_res)

        return {
            "auroc": round(float(np.mean([r["auroc"] for r in client_results])), 4),
            "f1_score": round(float(np.mean([r["f1_score"] for r in client_results])), 4),
            "precision": round(float(np.mean([r["precision"] for r in client_results])), 4),
            "recall": round(float(np.mean([r["recall"] for r in client_results])), 4),
            "false_alarm_rate": round(float(np.mean([r["false_alarm_rate"] for r in client_results])), 4),
        }

    def run_federated_pipeline(self) -> Tuple[Dict[str, float], FederatedServer, List[FederatedEdgeClient]]:
        """
        Proposed Solution: Lightweight Federated Learning (FedAvg / FedProx).
        """
        print("\n--- Running Proposed Lightweight Federated Learning Pipeline ---")
        clients = [
            FederatedEdgeClient(client_bundle=b, config=self.config, device=self.device)
            for b in self.bundles.values()
        ]
        server = FederatedServer(config=self.config, clients=clients, device=self.device)
        history = server.train_federated()

        final_round = history[-1]
        summary = {
            "auroc": final_round["avg_auroc"],
            "f1_score": final_round["avg_f1_score"],
            "precision": final_round["avg_precision"],
            "recall": final_round["avg_recall"],
            "false_alarm_rate": final_round["avg_false_alarm_rate"],
        }
        return summary, server, clients


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run IoT Anomaly Detection Benchmark")
    parser.add_argument("--config", type=str, default="./config/fl_config.yaml", help="Path to fl_config.yaml")
    parser.add_argument("--quick", action="store_true", help="Run fast 5-round benchmark for quick validation")
    args = parser.parse_args()

    # Load configuration
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "fl_config.yaml")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    if args.quick:
        config["federated"]["num_rounds"] = 5
        config["local_training"]["local_epochs"] = 2
        config["dataset"]["total_samples_per_client"] = 1200

    print("=========================================================================")
    print("        SCIENTIFIC BENCHMARK: IOT ON-DEVICE ANOMALY DETECTION            ")
    print(f" Mode: {'Quick (5 Rounds)' if args.quick else 'Full Benchmark (15 Rounds)'}")
    print("=========================================================================\n")

    # Generate synthetic sensor streams for the 5 IoT clients
    print("[1/3] Synthesizing Heterogeneous Sensor Data...")
    generator = SensorDataGenerator(seed=config["system"].get("seed", 42))
    raw_federated = generator.generate_federated_dataset(
        n_samples_per_client=config["dataset"].get("total_samples_per_client", 3500),
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        test_anomaly_ratio=config["dataset"]["anomaly_ratio_test"],
    )

    # Windowing and normal calibration
    print("[2/3] Normalizing Local Baselines & Creating Sliding Windows...")
    bundles = prepare_federated_client_datasets(
        raw_federated, window_size=config["model"]["window_size"]
    )

    # Execute benchmarks
    print("[3/3] Running Comparative Paradigm Evaluation...")
    runner = BenchmarkRunner(config, bundles, device=config["system"].get("device", "cpu"))
    isolated = runner.run_isolated_baseline()
    centralized = runner.run_centralized_baseline()
    federated, server, _ = runner.run_federated_pipeline()

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS SUMMARY                    ")
    print("=" * 65)
    print(f"{'Metric':<18} | {'Isolated Edge':<14} | {'Proposed FL':<14} | {'Centralized':<14}")
    print("-" * 65)
    for m in ["auroc", "f1_score", "precision", "recall", "false_alarm_rate"]:
        print(
            f"{m:<18} | "
            f"{isolated.get(m, 0.0):<14.4f} | "
            f"{federated.get(m, 0.0):<14.4f} | "
            f"{centralized.get(m, 0.0):<14.4f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
