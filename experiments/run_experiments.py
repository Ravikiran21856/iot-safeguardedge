"""
Main Orchestration Script:
End-to-End Execution of Lightweight Federated Learning for On-Device Anomaly Detection.
Runs baseline comparisons, quantization, communications profiling, and exports figures.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.synthetic_generator import SensorDataGenerator, SENSOR_NAMES
from data.dataset_loader import prepare_federated_client_datasets, export_client_data_to_csv
from models.lightweight_autoencoder import count_parameters, get_model_size_kb
from models.quantization import benchmark_quantization
from evaluation.benchmark import BenchmarkRunner
from evaluation.visualizer import Visualizer


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(config_path: str = "./config/fl_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["system"].get("seed", 42)
    device = config["system"].get("device", "cpu")
    out_dir = config["system"].get("output_dir", "./results")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    print("=========================================================================")
    print(" ON-DEVICE FEDERATED ANOMALY DETECTION FOR INDUSTRIAL IOT & SMART HOMES ")
    print("=========================================================================\n")

    # 1. Generate Synthetic Edge Telemetry
    print("[1/6] Synthesizing Heterogeneous Sensor Telemetry...")
    generator = SensorDataGenerator(seed=seed)
    n_samples = config["dataset"].get("total_samples_per_client", 3500)
    raw_federated = generator.generate_federated_dataset(
        n_samples_per_client=n_samples,
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        test_anomaly_ratio=config["dataset"]["anomaly_ratio_test"],
    )

    # 2. Preprocess & Windowing
    print("[2/6] Normalizing Local Baselines & Creating Temporal Sliding Windows...")
    window_size = config["model"]["window_size"]
    client_bundles = prepare_federated_client_datasets(raw_federated, window_size=window_size)

    # Export sample test CSVs
    dataset_export_dir = os.path.join(out_dir, "datasets")
    for b in client_bundles.values():
        csv_p = export_client_data_to_csv(b, dataset_export_dir)
    print(f"      -> Sample test datasets exported to: {dataset_export_dir}")

    # 3. Benchmark Comparisons (Isolated vs Centralized vs Federated)
    print("[3/6] Running Paradigm Benchmarking...")
    benchmarker = BenchmarkRunner(config, client_bundles, device=device)
    isolated_metrics = benchmarker.run_isolated_baseline()
    centralized_metrics = benchmarker.run_centralized_baseline()
    federated_metrics, server, clients = benchmarker.run_federated_pipeline()

    # 4. TinyML Edge Quantization Benchmarking
    print("[4/6] Benchmarking Edge Model Quantization (FP32 vs INT8)...")
    sample_window = torch.from_numpy(list(client_bundles.values())[0].x_test[:10]).float()
    quant_results = benchmark_quantization(
        fp32_model=server.global_model,
        sample_window=sample_window,
        temp_dir=os.path.join(out_dir, "models"),
    )
    param_count = count_parameters(server.global_model)

    # 5. Communication Overhead and Bandwidth Savings
    print("[5/6] Calculating Communication Bandwidth Savings...")
    total_sensor_readings = sum(len(b.raw_test_data) + len(b.x_train) + len(b.x_val) for b in client_bundles.values())
    bandwidth_stats = server.tracker.compute_bandwidth_savings(
        total_sensor_samples=total_sensor_readings,
        num_sensors=config["model"]["input_dim"],
    )

    # 6. Evaluation & Visualization Generation
    print("[6/6] Generating Publication Figures and Diagnostic Plots...")
    viz = Visualizer(output_dir=os.path.join(out_dir, "plots"))
    p_conv = viz.plot_training_convergence(server.history)
    
    # Timeline plot for primary industrial client (CNC Milling)
    cnc_client = clients[0]
    eval_cnc = cnc_client.evaluate_local(sensor_names=SENSOR_NAMES)
    p_timeline = viz.plot_anomaly_detection_timeline(
        raw_sensor_data=cnc_client.bundle.raw_test_data,
        sensor_names=SENSOR_NAMES,
        reconstruction_errors=eval_cnc["reconstruction_errors"],
        ground_truth=cnc_client.bundle.y_test,
        threshold=eval_cnc["threshold"],
        client_name=cnc_client.bundle.client_id,
        filename="anomaly_timeline_cnc_milling.png",
    )

    # Paradigm comparison plot
    benchmarks_dict = {
        "Isolated Edge": isolated_metrics,
        "Proposed FedAvg": federated_metrics,
        "Centralized (Upper)": centralized_metrics,
    }
    p_bench = viz.plot_benchmark_comparison(benchmarks_dict)

    # Edge efficiency plot
    p_edge = viz.plot_edge_efficiency(quant_results, bandwidth_stats)

    # Root-cause sensor contribution plot
    p_attr = viz.plot_sensor_attributions(
        eval_cnc["sensor_contributions_pct"],
        client_name=cnc_client.bundle.client_id,
        filename="sensor_root_cause_cnc.png",
    )

    # Consolidate complete results summary
    full_summary = {
        "configuration": config,
        "model_architecture": {
            "name": config["model"]["architecture"],
            "parameter_count": param_count,
            "fp32_size_kb": quant_results["fp32_size_kb"],
            "int8_size_kb": quant_results["int8_size_kb"],
            "compression_ratio": quant_results["compression_ratio"],
            "fp32_latency_ms": quant_results["fp32_latency_ms"],
            "int8_latency_ms": quant_results["int8_latency_ms"],
        },
        "bandwidth_savings": bandwidth_stats,
        "benchmark_comparison": benchmarks_dict,
        "federated_history": server.history,
        "primary_client_evaluation": {
            "client_id": cnc_client.client_id,
            "auroc": eval_cnc["auroc"],
            "f1_score": eval_cnc["f1_score"],
            "precision": eval_cnc["precision"],
            "recall": eval_cnc["recall"],
            "false_alarm_rate": eval_cnc["false_alarm_rate"],
            "confusion_matrix": eval_cnc["confusion_matrix"],
            "sensor_attributions": eval_cnc["sensor_contributions_pct"],
        },
        "generated_plots": [p_conv, p_timeline, p_bench, p_edge, p_attr],
    }

    summary_json_path = os.path.join(out_dir, "experiment_summary.json")
    with open(summary_json_path, "w") as f:
        # Convert non-serializable items
        json.dump(full_summary, f, indent=2, default=str)

    # Print Terminal Report
    print("\n" + "=" * 70)
    print("                    EXPERIMENTAL RESULTS SUMMARY                    ")
    print("=" * 70)
    print(f" Model Parameter Count:        {param_count:,} parameters")
    print(f" Model Size (FP32 -> INT8):    {quant_results['fp32_size_kb']} KB -> {quant_results['int8_size_kb']} KB ({quant_results['compression_ratio']}x compression)")
    print(f" Edge Inference Latency (INT8):{quant_results['int8_latency_ms']} ms / window")
    print(f" Raw Telemetry Streamed:       {bandwidth_stats['raw_telemetry_stream_kb']} KB")
    print(f" FL Communication Total:       {bandwidth_stats['fl_total_comm_kb']} KB")
    print(f" Network Bandwidth Savings:    {bandwidth_stats['bandwidth_reduction_pct']}%")
    print("-" * 70)
    print(f"{'Metric':<18} | {'Isolated Edge':<14} | {'Proposed FL':<14} | {'Centralized':<14}")
    print("-" * 70)
    for m in ["auroc", "f1_score", "precision", "recall", "false_alarm_rate"]:
        print(
            f"{m:<18} | "
            f"{isolated_metrics.get(m, 0.0):<14.4f} | "
            f"{federated_metrics.get(m, 0.0):<14.4f} | "
            f"{centralized_metrics.get(m, 0.0):<14.4f}"
        )
    print("=" * 70)
    print(f"\nAll plots and summary saved in: {out_dir}")
    return full_summary


if __name__ == "__main__":
    run_pipeline()

