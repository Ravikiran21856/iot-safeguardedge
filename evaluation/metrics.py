"""
Evaluation Metrics and Statistical Aggregators for IoT Anomaly Detection.
Provides calculation of Classification Metrics, Edge Hardware Footprint,
and Communication Bandwidth Metrics.
"""

from typing import Dict, List, Any
import numpy as np


def compute_aggregate_metrics(eval_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes mean and standard deviation for key performance indicators across clients."""
    metrics = ["auroc", "auprc", "f1_score", "precision", "recall", "false_alarm_rate", "accuracy"]
    aggregated = {}
    for m in metrics:
        values = [e[m] for e in eval_list if m in e]
        if values:
            aggregated[f"mean_{m}"] = round(float(np.mean(values)), 4)
            aggregated[f"std_{m}"] = round(float(np.std(values)), 4)
        else:
            aggregated[f"mean_{m}"] = 0.0
            aggregated[f"std_{m}"] = 0.0
    return aggregated


def format_confusion_matrix_table(cm: Dict[str, int]) -> str:
    """Formats a human-readable 2x2 confusion matrix."""
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    return (
        f"+-------------------+-------------------+\n"
        f"| True Positives:  {tp:<4} | False Positives: {fp:<4} |\n"
        f"| False Negatives: {fn:<4} | True Negatives:  {tn:<4} |\n"
        f"+-------------------+-------------------+"
    )

