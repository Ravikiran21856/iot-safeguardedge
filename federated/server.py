"""
Federated Learning Central Aggregator and Server.
Orchestrates communication rounds, client sampling, FedAvg/FedProx aggregation,
and tracks global convergence metrics across IoT edge devices.
"""

from typing import Dict, List, Any, Tuple, Optional
import copy
import numpy as np
import torch
import torch.nn as nn

from models.lightweight_autoencoder import build_model
from federated.client import FederatedEdgeClient
from federated.communication import CommunicationTracker


class FederatedServer:
    """
    Central Coordinator for the Federated Learning process across IoT devices.
    Executes aggregation algorithms (FedAvg, FedProx) and monitors convergence.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        clients: List[FederatedEdgeClient],
        device: str = "cpu",
    ):
        self.config = config
        self.clients = clients
        self.device = device

        # Instantiate global architecture
        self.global_model = build_model(config["model"]).to(device)
        self.tracker = CommunicationTracker()

        self.history: List[Dict[str, Any]] = []
        self.best_f1 = 0.0
        self.best_global_params: Optional[Dict[str, torch.Tensor]] = None

    def aggregate_parameters(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes sample-weighted average of client parameter updates (FedAvg):
        w_{global} = sum_{k} (n_k / N) * w_k
        """
        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples == 0:
            raise ValueError("Total client samples across active clients is zero.")

        # Initialize accumulator with zeros matching first client's state dict
        first_state = client_updates[0][0]
        aggregated_state = {k: torch.zeros_like(v) for k, v in first_state.items()}

        for state_dict, num_samples in client_updates:
            weight = float(num_samples) / float(total_samples)
            for k in aggregated_state.keys():
                # Cast to float for precision accumulation
                aggregated_state[k] += weight * state_dict[k].to(aggregated_state[k].device)

        return aggregated_state

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        """
        Executes a single communication round:
        1. Broadcast global weights to edge clients
        2. Edge clients execute local training (DP & proximal penalty)
        3. Clients upload parameter weights
        4. Central server aggregates weights via FedAvg
        5. Evaluate updated model on all client test streams
        """
        # Global weights snapshot
        global_params = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
        broadcast_bytes = self.tracker.measure_state_dict_bytes(global_params)

        fraction_fit = self.config["federated"].get("fraction_fit", 1.0)
        num_selected = max(1, int(len(self.clients) * fraction_fit))
        selected_clients = list(np.random.choice(self.clients, size=num_selected, replace=False))

        client_updates = []
        local_losses = []
        upload_bytes_sample = 0

        # Local Edge Training
        for client in selected_clients:
            client.set_parameters(global_params)
            local_loss = client.train_local_epoch(global_params=global_params)
            local_losses.append(local_loss)

            client_state = client.get_parameters()
            n_samples = len(client.bundle.x_train)
            client_updates.append((client_state, n_samples))

            if upload_bytes_sample == 0:
                upload_bytes_sample = self.tracker.measure_state_dict_bytes(client_state)

        # Communication volume logging
        self.tracker.record_round(
            round_idx=round_idx,
            num_participating_clients=len(selected_clients),
            client_upload_bytes=upload_bytes_sample,
            server_broadcast_bytes=broadcast_bytes,
        )

        # Federated Aggregation
        aggregated_params = self.aggregate_parameters(client_updates)
        self.global_model.load_state_dict(aggregated_params)

        # Global Evaluation across all Edge Nodes
        client_evals = []
        for client in self.clients:
            client.set_parameters(aggregated_params)
            res = client.evaluate_local()
            res["client_id"] = client.client_id
            res["category"] = client.category
            client_evals.append(res)

        avg_loss = float(np.mean(local_losses))
        avg_auroc = float(np.mean([e["auroc"] for e in client_evals]))
        avg_f1 = float(np.mean([e["f1_score"] for e in client_evals]))
        avg_precision = float(np.mean([e["precision"] for e in client_evals]))
        avg_recall = float(np.mean([e["recall"] for e in client_evals]))
        avg_far = float(np.mean([e["false_alarm_rate"] for e in client_evals]))

        if avg_f1 > self.best_f1:
            self.best_f1 = avg_f1
            self.best_global_params = copy.deepcopy(aggregated_params)

        round_summary = {
            "round": round_idx,
            "avg_local_loss": round(avg_loss, 5),
            "avg_auroc": round(avg_auroc, 4),
            "avg_f1_score": round(avg_f1, 4),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_false_alarm_rate": round(avg_far, 4),
            "client_evals": client_evals,
        }

        self.history.append(round_summary)
        return round_summary

    def train_federated(self, num_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """Runs full multi-round federated training cycle."""
        rounds = num_rounds or self.config["federated"]["num_rounds"]
        print(f"\n=======================================================")
        print(f" Starting Federated Learning Training: {rounds} Rounds")
        print(f" Participating Clients: {len(self.clients)} Edge Nodes")
        print(f" Aggregation Algorithm: {self.config['federated'].get('aggregation_algorithm', 'FedAvg')}")
        print(f" Local DP Enabled: {self.config['federated'].get('differential_privacy', {}).get('enabled', False)}")
        print(f"=======================================================\n")

        for r in range(1, rounds + 1):
            summary = self.run_round(r)
            print(
                f"[Round {r:02d}/{rounds:02d}] "
                f"Loss: {summary['avg_local_loss']:.4f} | "
                f"AUROC: {summary['avg_auroc']:.4f} | "
                f"F1: {summary['avg_f1_score']:.4f} | "
                f"Prec: {summary['avg_precision']:.4f} | "
                f"Rec: {summary['avg_recall']:.4f} | "
                f"FAR: {summary['avg_false_alarm_rate']:.4f}"
            )

        # Restore optimal parameters found during training
        if self.best_global_params is not None:
            self.global_model.load_state_dict(self.best_global_params)

        return self.history

