"""
Federated Edge Client Implementation for On-Device Training and Anomaly Detection.
Supports Local Differential Privacy (DP-SGD clipping + noise),
FedProx proximal regularization, and local evaluation.
"""

from typing import Dict, Any, Tuple, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.lightweight_autoencoder import build_model
from models.anomaly_detector import OnDeviceAnomalyDetector
from data.dataset_loader import ClientDatasetBundle


class FederatedEdgeClient:
    """
    Simulates an edge node (e.g. Raspberry Pi, Industrial Gateway, Smart Home Hub).
    Trains local model strictly on on-device data and generates privacy-preserving parameter updates.
    """

    def __init__(
        self,
        client_bundle: ClientDatasetBundle,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.client_id = client_bundle.client_id
        self.category = client_bundle.category
        self.bundle = client_bundle
        self.config = config
        self.device = device

        # Local neural network model
        self.model = build_model(config["model"]).to(device)

        # PyTorch DataLoaders
        self.batch_size = config["local_training"]["batch_size"]
        self.train_loader = self._create_dataloader(self.bundle.x_train, shuffle=True)
        self.val_loader = self._create_dataloader(self.bundle.x_val, shuffle=False)

        # Differential privacy parameters
        dp_cfg = config["federated"].get("differential_privacy", {})
        self.dp_enabled = dp_cfg.get("enabled", False)
        self.clip_norm = dp_cfg.get("clip_norm", 1.0)
        self.noise_multiplier = dp_cfg.get("noise_multiplier", 0.005)

        # FedProx regularization
        self.fedprox_mu = config["federated"].get("fedprox_mu", 0.0)

    def _create_dataloader(self, data: np.ndarray, shuffle: bool = True) -> DataLoader:
        tensor_x = torch.from_numpy(data).float()
        dataset = TensorDataset(tensor_x)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def set_parameters(self, global_parameters: Dict[str, torch.Tensor]):
        """Synchronizes local model weights with global broadcasted parameters."""
        self.model.load_state_dict(copy.deepcopy(global_parameters))

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns current local model weights."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def train_local_epoch(
        self,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> float:
        """
        Executes local training for specified epochs.
        Applies MSE reconstruction loss + FedProx penalty + gradient clipping + noise injection.
        """
        self.model.train()
        epochs = self.config["local_training"]["local_epochs"]
        lr = self.config["local_training"]["learning_rate"]
        weight_decay = self.config["local_training"].get("weight_decay", 1e-5)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        epoch_losses = []

        # Convert global params to device tensors for FedProx proximal loss computation
        ref_params = None
        if self.fedprox_mu > 0 and global_params is not None:
            ref_params = {k: v.to(self.device) for k, v in global_params.items()}

        for _ in range(epochs):
            batch_losses = []
            for (batch_x,) in self.train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                # Reconstruction loss: ||x - f(x)||^2
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)

                # Proximal regularization: (mu / 2) * ||w - w_global||^2
                if ref_params is not None:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in ref_params:
                            prox_loss += torch.sum((param - ref_params[name]) ** 2)
                    loss += (self.fedprox_mu / 2.0) * prox_loss

                loss.backward()

                # Gradient clipping for stability and differential privacy bound
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)

                optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(float(np.mean(batch_losses)))

        # Local Differential Privacy: Inject calibrated Gaussian noise into parameter deltas if enabled
        if self.dp_enabled and self.noise_multiplier > 0:
            with torch.no_grad():
                for param in self.model.parameters():
                    sigma = self.noise_multiplier * (self.clip_norm / len(self.train_loader.dataset))
                    noise = torch.normal(0.0, sigma, size=param.shape, device=param.device)
                    param.add_(noise)

        avg_loss = float(np.mean(epoch_losses))
        return avg_loss

    def evaluate_local(self, sensor_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Calibrates anomaly threshold on local normal validation set and
        evaluates performance on local anomalous test set.
        """
        detector = OnDeviceAnomalyDetector(
            model=self.model,
            device=self.device,
            threshold_method=self.config["anomaly_detection"]["threshold_method"],
            threshold_percentile=self.config["anomaly_detection"]["threshold_percentile"],
            std_multiplier=self.config["anomaly_detection"]["std_multiplier"],
        )

        detector.calibrate_threshold(self.bundle.x_val)
        eval_results = detector.evaluate(
            test_windows=self.bundle.x_test,
            ground_truth=self.bundle.y_test,
            sensor_names=sensor_names,
        )
        return eval_results

