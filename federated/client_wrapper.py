"""
FedRep + FedBN Flower NumPy Client Wrapper.
Persists local classification heads across rounds and restricts global aggregation to backbone weights.
"""

import os
import torch
import flwr as fl
from typing import Dict, Tuple

class FedRepClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, train_loader, val_loader, module_contract, client_id: str, device: str = "cpu", max_batches: int = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.module = module_contract
        self.client_id = client_id
        self.device = device
        self.max_batches = max_batches
        self.checkpoints_dir = "checkpoints"
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.head_path = os.path.join(self.checkpoints_dir, f"client_{self.client_id}_head.pth")

        # Load persisted local head if exists
        if os.path.exists(self.head_path):
            try:
                self.model.fc.load_state_dict(torch.load(self.head_path, map_location=self.device))
                print(f"[Client {self.client_id}] Restored persisted classification head from {self.head_path}")
            except Exception as e:
                print(f"[Client {self.client_id}] Failed to load head checkpoint: {e}")

    def _is_shared_param(self, name: str) -> bool:
        # Exclude local classification head (fc) and local batch norm (bn / downsample.1)
        if "fc" in name or "bn" in name or "downsample.1" in name:
            return False
        return True

    def get_parameters(self, config: Dict[str, str]):
        """Extract and return ONLY the shared backbone parameters to the server."""
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if self._is_shared_param(name)
        ]

    def set_parameters(self, parameters):
        """Apply server-aggregated backbone parameters while preserving local classification head and batch norm."""
        shared_keys = [k for k in self.model.state_dict().keys() if self._is_shared_param(k)]
        if len(shared_keys) != len(parameters):
            print(f"[Client {self.client_id}] Warning: Shared keys count ({len(shared_keys)}) mismatch with parameter count ({len(parameters)})")
        
        state_dict = self.model.state_dict()
        for k, v in zip(shared_keys, parameters):
            state_dict[k] = torch.tensor(v)
        
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config: Dict[str, str]) -> Tuple[list, int, dict]:
        """Execute local training for one federated round."""
        try:
            self.set_parameters(parameters)
            epochs = int(config.get("epochs", 1))

            # Run local module training round
            try:
                _, num_samples, loss = self.module.train_one_round(
                    self.model, self.train_loader, epochs=epochs, device=self.device, max_batches=self.max_batches
                )
            except TypeError:
                _, num_samples, loss = self.module.train_one_round(
                    self.model, self.train_loader, epochs=epochs, device=self.device
                )

            # Save local classification head checkpoint
            try:
                torch.save(self.model.fc.state_dict(), self.head_path)
            except Exception as e:
                print(f"[Client {self.client_id}] Warning: Could not save head checkpoint: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return self.get_parameters(config), int(num_samples), {"train_loss": float(loss)}
        except Exception as err:
            import traceback
            print(f"[Client {self.client_id}] ERROR during fit(): {err}")
            traceback.print_exc()
            raise err

    def evaluate(self, parameters, config: Dict[str, str]) -> Tuple[float, int, dict]:
        """Evaluate local personalized model on local validation data."""
        try:
            self.set_parameters(parameters)
            try:
                loss, metrics = self.module.evaluate(self.model, self.val_loader, device=self.device, max_batches=self.max_batches)
            except TypeError:
                loss, metrics = self.module.evaluate(self.model, self.val_loader, device=self.device)

            clean_metrics = {str(k): float(v) for k, v in metrics.items()}
            num_val_samples = len(self.val_loader.dataset) if hasattr(self.val_loader, "dataset") else len(self.val_loader) * 16

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return float(loss), int(num_val_samples), clean_metrics
        except Exception as err:
            import traceback
            print(f"[Client {self.client_id}] ERROR during evaluate(): {err}")
            traceback.print_exc()
            raise err
