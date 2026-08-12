# FedMedDx Federated Core Architecture Guide

This document describes the federated learning infrastructure built on Flower (`flwr`), covering client wrapping, strategy execution (FedAvg, FedProx, DP, SecAgg+), simulation resource allocation, and experiment metric logging.

---

## 1. Flower Virtual Client Engine (VCE)

FedMedDx uses Flower's Virtual Client Engine (`flwr[simulation]`) to simulate realistic multi-hospital federated networks on a single GPU node.

### Client Resource Allocation
To ensure parallel client training utilizes CUDA without Out-Of-Memory (OOM) errors:
```python
client_resources = {
    "num_gpus": 0.25 if torch.cuda.is_available() else 0.0,
    "num_cpus": 1
}
```

---

## 2. Generic Flower Client Wrapper (`federated/client_wrapper.py`)

The core lead provides a unified `FlowerNumPyClient` class that interfaces with any modality module implementing `CONTRACT.md`:

```python
import flwr as fl
import torch

class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(self, module, train_loader, val_loader, device="cuda"):
        self.module = module
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = module.get_model().to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config.get("local_epochs", 1)
        state_dict, num_examples, loss = self.module.train_one_round(
            self.model, self.train_loader, epochs=epochs, device=self.device
        )
        return self.get_parameters(config={}), num_examples, {"loss": loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = self.module.evaluate(
            self.model, self.val_loader, device=self.device
        )
        return float(loss), len(self.val_loader.dataset), metrics
```

---

## 3. Supported Federated Strategies

### FedAvg (`run_fedavg.py`)
Standard Federated Averaging strategy using `flwr.server.strategy.FedAvg`.

### FedProx (`run_fedprox.py`)
FedProx strategy for handling severe non-IID data distributions ($\alpha = 0.1$) with a proximal term hyperparameter $\mu \in [0.01, 0.1, 1.0]$.

### Differential Privacy (`run_dp.py`)
Client-side fixed clipping and Gaussian noise injection via `flwr.server.strategy.DifferentialPrivacyClientSideFixedClipping` with noise multipliers $e \in [0.5, 1.0, 2.0]$.

### Secure Aggregation (`run_secagg.py`)
SecAgg+ cryptographic protocol using `SecAggPlusWorkflow` on server and `secaggplus_mod` on clients to protect updates from server inspection.
