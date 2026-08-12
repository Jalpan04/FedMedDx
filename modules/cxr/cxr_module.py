"""
Chest X-Ray (CXR) Modality Module — Person 1

This file MUST be implemented by Person 1.
Follow instructions in modules/cxr/README.md and CONTRACT.md.
Do NOT change function signatures or return types.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import numpy as np

def get_model() -> torch.nn.Module:
    """Return a fresh ResNet18 model configured for 4-class CXR classification."""
    raise NotImplementedError("Person 1: Implement get_model() in modules/cxr/cxr_module.py")

def get_hospital_partitions(num_hospitals: int, alpha: float) -> List[Tuple[DataLoader, DataLoader]]:
    """Partition COVID-19 Radiography dataset into Non-IID hospital splits using Dirichlet(alpha)."""
    raise NotImplementedError("Person 1: Implement get_hospital_partitions() in modules/cxr/cxr_module.py")

def train_one_round(model: torch.nn.Module, train_loader: DataLoader, epochs: int, device: str) -> Tuple[dict, int, float]:
    """Execute local training on one hospital partition for one federated round."""
    raise NotImplementedError("Person 1: Implement train_one_round() in modules/cxr/cxr_module.py")

def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, Dict[str, float]]:
    """Evaluate current model performance on hospital validation set."""
    raise NotImplementedError("Person 1: Implement evaluate() in modules/cxr/cxr_module.py")

def explain(model: torch.nn.Module, image: torch.Tensor, device: str) -> Tuple[int, float, np.ndarray]:
    """Generate Grad-CAM activation heatmap overlay for a single CXR image."""
    raise NotImplementedError("Person 1: Implement explain() in modules/cxr/cxr_module.py")

if __name__ == "__main__":
    print("Run standalone self-test after implementing all 5 functions.")
