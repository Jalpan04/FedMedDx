"""
Skin Lesion (Skin Cancer HAM10000) Modality Module — Person 2

This file MUST be implemented by Person 2.
Follow instructions in modules/skin/README.md and CONTRACT.md.
Do NOT change function signatures or return types.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import numpy as np

def get_model() -> torch.nn.Module:
    """Return a fresh ResNet18 model configured for 7-class HAM10000 classification."""
    raise NotImplementedError("Person 2: Implement get_model() in modules/skin/skin_module.py")

def get_hospital_partitions(num_hospitals: int, alpha: float) -> List[Tuple[DataLoader, DataLoader]]:
    """Partition HAM10000 dataset into Non-IID hospital splits using Dirichlet(alpha) & Grouped Lesion split."""
    raise NotImplementedError("Person 2: Implement get_hospital_partitions() in modules/skin/skin_module.py")

def train_one_round(model: torch.nn.Module, train_loader: DataLoader, epochs: int, device: str) -> Tuple[dict, int, float]:
    """Execute local training on one hospital partition for one federated round."""
    raise NotImplementedError("Person 2: Implement train_one_round() in modules/skin/skin_module.py")

def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, Dict[str, float]]:
    """Evaluate current model performance on hospital validation set."""
    raise NotImplementedError("Person 2: Implement evaluate() in modules/skin/skin_module.py")

def explain(model: torch.nn.Module, image: torch.Tensor, device: str) -> Tuple[int, float, np.ndarray]:
    """Generate Grad-CAM activation heatmap overlay for a single skin lesion image."""
    raise NotImplementedError("Person 2: Implement explain() in modules/skin/skin_module.py")

if __name__ == "__main__":
    print("Run standalone self-test after implementing all 5 functions.")
