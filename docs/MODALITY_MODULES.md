# FedMedDx Modality Module Developer Guide

This guide is for Priyanka, Gargee, Smit, and Hirva building `cxr_module.py`, `skin_module.py`, `mri_module.py`, and `retina_module.py`.

---

## 1. Module Structure Template

Each team member's module should follow this structural outline:

```python
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple, List
import numpy as np

# 1. get_model()
def get_model() -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_classes = 4 # Adjust per modality
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 2. get_hospital_partitions()
def get_hospital_partitions(num_hospitals: int, alpha: float):
    # Implement dataset loading & Dirichlet partitioning
    pass

# 3. train_one_round()
def train_one_round(model, train_loader, epochs, device):
    # Implement standard PyTorch training loop
    pass

# 4. evaluate()
def evaluate(model, val_loader, device):
    # Calculate loss, accuracy, f1_macro, auc_ovr
    pass

# 5. explain()
def explain(model, image, device):
    # Generate Grad-CAM visualization
    pass
```

---

## 2. Standalone Self-Test Requirement (Day 6 Deadline)

Before submitting your module on Day 7 Integration Day, add a standalone test block at the bottom of your file:

```python
if __name__ == "__main__":
    print("Testing standalone module functions...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test 1: get_model
    model = get_model().to(device)
    print("get_model OK. FC out features:", model.fc.out_features)
    
    # Test 2: get_hospital_partitions
    partitions = get_hospital_partitions(num_hospitals=2, alpha=0.5)
    print(f"get_hospital_partitions OK. Created {len(partitions)} hospital splits.")
    
    # Test 3: train_one_round
    state_dict, count, loss = train_one_round(model, partitions[0][0], epochs=1, device=device)
    print(f"train_one_round OK. Trained on {count} samples, loss: {loss:.4f}")
    
    # Test 4: evaluate
    val_loss, metrics = evaluate(model, partitions[0][1], device=device)
    print("evaluate OK. Metrics:", metrics)
    
    # Test 5: explain
    dummy_img = torch.randn(3, 224, 224)
    pred, conf, overlay = explain(model, dummy_img, device)
    print(f"explain OK. Prediction: {pred}, Confidence: {conf:.2f}, Overlay shape: {overlay.shape}")
```
