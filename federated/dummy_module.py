import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List
import numpy as np

class DummyModel(nn.Module):
    def __init__(self, in_features=10, hidden_dim=32, num_classes=2):
        super().__init__()
        # Backbone layers
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
        )
        # Local classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def get_model() -> torch.nn.Module:
    """Returns clean DummyModel instance with distinct backbone and head."""
    return DummyModel()

def get_hospital_partitions(num_hospitals: int = 3, alpha: float = 0.5) -> List[Tuple[DataLoader, DataLoader]]:
    """Returns mock train and val loaders for simulated hospitals."""
    partitions = []
    for h in range(num_hospitals):
        x_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        x_val = torch.randn(30, 10)
        y_val = torch.randint(0, 2, (30,))
        
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16, shuffle=False)
        partitions.append((train_loader, val_loader))
    return partitions

def train_one_round(model: torch.nn.Module, train_loader: DataLoader, epochs: int, device: str) -> Tuple[dict, int, float]:
    """Mock single-round local training."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(target)
            total_samples += len(target)
            
    avg_loss = total_loss / max(1, total_samples)
    return model.state_dict(), total_samples, avg_loss

def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, Dict[str, float]]:
    """Mock local validation evaluation."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * len(target)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += len(target)
            
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, {"accuracy": acc, "f1_macro": acc, "auc_ovr": acc}

def explain(model: torch.nn.Module, image: torch.Tensor, device: str) -> Tuple[int, float, np.ndarray]:
    """Mock explanation returning dummy overlay array."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        if image.dim() == 1:
            input_tensor = image.unsqueeze(0).to(device)
        else:
            input_tensor = image.to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
    dummy_overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    return pred.item(), conf.item(), dummy_overlay

if __name__ == "__main__":
    print("Testing dummy_module standalone contract compliance...")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = get_model()
    parts = get_hospital_partitions(2, 0.5)
    s, c, l = train_one_round(m, parts[0][0], 1, dev)
    loss, metrics = evaluate(m, parts[0][1], dev)
    p, conf, over = explain(m, torch.randn(10), dev)
    print("Contract verification complete. Metrics:", metrics)
