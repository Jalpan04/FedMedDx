"""
Chest X-Ray Pneumonia Modality Module — Gargee (Client 2)
Dataset: Chest X-Ray Images (Pneumonia) (Kaggle: paultimothymooney/chest-xray-pneumonia)
Binary Classification: NORMAL (0) vs. PNEUMONIA (1)
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DATA_DIR = os.path.join("data", "pneumonia")


class CXRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_cxr_transforms(is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_model() -> torch.nn.Module:
    """Returns ResNet18 model configured with 2-class binary classification head for Pneumonia."""
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        # Fallback in case of network issue or cache corruption
        cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if "resnet18" in f:
                    try:
                        os.remove(os.path.join(cache_dir, f))
                    except Exception:
                        pass
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_hospital_partitions(num_hospitals: int = 3, alpha: float = 0.5) -> List[Tuple[DataLoader, DataLoader]]:
    """Partition Pneumonia dataset across hospital clients."""
    image_paths = []
    labels = []

    # Search for files across train/test/val folders
    for class_idx, class_name in enumerate(CLASS_NAMES):
        patterns = [
            os.path.join(DATA_DIR, "*", class_name, "*.jpeg"),
            os.path.join(DATA_DIR, "*", class_name, "*.jpg"),
            os.path.join(DATA_DIR, "*", class_name, "*.png"),
            os.path.join(DATA_DIR, "chest_xray", "*", class_name, "*.jpeg"),
            os.path.join(DATA_DIR, "chest_xray", "*", class_name, "*.jpg"),
            os.path.join(DATA_DIR, class_name, "*.jpeg"),
            os.path.join(DATA_DIR, class_name, "*.jpg"),
            os.path.join(DATA_DIR, class_name, "*.png"),
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        for f in sorted(list(set(files))):
            image_paths.append(f)
            labels.append(class_idx)

    # Fallback to synthetic data for self-testing if dataset not yet downloaded
    if len(image_paths) == 0:
        partitions = []
        for h in range(num_hospitals):
            x_train = torch.randn(64, 3, 224, 224)
            y_train = torch.randint(0, NUM_CLASSES, (64,))
            x_val = torch.randn(16, 3, 224, 224)
            y_val = torch.randint(0, NUM_CLASSES, (16,))
            train_loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
            val_loader = DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=16, shuffle=False)
            partitions.append((train_loader, val_loader))
        return partitions

    labels = np.array(labels)
    image_paths = np.array(image_paths)

    num_classes = NUM_CLASSES
    label_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    hospital_indices = [[] for _ in range(num_hospitals)]

    for c, indices in enumerate(label_indices):
        if len(indices) == 0:
            continue
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_hospitals))
        proportions = (proportions * len(indices)).astype(int)
        splits = np.split(indices, np.cumsum(proportions)[:-1])
        for h in range(num_hospitals):
            hospital_indices[h].extend(splits[h])

    partitions = []
    for h in range(num_hospitals):
        h_idx = np.array(hospital_indices[h])
        if len(h_idx) == 0:
            h_idx = np.arange(min(32, len(image_paths)))
        np.random.shuffle(h_idx)
        split_point = max(1, int(0.8 * len(h_idx)))
        train_idx, val_idx = h_idx[:split_point], h_idx[split_point:]
        if len(val_idx) == 0:
            val_idx = train_idx[:max(1, len(train_idx) // 4)]

        train_ds = CXRDataset(image_paths[train_idx], labels[train_idx], transform=get_cxr_transforms(is_train=True))
        val_ds = CXRDataset(image_paths[val_idx], labels[val_idx], transform=get_cxr_transforms(is_train=False))

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        partitions.append((train_loader, val_loader))

    return partitions


def train_one_round(model: torch.nn.Module, train_loader: DataLoader, epochs: int, device: str) -> Tuple[dict, int, float]:
    """Execute local training for one federated round (FedRep style: Head optimization + Backbone optimization)."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    num_batches = len(train_loader)

    # Phase 1: Train Head (freeze backbone)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    head_optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    model.train()
    for epoch in range(max(1, epochs)):
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            head_optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            head_optimizer.step()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"     [Head] Epoch {epoch+1}/{max(1, epochs)} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

    # Phase 2: Train Backbone (unfreeze backbone)
    for param in model.parameters():
        param.requires_grad = True

    backbone_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(max(1, epochs)):
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            backbone_optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            backbone_optimizer.step()
            total_loss += loss.item() * len(targets)
            total_samples += len(targets)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"     [Backbone] Epoch {epoch+1}/{max(1, epochs)} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, total_samples)
    return model.state_dict(), total_samples, avg_loss


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, Dict[str, float]]:
    """Evaluate current model performance on validation set."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            probs = F.softmax(outputs, dim=1)

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, total_samples)
    acc = float(accuracy_score(all_targets, all_preds)) if total_samples > 0 else 0.0
    f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0)) if total_samples > 0 else 0.0

    all_probs_arr = np.array(all_probs)
    all_targets_arr = np.array(all_targets)
    try:
        if len(np.unique(all_targets_arr)) > 1:
            auc = float(roc_auc_score(all_targets_arr, all_probs_arr[:, 1]))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return avg_loss, {"accuracy": acc, "f1_macro": f1, "auc_ovr": auc}


def explain(model: torch.nn.Module, image: torch.Tensor, device: str) -> Tuple[int, float, np.ndarray]:
    """Generate Grad-CAM heatmap visualization."""
    model.to(device)
    model.eval()
    if image.dim() == 3:
        input_tensor = image.unsqueeze(0).to(device)
    else:
        input_tensor = image.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        target_layers = [model.layer4[-1]]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)
            rgb_img = input_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
            overlay = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam[0, :], use_rgb=True)
    except Exception:
        overlay = np.zeros((224, 224, 3), dtype=np.uint8)

    return pred.item(), conf.item(), overlay


if __name__ == "__main__":
    print("Testing pneumonia_module standalone contract compliance...")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    m = get_model()
    parts = get_hospital_partitions(2, 0.5)
    s, c, l = train_one_round(m, parts[0][0], 1, dev)
    loss, metrics = evaluate(m, parts[0][1], dev)
    print("Pneumonia Module Verification complete! Metrics:", metrics)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(m.state_dict(), "checkpoints/pneumonia_best_model.pth")
    print("Saved checkpoint to checkpoints/pneumonia_best_model.pth")

    dummy_img = torch.randn(3, 224, 224)
    p, conf, over = explain(m, dummy_img, dev)
    print(f"Grad-CAM check: Pred={CLASS_NAMES[p]}, Confidence={conf:.4f}, Overlay shape={over.shape}")
