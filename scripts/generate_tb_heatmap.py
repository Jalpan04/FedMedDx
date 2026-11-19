"""
Generate high-resolution Grad-CAM heatmaps for Tuberculosis (TB) Chest X-Rays
matching the multi-sample grid format:
- 1x4 horizontal comparison grid with randomly selected CXR images on each run
- Attention heatmap overlaid on CXR
- Dual-line colored header per sample:
    True: <LABEL>
    Pred: <LABEL> (<CONFIDENCE>%)
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = ["NORMAL", "TUBERCULOSIS"]
DATA_DIR = os.path.join("data", "tb")
OUTPUT_PATH = "tb_heatmap.png"


def get_model(device: str) -> nn.Module:
    """Initialize ResNet18 and load available trained checkpoint."""
    try:
        model = models.resnet18(weights=None)
    except Exception:
        model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Check for saved checkpoints
    ckpt_candidates = [
        "tb_model_checkpoint.pth",
        os.path.join("checkpoints", "tb_model_checkpoint.pth"),
        os.path.join("checkpoints", "best_full_checkpoint.pth")
    ]
    loaded = False
    for ckpt in ckpt_candidates:
        if os.path.exists(ckpt):
            try:
                state_dict = torch.load(ckpt, map_location=device)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model_dict = model.state_dict()
                filtered = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(filtered)
                model.load_state_dict(model_dict)
                print(f"[+] Loaded trained model weights from: {ckpt} ({len(filtered)} layers)")
                loaded = True
                break
            except Exception as e:
                print(f"[-] Could not load {ckpt}: {e}")

    if not loaded:
        print("[!] No pre-trained checkpoint found; using model initialization.")

    model.to(device)
    model.eval()
    return model


def explain_gradcam(model: nn.Module, input_tensor: torch.Tensor, device: str):
    """Compute Grad-CAM activations on model layer4[-1] using pure PyTorch."""
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    with torch.enable_grad():
        x = input_tensor.clone().detach().to(device).requires_grad_(True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_idx = pred.item()
        confidence = conf.item()

        score = logits[0, pred_idx]
        score.backward()

    h_fwd.remove()
    h_bwd.remove()

    # Calculate CAM heatmap
    act = activations[0].detach()
    grad = gradients[0].detach()
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Reconstruct displayable RGB image
    img_np = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    # Jet colormap blending
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[:, :, :3]
    overlay = 0.5 * heatmap + 0.5 * img_np
    overlay = np.clip(overlay, 0, 1)

    return pred_idx, confidence, overlay


def load_test_samples(num_samples: int = 4):
    """Load randomly sampled test images from disk (balanced across classes)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    samples = []
    # Search for all image files per class
    for label_idx, cls_name in enumerate(["Normal", "Tuberculosis"]):
        patterns = [
            os.path.join(DATA_DIR, cls_name, "*.*"),
            os.path.join(DATA_DIR, cls_name.lower(), "*.*"),
            os.path.join(DATA_DIR, "*", cls_name, "*.*"),
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        valid_files = [f for f in set(files) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Randomly choose images for this class
        target_k = max(1, num_samples // 2)
        if len(valid_files) >= target_k:
            selected = random.sample(valid_files, target_k)
        else:
            selected = valid_files

        for f in selected:
            try:
                img = Image.open(f).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                samples.append((tensor, label_idx, os.path.basename(f)))
            except Exception:
                pass

    # If dataset has one class or few images, fill remainder randomly
    if len(samples) < num_samples:
        all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.*"), recursive=True)
        img_files = [f for f in set(all_files) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(img_files)
        for f in img_files:
            if len(samples) >= num_samples:
                break
            lbl = 1 if "tuber" in f.lower() else 0
            try:
                img = Image.open(f).convert("RGB")
                samples.append((transform(img).unsqueeze(0), lbl, os.path.basename(f)))
            except Exception:
                pass

    # Fallback to synthetic if no images found
    if len(samples) < num_samples:
        for i in range(num_samples - len(samples)):
            samples.append((torch.randn(1, 3, 224, 224), i % 2, f"synthetic_{i}"))

    # Randomly shuffle the final selection of 4 images
    random.shuffle(samples)
    return samples[:num_samples]


def generate_figure():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running Grad-CAM figure generation on device: {device}")

    model = get_model(device)
    samples = load_test_samples(num_samples=4)
    n_samples = len(samples)

    fig, axes = plt.subplots(1, n_samples, figsize=(4.2 * n_samples, 4.2), dpi=300)
    if n_samples == 1:
        axes = [axes]

    print("[*] Randomly selected scans:")
    for idx, (img_tensor, true_label, fname) in enumerate(samples):
        pred_idx, conf, overlay = explain_gradcam(model, img_tensor, device)
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_idx]
        print(f"    - Subplot {idx+1}: {fname} | True: {true_name} | Pred: {pred_name} ({conf*100:.1f}%)")

        axes[idx].imshow(overlay)
        axes[idx].axis("off")

        # Green header if correct, dark red if misclassified
        color = "#15803d" if pred_idx == true_label else "#b91c1c"
        axes[idx].set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf * 100:.1f}%)",
            fontsize=10.5,
            fontweight="semibold",
            color=color,
            pad=8
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[SUCCESS] High-res Grad-CAM heatmap saved to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    generate_figure()
