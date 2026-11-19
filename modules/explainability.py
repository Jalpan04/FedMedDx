"""
Native PyTorch Grad-CAM Explainability Engine for FedMedDx ResNet-18 Backbone.
Zero external C-extension or pip dependencies required.
Extracts gradient-weighted activation maps from layer4[-1].
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple


def explain_cxr(model: torch.nn.Module, image: torch.Tensor, device: str) -> Tuple[int, float, np.ndarray]:
    """
    Computes class activation map on layer4[-1] for a single CXR tensor.
    Returns: (predicted_class_idx, confidence_float, rgb_overlay_uint8)
    """
    model.to(device)
    model.eval()

    if image.dim() == 3:
        input_tensor = image.unsqueeze(0).to(device)
    else:
        input_tensor = image.to(device)

    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    with torch.enable_grad():
        x = input_tensor.clone().detach().to(device).requires_grad_(True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_idx = pred.item()
        score = logits[0, pred_idx]
        model.zero_grad()
        score.backward()

    h1.remove()
    h2.remove()

    try:
        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Reconstruct normalized RGB CXR
        rgb_img = input_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = np.clip(std * rgb_img + mean, 0, 1)

        cmap = plt.get_cmap("jet")
        heatmap = cmap(cam)[:, :, :3]
        overlay = (np.clip(0.5 * heatmap + 0.5 * rgb_img, 0, 1) * 255).astype(np.uint8)
    except Exception as e:
        print(f"[GradCAM Warning] Fallback overlay: {e}")
        overlay = np.zeros((224, 224, 3), dtype=np.uint8)

    return pred_idx, conf.item(), overlay
