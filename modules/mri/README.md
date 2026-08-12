# Brain MRI Modality — Smit Execution Guide

**Role Owner**: Smit  
**Target Modality**: Brain Tumor MRI Classification (+ Continual Learning stretch goal)  
**Target Module File**: `modules/mri/mri_module.py`  
**Dataset Target Directory**: `data/mri/`

---

## 1. Dataset Overview & Acquisition

- **Dataset**: Brain Tumor MRI Dataset
- **Classes (4)**: `glioma`, `meningioma`, `notumor`, `pituitary`
- **Total Images**: 7,023 images
- **Kaggle Setup & Download Commands**:
  ```bash
  kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
  unzip brain-tumor-mri-dataset.zip -d data/mri
  ```

---

## 2. Your Required Build Order (Day 1 – Day 6)

Follow these steps sequentially to complete your module before Day 7 Integration:

### Step 1: Data Inspection & Class Distribution
- Inspect downloaded image directories (`Training` and `Testing` folders).
- Count samples across all 4 tumor categories.
- Print class distribution counts.

### Step 2: Preprocessing & Data Loaders
- Resize images to 224x224 pixels.
- Normalize using ImageNet mean/std (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Apply standard augmentations (`RandomHorizontalFlip`, `RandomRotation(10)`).
- Perform a 70/15/15 stratified train/val/test split.

### Step 3: Centralized ResNet18 Baseline
- Load `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
- Replace final FC layer: `model.fc = nn.Linear(model.fc.in_features, 4)`.
- Train centralized model for 10-15 epochs using Adam optimizer (`lr=1e-4`) and CrossEntropyLoss.
- Record baseline validation accuracy, macro F1, and ROC AUC score.

### Step 4: Implement 5 Contract Functions in `modules/mri/mri_module.py`
Implement the exact 5 functions specified in `CONTRACT.md`:
1. `get_model() -> torch.nn.Module`
2. `get_hospital_partitions(num_hospitals: int, alpha: float) -> List[Tuple[DataLoader, DataLoader]]`
   - Use Dirichlet distribution ($\alpha=0.5$ default, $\alpha=0.1$ severe non-IID).
   - Use 3 hospital client partitions (given dataset size).
3. `train_one_round(model, train_loader, epochs, device) -> Tuple[dict, int, float]`
4. `evaluate(model, val_loader, device) -> Tuple[float, Dict[str, float]]`
   - Compute and return `loss`, `accuracy`, `f1_macro`, `auc_ovr`.
5. `explain(model, image, device) -> Tuple[int, float, np.ndarray]`
   - Compute Grad-CAM heatmap over `model.layer4[-1]`.
   - Return predicted class index, confidence score, and RGB overlay array `(224, 224, 3)`.

### Step 5: Standalone Module Self-Test
Run your module independently to verify all 5 functions execute cleanly:
```bash
python modules/mri/mri_module.py
```

### Step 6: Continual Learning Attempt (Day 11-12 Stretch Goal)
- If Tier 0/1 federated milestones are on schedule, build a task-incremental learning extension evaluating catastrophic forgetting when new MRI tumor classes arrive sequentially.

---

## 3. GitHub Issue Assignments

Check GitHub Issues assigned to you:
- **Issue #14**: Day 1-2: Kaggle Setup & Data Inspection
- **Issue #15**: Day 3-4: Centralized Baseline & Local Pipeline
- **Issue #16**: Day 5-6: CONTRACT.md Implementation & Standalone Test
- **Issue #17**: Day 11-12: Continual Learning Stretch Goal

---

## 4. Crucial Guidelines

- **GPU Usage**: Pass `device="cuda"` if CUDA is available.
- **Do NOT Change Function Signatures**: Keep parameter types and return formats strictly aligned with `CONTRACT.md`.
- **Handoff Target**: Hand off a fully working `mri_module.py` by **Day 6 end**.
