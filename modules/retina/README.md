# Retinal Fundus (APTOS 2019) Modality — Person 4 Execution Guide

**Role Owner**: Person 4  
**Target Modality**: Retinal Fundus Diabetic Retinopathy Classification  
**Target Module File**: `modules/retina/retina_module.py`  
**Dataset Target Directory**: `data/retina/`

---

## 1. Dataset Overview & Acquisition

- **Dataset**: APTOS 2019 Blindness Detection
- **Classes (5)**: `0: No DR`, `1: Mild`, `2: Moderate`, `3: Severe`, `4: Proliferative DR`
- **Total Images**: 3,662 images
- **Kaggle Setup & Download Commands**:
  ```bash
  # IMPORTANT: Go to https://kaggle.com/c/aptos2019-blindness-detection on browser, click "Join Competition", and accept rules first!
  kaggle competitions download -c aptos2019-blindness-detection
  unzip aptos2019-blindness-detection.zip -d data/retina
  ```

---

## 2. Your Required Build Order (Day 1 – Day 6)

Follow these steps sequentially to complete your module before Day 7 Integration:

### Step 1: Data Inspection & Class Distribution
- Inspect `train.csv` mapping `id_code` to `diagnosis` (0 to 4).
- Note severe class imbalance: ~49% of dataset is class 0 (No DR).
- Print class distribution summary.

### Step 2: Specialized Retinal Preprocessing
- High-resolution fundus images contain large black background borders.
- Preprocessing recommendation: Crop uninformative black borders to isolate the circular retina region (Ben Graham method) and resize to 224x224.
- Apply ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Apply standard augmentations (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15)`).

### Step 3: Centralized ResNet18 Baseline
- Load `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
- Replace final FC layer: `model.fc = nn.Linear(model.fc.in_features, 5)`.
- Compute class weights using `sklearn.utils.class_weight.compute_class_weight` to address class imbalance.
- Train centralized model for 10-15 epochs using Adam optimizer (`lr=1e-4`) and `nn.CrossEntropyLoss(weight=class_weights)`.
- Record baseline validation accuracy, macro F1, and ROC AUC score.

### Step 4: Implement 5 Contract Functions in `modules/retina/retina_module.py`
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
python modules/retina/retina_module.py
```

---

## 3. GitHub Issue Assignments

Check GitHub Issues assigned to you:
- **Issue #18**: Day 1-2: Kaggle Competition Setup & Inspection
- **Issue #19**: Day 3-4: Preprocessing Pipeline & Centralized Baseline
- **Issue #20**: Day 5-6: CONTRACT.md Implementation & Standalone Test

---

## 4. Crucial Guidelines

- **Kaggle Rules Acceptance**: Ensure you click "Join Competition" on Kaggle website before downloading.
- **GPU Usage**: Pass `device="cuda"` if CUDA is available.
- **Do NOT Change Function Signatures**: Keep parameter types and return formats strictly aligned with `CONTRACT.md`.
- **Handoff Target**: Hand off a fully working `retina_module.py` by **Day 6 end**.
