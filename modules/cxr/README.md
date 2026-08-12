# Chest X-Ray (CXR) Modality — Person 1 Execution Guide

**Role Owner**: Person 1  
**Target Modality**: Chest X-ray Classification  
**Target Module File**: `modules/cxr/cxr_module.py`  
**Dataset Target Directory**: `data/cxr/`

---

## 1. Dataset Overview & Acquisition

- **Dataset**: COVID-19 Radiography Database
- **Classes (4)**: `COVID-19`, `Normal`, `Lung Opacity`, `Viral Pneumonia`
- **Total Images**: ~21,165 images
- **Kaggle Setup & Download Commands**:
  ```bash
  # Ensure kaggle.json token is set up in ~/.kaggle/
  kaggle datasets download -d tawsifurrahman/covid19-radiography-database
  unzip covid19-radiography-database.zip -d data/cxr
  ```

---

## 2. Your Required Build Order (Day 1 – Day 6)

Follow these steps sequentially to complete your module before Day 7 Integration:

### Step 1: Data Inspection & Class Distribution
- Inspect downloaded images inside `data/cxr/`.
- Verify directory structure (e.g. subfolders for each of the 4 classes).
- Print a class distribution table to check sample counts per class.

### Step 2: Preprocessing & Data Loaders
- Resize images to 224x224 pixels.
- Normalize using ImageNet mean/std (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Apply standard augmentations (`RandomHorizontalFlip`, `RandomRotation(10)`).
- Perform a 70/15/15 stratified train/val/test split using `sklearn.model_selection.train_test_split`.

### Step 3: Centralized ResNet18 Baseline
- Load `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
- Replace final FC layer: `model.fc = nn.Linear(model.fc.in_features, 4)`.
- Train centralized model for 10-15 epochs using Adam optimizer (`lr=1e-4`) and CrossEntropyLoss.
- Record final validation accuracy, macro F1, and ROC AUC score. This is your benchmark comparison number.

### Step 4: Implement 5 Contract Functions in `modules/cxr/cxr_module.py`
You must implement the exact 5 functions specified in `CONTRACT.md`:
1. `get_model() -> torch.nn.Module`
2. `get_hospital_partitions(num_hospitals: int, alpha: float) -> List[Tuple[DataLoader, DataLoader]]`
   - Use Dirichlet distribution ($\alpha=0.5$ default, $\alpha=0.1$ severe non-IID) across class proportions.
   - Use 4 to 5 hospital client partitions.
3. `train_one_round(model, train_loader, epochs, device) -> Tuple[dict, int, float]`
4. `evaluate(model, val_loader, device) -> Tuple[float, Dict[str, float]]`
   - Compute and return `loss`, `accuracy`, `f1_macro`, `auc_ovr`.
5. `explain(model, image, device) -> Tuple[int, float, np.ndarray]`
   - Compute Grad-CAM heatmap over `model.layer4[-1]`.
   - Return prediction index, confidence probability, and RGB overlay numpy array `(224, 224, 3)`.

### Step 5: Standalone Module Self-Test
Run your module independently to verify all 5 functions execute cleanly without errors:
```bash
python modules/cxr/cxr_module.py
```

---

## 3. GitHub Issue Assignments

Check GitHub Issues assigned to you:
- **Issue #8**: Day 1-2: Kaggle Setup & Data Inspection
- **Issue #9**: Day 3-4: Preprocessing Pipeline & Centralized Baseline
- **Issue #10**: Day 5-6: CONTRACT.md Implementation & Standalone Test

---

## 4. Crucial Guidelines

- **GPU Usage**: Always pass `device="cuda"` if CUDA is available.
- **Do NOT Change Function Signatures**: Keep parameter types and return formats strictly aligned with `CONTRACT.md`.
- **Handoff Target**: Hand off a fully working `cxr_module.py` by **Day 6 end**.
