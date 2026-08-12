# Skin Lesion (HAM10000) Modality — Gargee Execution Guide

**Role Owner**: Gargee  
**Target Modality**: Skin Lesion Classification  
**Target Module File**: `modules/skin/skin_module.py`  
**Dataset Target Directory**: `data/skin/`

---

## 1. Dataset Overview & Acquisition

- **Dataset**: HAM10000 (Skin Cancer MNIST)
- **Classes (7)**: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
- **Total Images**: 10,015 images
- **Kaggle Setup & Download Commands**:
  ```bash
  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
  unzip skin-cancer-mnist-ham10000.zip -d data/skin
  ```

---

## 2. Your Required Build Order (Day 1 – Day 6)

Follow these steps sequentially to complete your module before Day 7 Integration:

### Step 1: Data Inspection & Image Path Mapping
- HAM10000 splits raw images across two subfolders (`HAM10000_images_part_1` and `HAM10000_images_part_2`).
- Parse `HAM10000_metadata.csv` to map `image_id` -> file path and diagnosis label (`dx`).
- Print class distribution counts.

### Step 2: Critical Data Leakage Protection (Grouped Lesion Split)
- **CRITICAL**: Multiple images belong to the same `lesion_id`.
- Do NOT perform a simple row-wise split! If you split by row, images of the same lesion will appear in both training and testing sets, causing artificially inflated accuracy.
- Use `sklearn.model_selection.GroupShuffleSplit` or `GroupKFold` grouped by `lesion_id` to ensure unique lesions stay strictly inside train, val, or test sets.

### Step 3: Preprocessing & Data Loaders
- Resize images to 224x224 pixels.
- Apply ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Apply standard augmentations (`RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(brightness=0.1, contrast=0.1)`).

### Step 4: Centralized ResNet18 Baseline
- Load `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
- Replace final FC layer: `model.fc = nn.Linear(model.fc.in_features, 7)`.
- Compute class weights using `sklearn.utils.class_weight.compute_class_weight` to address severe class imbalance (e.g. `nv` represents >65% of samples).
- Train centralized model for 10-15 epochs using Adam optimizer (`lr=1e-4`) and `nn.CrossEntropyLoss(weight=class_weights)`.
- Record baseline validation accuracy, macro F1, and ROC AUC score.

### Step 5: Implement 5 Contract Functions in `modules/skin/skin_module.py`
Implement the exact 5 functions specified in `CONTRACT.md`:
1. `get_model() -> torch.nn.Module`
2. `get_hospital_partitions(num_hospitals: int, alpha: float) -> List[Tuple[DataLoader, DataLoader]]`
   - Use Dirichlet distribution ($\alpha=0.5$ default, $\alpha=0.1$ severe non-IID).
   - Partition dataset across 4 to 5 hospital clients.
3. `train_one_round(model, train_loader, epochs, device) -> Tuple[dict, int, float]`
4. `evaluate(model, val_loader, device) -> Tuple[float, Dict[str, float]]`
   - Compute and return `loss`, `accuracy`, `f1_macro`, `auc_ovr`.
5. `explain(model, image, device) -> Tuple[int, float, np.ndarray]`
   - Compute Grad-CAM heatmap over `model.layer4[-1]`.
   - Return predicted class index, confidence score, and RGB overlay array `(224, 224, 3)`.

### Step 6: Standalone Module Self-Test
Run your module independently to verify all 5 functions execute cleanly:
```bash
python modules/skin/skin_module.py
```

---

## 3. GitHub Issue Assignments

Check GitHub Issues assigned to you:
- **Issue #11**: Day 1-2: Kaggle Setup & Grouped Lesion Inspection
- **Issue #12**: Day 3-4: Grouped Split Pipeline & Centralized Baseline
- **Issue #13**: Day 5-6: CONTRACT.md Implementation & Standalone Test

---

## 4. Crucial Guidelines

- **Group Split by Lesion ID**: Ensure no `lesion_id` overlap between train, val, and hospital splits.
- **GPU Usage**: Pass `device="cuda"` if CUDA is available.
- **Handoff Target**: Hand off a fully working `skin_module.py` by **Day 6 end**.
