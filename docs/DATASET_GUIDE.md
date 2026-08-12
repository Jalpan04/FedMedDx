# FedMedDx Dataset Acquisition & Preprocessing Guide

This document specifies data download paths, image resolutions, class distributions, and preprocessing specifications for each of the 4 medical imaging modalities.

---

## 1. Modality Specifications

### Modality 1: Chest X-ray (Person 1)
- **Dataset**: COVID-19 Radiography Database
- **Classes**: 4 classes (`COVID-19`, `Normal`, `Lung Opacity`, `Viral Pneumonia`)
- **Image Count**: ~21,165 images
- **Target Folder**: `data/cxr/`
- **Download Command**:
  ```bash
  kaggle datasets download -d tawsifurrahman/covid19-radiography-database
  unzip covid19-radiography-database.zip -d data/cxr
  ```

---

### Modality 2: Skin Lesion (Person 2)
- **Dataset**: HAM10000 (Skin Cancer MNIST)
- **Classes**: 7 classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`)
- **Image Count**: 10,015 images
- **Target Folder**: `data/skin/`
- **Crucial Requirement**: HAM10000 stores images across `HAM10000_images_part_1` and `HAM10000_images_part_2`. Combine image paths into one mapping using `HAM10000_metadata.csv`. Perform **Grouped Split by `lesion_id`** to prevent data leakage across train/test partitions!
- **Download Command**:
  ```bash
  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
  unzip skin-cancer-mnist-ham10000.zip -d data/skin
  ```

---

### Modality 3: Brain MRI (Person 3)
- **Dataset**: Brain Tumor MRI Dataset
- **Classes**: 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`)
- **Image Count**: 7,023 images
- **Target Folder**: `data/mri/`
- **Download Command**:
  ```bash
  kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
  unzip brain-tumor-mri-dataset.zip -d data/mri
  ```

---

### Modality 4: Retinal Fundus (Person 4)
- **Dataset**: APTOS 2019 Blindness Detection
- **Classes**: 5 severity grades (`0: No DR`, `1: Mild`, `2: Moderate`, `3: Severe`, `4: Proliferative DR`)
- **Image Count**: 3,662 images
- **Target Folder**: `data/retina/`
- **Preprocessing Recommendation**: Apply circular border cropping and Ben Graham contrast enhancement to remove uninformative black borders.
- **Download Command**:
  ```bash
  # Ensure you accept competition terms on website first!
  kaggle competitions download -c aptos2019-blindness-detection
  unzip aptos2019-blindness-detection.zip -d data/retina
  ```

---

## 2. Standardized PyTorch Image Pipeline

All modalities **MUST** normalize and resize images using the standard ImageNet pipeline:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 3. Non-IID Dirichlet Hospital Partitioning

Hospital datasets are created using a Dirichlet distribution over target class proportions:

```python
import numpy as np

def generate_dirichlet_splits(labels, num_hospitals, alpha):
    """Generates Non-IID data indices for each hospital client."""
    num_classes = len(np.unique(labels))
    label_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    
    hospital_indices = [[] for _ in range(num_hospitals)]
    for c, indices in enumerate(label_indices):
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_hospitals))
        proportions = (proportions * len(indices)).astype(int)
        
        # Split class indices across hospitals
        splits = np.split(indices, np.cumsum(proportions)[:-1])
        for h in range(num_hospitals):
            hospital_indices[h].extend(splits[h])
            
    return hospital_indices
```
