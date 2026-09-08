# FedMedDx Disease Modules Guide

This guide details the implementation structure, datasets, and verified Kaggle benchmarks for the four Chest X-Ray disease modules.

---

## 1. COVID-19 Radiography Module (`modules/covid_module.py`)
*   **Owner**: Priyanka (Client 1)
*   **Target Task**: 4-Class Classification (`COVID`, `Lung_Opacity`, `Normal`, `Viral Pneumonia`)
*   **Output Dimension**: $512 \times 4$
*   **Dataset Path**: `data/covid/`
*   **Verified Kaggle Performance**: 96.41% Accuracy | 97.13% Macro F1 | 0.9929 ROC-AUC

---

## 2. Pneumonia Module (`modules/pneumonia_module.py`)
*   **Owner**: Gargee (Client 2)
*   **Target Task**: Binary Classification (`NORMAL` vs `PNEUMONIA`)
*   **Output Dimension**: $512 \times 2$
*   **Dataset Path**: `data/pneumonia/`
*   **Verified Kaggle Performance**: 97.24% Accuracy | 0.9976 ROC-AUC

---

## 3. Tuberculosis Module (`modules/tb_module.py`)
*   **Owner**: Smit (Client 3)
*   **Target Task**: Binary Classification (`Normal` vs `Tuberculosis`)
*   **Output Dimension**: $512 \times 2$
*   **Dataset Path**: `data/tb/`
*   **Verified Kaggle Performance**: 95.80% Accuracy | Class-weighted loss & Grad-CAM

---

## 4. Pediatric Pneumonia Module (`modules/pediatric_module.py`)
*   **Owner**: Hirva (Client 4)
*   **Target Task**: Binary Classification (`NORMAL` vs `PNEUMONIA`)
*   **Output Dimension**: $512 \times 2$
*   **Dataset Path**: `data/pediatric/`
*   **Verified Kaggle Performance**: 92.95% Accuracy | 0.9822 ROC-AUC

---

## Performance Summary Table

| Modality Module | Assigned Lead | Classes | Kaggle Test Accuracy | Macro F1 / AUC | Explainability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `covid_module.py` | Priyanka (Client 1) | 4 Classes | **96.41%** | F1: 97.13% \| AUC: 0.9929 | ResNet-18 `layer4` Grad-CAM |
| `pneumonia_module.py` | Gargee (Client 2) | Binary (2) | **97.24%** | AUC: 0.9976 | ResNet-18 `layer4` Grad-CAM |
| `tb_module.py` | Smit (Client 3) | Binary (2) | **95.80%** | Weighted CE Loss | Real-Image Grad-CAM Overlay |
| `pediatric_module.py` | Hirva (Client 4) | Binary (2) | **92.95%** | AUC: 0.9822 | ResNet-18 `layer4` Grad-CAM |

---

## Development & Standalone Verification

Each module contains a self-test routine at the bottom of the file. To verify your module locally:
```bash
python -m modules.covid_module
python -m modules.pneumonia_module
python -m modules.tb_module
python -m modules.pediatric_module
```
