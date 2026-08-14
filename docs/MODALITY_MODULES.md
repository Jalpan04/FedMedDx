# FedMedDx Disease Modules Guide

This guide details the implementation structure and datasets for the four Chest X-Ray disease modules.

---

## 1. COVID-19 Radiography Module (`modules/covid_module.py`)
*   **Owner**: Priyanka (Client 1)
*   **Target Task**: 4-Class Classification (`COVID-19`, `Normal`, `Lung Opacity`, `Viral Pneumonia`)
*   **Output Dimension**: $512 \times 4$
*   **Dataset Path**: `data/covid/`

---

## 2. Pneumonia Module (`modules/pneumonia_module.py`)
*   **Owner**: Gargee (Client 2)
*   **Target Task**: Binary Classification (`NORMAL` vs `PNEUMONIA`)
*   **Output Dimension**: $512 \times 2$
*   **Dataset Path**: `data/pneumonia/`

---

## 3. Tuberculosis Module (`modules/tb_module.py`)
*   **Owner**: Smit (Client 3)
*   **Target Task**: Binary Classification (`Normal` vs `Tuberculosis`)
*   **Output Dimension**: $512 \times 2$
*   **Dataset Path**: `data/tb/`

---

## 4. Pneumothorax Module (`modules/pneumothorax_module.py`)
*   **Owner**: Hirva (Client 4)
*   **Target Task**: Binary Classification (`Normal` vs `Pneumothorax`)
*   **Output Dimension**: $512 \times 2$
*   **Dataset Path**: `data/pneumothorax/`

---

## Development & Standalone Verification

Each module contains a self-test routine at the bottom of the file. To verify your module locally:
```bash
python -m modules.covid_module
python -m modules.pneumonia_module
python -m modules.tb_module
python -m modules.pneumothorax_module
```
