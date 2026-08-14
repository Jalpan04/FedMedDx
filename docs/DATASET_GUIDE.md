# FedMedDx Dataset Guide

FedMedDx utilizes four high-quality, standardized Chest X-Ray (CXR) datasets in standard PNG and JPEG formats.

---

## 1. COVID-19 Radiography Database (Priyanka — Client 1)
*   **Target Task**: 4-Class Classification (`COVID-19`, `Normal`, `Lung Opacity`, `Viral Pneumonia`)
*   **Format**: Standard PNG images
*   **Kaggle Download**:
    ```bash
    mkdir -p data/covid
    cd data/covid
    kaggle datasets download -d tawsifurrahman/covid19-radiography-database
    unzip -q covid19-radiography-database.zip
    cd ../..
    ```

---

## 2. Chest X-Ray Images (Pneumonia) (Gargee — Client 2)
*   **Target Task**: Binary Classification (`NORMAL` vs `PNEUMONIA`)
*   **Format**: Standard JPEG images
*   **Kaggle Download**:
    ```bash
    mkdir -p data/pneumonia
    cd data/pneumonia
    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
    unzip -q chest-xray-pneumonia.zip
    cd ../..
    ```

---

## 3. Tuberculosis (TB) Chest X-Ray Database (Smit — Client 3)
*   **Target Task**: Binary Classification (`Normal` vs `Tuberculosis`)
*   **Format**: Standard PNG images
*   **Kaggle Download**:
    ```bash
    mkdir -p data/tb
    cd data/tb
    kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
    unzip -q tuberculosis-tb-chest-xray-dataset.zip
    cd ../..
    ```

---

## 4. Chest X-Ray Pneumothorax Dataset (Hirva — Client 4)
*   **Target Task**: Binary Classification (`Normal` vs `Pneumothorax`)
*   **Format**: Standard PNG images
*   **Kaggle Download**:
    ```bash
    mkdir -p data/pneumothorax
    cd data/pneumothorax
    kaggle datasets download -d vsereda/chest-xray-pneumothorax-dataset
    unzip -q chest-xray-pneumothorax-dataset.zip
    cd ../..
    ```

---

## Unified Preprocessing Pipeline

Every dataset uses the same standardized image pipeline to ensure 100% compatibility:
*   RGB Conversion: Forced 3-channel RGB `Image.convert('RGB')`
*   Resolution: Resized to `(224, 224)`
*   Normalization: Standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
