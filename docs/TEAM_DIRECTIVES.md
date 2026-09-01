# FedMedDx Team Member Action Directives

This document provides explicit, copy-pasteable instructions for each team member to complete their portion of the project and connect to the multi-machine distributed federated run.

---

## 1. Priyanka — Client 1 (COVID-19 Radiography)

### Overview
* **Assigned Modality**: 4-Class COVID-19 Radiography (`COVID-19`, `Normal`, `Lung Opacity`, `Viral Pneumonia`)
* **Target Output**: 4-class classification head (`512 x 4`)
* **Dataset**: `tawsifurrahman/covid19-radiography-database`

### Copy-Paste Message for Priyanka:
```text
Hi Priyanka,

We are preparing for our multi-machine distributed federated run. Here are the exact steps to set up and verify your COVID-19 module:

1. Dataset Setup (Run in terminal from project root):
   mkdir -p data/covid
   cd data/covid
   kaggle datasets download -d tawsifurrahman/covid19-radiography-database
   unzip -q covid19-radiography-database.zip
   cd ../..

2. Standalone Verification:
   Run the module test on your machine to verify that your GPU/CPU executes the 4-class pipeline:
   python -m modules.covid_module

3. Push Your Branch:
   git checkout -b priyanka-covid
   git add modules/covid_module.py
   git commit -m "Complete COVID-19 4-class module verification"
   git push origin priyanka-covid

4. Connect to Jalpan's Coordinator Server:
   When Jalpan starts the server, connect your client by running:
   python -m federated.client --server <JALPAN_IP>:8080 --modality covid --hospital_id 0
```

---

## 2. Gargee — Client 2 (Adult Pneumonia Detection)

### Overview
* **Assigned Modality**: Adult Pneumonia Detection (`NORMAL`, `PNEUMONIA`)
* **Target Output**: Binary classification head (`512 x 2`)
* **Dataset**: `paultimothymooney/chest-xray-pneumonia`

### Copy-Paste Message for Gargee:
```text
Hi Gargee,

Great job on the pneumonia notebook! Your test accuracy (97.24%) and AUC (0.9976) on Kaggle are excellent.

Here are your next steps to get connected to our live federated network:

1. Local Dataset Setup:
   mkdir -p data/pneumonia
   cd data/pneumonia
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip -q chest-xray-pneumonia.zip
   cd ../..

2. Local Verification:
   Verify that your local dataset is recognized by the module:
   python -m modules.pneumonia_module

3. Connect to the Live Coordinator:
   When Jalpan initiates the Flower server, run:
   python -m federated.client --server <JALPAN_IP>:8080 --modality pneumonia --hospital_id 1
```

---

## 3. Smit — Client 3 (Tuberculosis Screening)

### Overview
* **Assigned Modality**: Tuberculosis Screening (`Normal`, `Tuberculosis`)
* **Target Output**: Binary classification head (`512 x 2`)
* **Dataset**: `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`

### Copy-Paste Message for Smit:
```text
Hi Smit,

Great job implementing the batch logging, checkpoint saving, and real-image Grad-CAM preview directly in `modules/tb_module.py`.

Here are your next steps:

1. Local Dataset Download:
   mkdir -p data/tb
   cd data/tb
   kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
   unzip -q tuberculosis-tb-chest-xray-dataset.zip
   cd ../..

2. Run Full Local Verification:
   python -m modules.tb_module
   (This will train a test round and save `checkpoints/tb_model_checkpoint.pth`)

3. Connect to the Live Coordinator:
   When Jalpan starts the server, run:
   python -m federated.client --server <JALPAN_IP>:8080 --modality tb --hospital_id 2
```

---

## 4. Hirva — Client 4 (Pediatric Pneumonia)

### Overview
* **Assigned Modality**: Pediatric Pneumonia Detection (`NORMAL`, `PNEUMONIA`)
* **Target Output**: Binary classification head (`512 x 2`)
* **Dataset**: `tolgadincer/labeled-chest-xray-images`

### Copy-Paste Message for Hirva:
```text
Hi Hirva,

Great work on the pediatric pneumonia verification! Achieving 92.95% test accuracy and 0.9822 AUC on the Guangzhou Women's Hospital dataset is a strong result.

Here are your next steps:

1. Local Dataset Download:
   mkdir -p data/pediatric
   cd data/pediatric
   kaggle datasets download -d tolgadincer/labeled-chest-xray-images
   unzip -q labeled-chest-xray-images.zip
   cd ../..

2. Verify Local Execution:
   python -m modules.pediatric_module

3. Connect to the Live Coordinator:
   When Jalpan launches the server, run:
   python -m federated.client --server <JALPAN_IP>:8080 --modality pediatric --hospital_id 3
```

---

## 5. Jalpan — Core Lead (Coordinator Checklist)

### Coordinator Pre-Flight Checklist:
1. Obtain your local IP address:
   ```powershell
   ipconfig
   # Look for IPv4 Address under Wireless LAN adapter Wi-Fi (e.g., 192.168.1.X or 10.246.11.X)
   ```
2. Start the Flower Server:
   ```powershell
   python -m federated.server --port 8080 --rounds 20 --min_clients 4
   ```
3. Monitor client connection logs in the terminal as Priyanka, Gargee, Smit, and Hirva connect.
4. Launch the Clinician Web App Demo:
   ```powershell
   streamlit run demo/app.py
   ```
