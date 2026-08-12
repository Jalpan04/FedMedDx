# FedMedDx — Federated Learning for Multi-Modal Medical Diagnostics

FedMedDx is a federated diagnostic framework supporting 4 medical imaging modalities (Chest X-ray, Skin Lesion, Brain MRI, Retinal Fundus) with Non-IID Dirichlet hospital partitioning, explainable AI (Grad-CAM), Differential Privacy, Secure Aggregation, and interactive Streamlit web applications.

---

## 1. Team & Responsibilities

| Role | Owner | Sub-module Path | Primary Deliverables |
| :--- | :--- | :--- | :--- |
| **Federated Learning Core** | Jalpan (Core Lead) | `federated/` | Client wrapper, FedAvg, FedProx, DP, SecAgg+, Demo App & Dashboard |
| **Chest X-ray (CXR)** | Priyanka | `modules/cxr_module.py` | 4-class COVID/Pneumonia classifier, Dirichlet splits, Grad-CAM |
| **Skin Lesion** | Gargee | `modules/skin_cancer.py` | 7-class HAM10000 classifier, Grouped Lesion split, Grad-CAM |
| **Brain MRI** | Smit | `modules/mri_module.py` | 4-class Brain Tumor classifier, Continual Learning attempt, Grad-CAM |
| **Retinal Fundus** | Hirva | `modules/retina_module.py` | 5-grade DR Severity classifier, Ben Graham preprocessing, Grad-CAM |

---

## 2. Tech Stack & Environment Requirements

- **Python**: 3.10+
- **Federated Framework**: `flwr[simulation]` (Flower Virtual Client Engine)
- **Model Backbone**: `torch`, `torchvision` (ResNet18)
- **Explainability**: `pytorch-grad-cam`
- **Metrics**: `scikit-learn`
- **Web App & Dashboard**: `streamlit`
- **Dataset Retrieval**: `kaggle` CLI
- **Database Tracker**: `sqlite3`

---

## 3. Quick Setup Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/Jalpan04/FedMedDx.git
cd FedMedDx
```

### Step 2: Install Dependencies (Everyone)
```bash
pip install -r requirements.txt
```

### Step 3: Kaggle API Setup (Everyone)
1. Go to `kaggle.com/settings` -> API -> Create New Token (downloads `kaggle.json`).
2. Move token to home directory:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 4: Dataset Download Commands

#### Priyanka (Chest X-ray)
```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip -d data/cxr
```

#### Gargee (Skin Lesion)
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/skin
```

#### Smit (Brain MRI)
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/mri
```

#### Hirva (Retinal Fundus)
> Note: Accept competition rules at `https://kaggle.com/c/aptos2019-blindness-detection` first!
```bash
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d data/retina
```

---

## 4. Directory Structure

```
FedMedDx/
├── CONTRACT.md                  <- Immutable modality interface spec
├── README.md                    <- Main onboarding guide
├── requirements.txt             <- Project dependencies
├── docs/                        <- Team documentation wiki suite
│   ├── SETUP_GUIDE.md           <- Environment & CUDA setup
│   ├── DATASET_GUIDE.md         <- Download & preprocessing details
│   ├── FEDERATED_CORE.md        <- Flower simulation execution guide
│   ├── MODALITY_MODULES.md      <- Modality module developer guide
│   └── DATABASE_SCHEMA.md       <- Experiment tracking DB schema
├── data/                        <- Local datasets (gitignored)
│   ├── cxr/
│   ├── skin/
│   ├── mri/
│   └── retina/
├── modules/                     <- Modality implementations
│   ├── cxr_module.py            <- Priyanka
│   ├── skin_module.py           <- Gargee
│   ├── mri_module.py            <- Smit
│   └── retina_module.py         <- Hirva
├── federated/                   <- Core FL infrastructure
│   ├── db.py                    <- SQLite metric tracker
│   ├── dummy_module.py          <- Mock module for testing
│   ├── client_wrapper.py        <- Generic Flower NumPyClient wrapper
│   ├── run_fedavg.py            <- FedAvg simulation script
│   ├── run_fedprox.py           <- FedProx strategy script
│   ├── run_dp.py                <- Differential Privacy script
│   └── run_secagg.py            <- Secure Aggregation script
├── results/                     <- Output metrics, DB, & heatmaps
│   └── fedmeddx_experiments.db  <- SQLite metrics DB
└── demo/                        <- Streamlit applications
    ├── app.py                   <- Patient diagnostic app
    └── dashboard.py             <- Hospital analytics dashboard
```

---

## 5. Execution Commands

### Running Federated Simulation on Dummy Module (Day 1 Self-Test)
```bash
python federated/run_fedavg.py --modality dummy --num_hospitals 3 --rounds 3
```

### Running Federated Simulation on CXR Module
```bash
python federated/run_fedavg.py --modality cxr --num_hospitals 5 --alpha 0.5 --rounds 20
```

### Launching Streamlit Patient Demo
```bash
streamlit run demo/app.py
```

### Launching Hospital Dashboard
```bash
streamlit run demo/dashboard.py
```

### Distributed Multi-Machine Run via ngrok (Real Network)
**Server Setup (Jalpan's Machine)**:
```bash
# Start Flower Server
python -m federated.server --port 8080 --rounds 20 --min_clients 4

# Expose port 8080 via ngrok in another terminal
ngrok tcp 8080
```
Copy the public TCP address from ngrok output (e.g., `0.tcp.ngrok.io:12345`).

**Client Setup (Friends' Machines)**:
Replace `0.tcp.ngrok.io:12345` with the server's ngrok address:
```bash
# Priyanka (CXR)
python -m federated.client --server 0.tcp.ngrok.io:12345 --modality cxr --hospital_id 0

# Gargee (Skin)
python -m federated.client --server 0.tcp.ngrok.io:12345 --modality skin --hospital_id 1

# Smit (MRI)
python -m federated.client --server 0.tcp.ngrok.io:12345 --modality mri --hospital_id 2

# Hirva (Retina)
python -m federated.client --server 0.tcp.ngrok.io:12345 --modality retina --hospital_id 3
```

---

## 6. Development Workflow & Contribution Guidelines

1. **Contract Adherence**: Any module in `modules/` must follow `CONTRACT.md` strictly without changing function names or parameter order.
2. **GPU Acceleration**: Always check `torch.cuda.is_available()` and pass `"cuda"` as device to speed up local training rounds.
3. **Standalone Testing**: Run a standalone test on your module before handing it off to Core Lead:
   ```python
   python modules/skin_module.py
   ```
4. **Issue Tracking**: Refer to GitHub Issues (`https://github.com/Jalpan04/FedMedDx/issues`) for assigned tasks and sprint milestones.
