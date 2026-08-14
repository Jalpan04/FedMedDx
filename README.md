# FedMedDx: Multi-Disease Representation Learning on Unified Chest Radiography

FedMedDx is a decentralized, privacy-preserving medical imaging platform. It leverages **Personalized Federated Learning (pFL)**, combining **FedRep** (Federated Representation Learning) and **FedBN** (Federated Batch Normalization) over a unified **Chest X-Ray (CXR)** modality.

Participating hospital nodes collaboratively train a shared **ResNet-18** feature-extraction backbone to learn universal pulmonary visual primitives (infiltrates, consolidations, opacities, pleural thickening) across diverse datasets, while keeping specialized disease classification heads local and private to each hospital.

---

## 1. Team Roles & Disease Allocations

| Role / Owner | Assigned Disease Task | Dataset | Target Output |
| :--- | :--- | :--- | :--- |
| **Jalpan (Core Lead)** | Federated Core Coordinator | Server & Client Engine | FedRep + FedBN Aggregator |
| **Priyanka (Client 1)** | COVID-19 Radiography | `tawsifurrahman/covid19-radiography-database` | 4 Classes (`COVID-19`, `Normal`, `Lung Opacity`, `Viral Pneumonia`) |
| **Gargee (Client 2)** | Pneumonia Detection | `paultimothymooney/chest-xray-pneumonia` | Binary (`NORMAL`, `PNEUMONIA`) |
| **Smit (Client 3)** | Tuberculosis Screening | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | Binary (`Normal`, `Tuberculosis`) |
| **Hirva (Client 4)** | Pneumothorax Detection | `vsereda/chest-xray-pneumothorax-dataset` | Binary (`Normal`, `Pneumothorax`) |

---

## 2. Technical Architecture: FedRep + FedBN

```
                      ┌────────────────────────────────────────┐
                      │    Central Server (Jalpan)             │
                      │  Aggregates ResNet-18 Backbone via     │
                      │  FedRep (Excludes Heads and Batch Norm)│
                      └───────────────────┬────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│   Priyanka       │            │   Gargee         │            │   Smit / Hirva   │
│ Task: COVID-19   │            │ Task: Pneumonia  │            │ Task: TB /       │
│ (4-Class Head)   │            │ (Binary Head)    │            │ Pneumothorax     │
│ Local Checkpoint │            │ Local Checkpoint │            │ Local Checkpoint │
└──────────────────┘            └──────────────────┘            └──────────────────┘
```

1. **Shared Global Backbone**: ResNet-18 convolutional layers (`conv1` through `layer4`) learn universal medical image features collaboratively.
2. **Persistent Local Heads**: Task classification layers ($W_{fc}$) remain 100% private to each client PC and are cached across communication rounds.
3. **Local Batch Normalization (FedBN)**: Batch norm running statistics are kept local to eliminate domain drift across disparate imaging sources.

---

## 3. Quickstart & Installation

### Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/Jalpan04/FedMedDx.git
cd FedMedDx

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Download Commands

### Priyanka (COVID-19 Radiography)
```bash
mkdir -p data/covid
cd data/covid
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip -q covid19-radiography-database.zip
cd ../..
```

### Gargee (Pneumonia)
```bash
mkdir -p data/pneumonia
cd data/pneumonia
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -q chest-xray-pneumonia.zip
cd ../..
```

### Smit (Tuberculosis)
```bash
mkdir -p data/tb
cd data/tb
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
unzip -q tuberculosis-tb-chest-xray-dataset.zip
cd ../..
```

### Hirva (Pneumothorax)
```bash
mkdir -p data/pneumothorax
cd data/pneumothorax
kaggle datasets download -d vsereda/chest-xray-pneumothorax-dataset
unzip -q chest-xray-pneumothorax-dataset.zip
cd ../..
```

---

## 5. Execution & Federated Training

### Distributed Multi-Machine Run via Local Wi-Fi / LAN

**Server Setup (Jalpan's Machine)**:
1. Find your server's Wi-Fi IPv4 address using `ipconfig`.
2. Start the Flower Server:
   ```bash
   python -m federated.server --port 8080 --rounds 20 --min_clients 4
   ```

**Client Setup (Friends' Machines)**:
Replace `<SERVER_IP>` with Jalpan's current local Wi-Fi IP address:
```bash
# Priyanka
python -m federated.client --server <SERVER_IP>:8080 --modality covid --hospital_id 0

# Gargee
python -m federated.client --server <SERVER_IP>:8080 --modality pneumonia --hospital_id 1

# Smit
python -m federated.client --server <SERVER_IP>:8080 --modality tb --hospital_id 2

# Hirva
python -m federated.client --server <SERVER_IP>:8080 --modality pneumothorax --hospital_id 3
```

### Single-Machine Local Simulation
```bash
python -m federated.run_fedavg --modality dummy --num_hospitals 2 --rounds 2
```

---

## 6. Repository Structure

```
FedMedDx/
├── CONTRACT.md                # Interface contract for disease modules
├── README.md                  # Team onboarding and setup guide
├── requirements.txt           # Project dependencies
├── docs/                      # Documentation guides (FedRep, Datasets, Setup)
├── data/                      # Dataset directories (ignored by git)
│   ├── covid/
│   ├── pneumonia/
│   ├── tb/
│   └── pneumothorax/
├── federated/                 # Federated learning core engine
│   ├── client.py              # Distributed client entry point
│   ├── client_wrapper.py      # FedRep + FedBN client wrapper with checkpointing
│   ├── db.py                  # Experiment metrics logging database
│   ├── dummy_module.py        # Contract-compliant verification stub
│   ├── run_fedavg.py          # Single-machine simulation engine
│   └── server.py              # Standalone Flower server
├── modules/                   # Disease modality modules
│   ├── covid_module.py        # Priyanka (COVID-19 Radiography)
│   ├── pneumonia_module.py    # Gargee (Pneumonia Detection)
│   ├── tb_module.py           # Smit (Tuberculosis Screening)
│   └── pneumothorax_module.py # Hirva (Pneumothorax Detection)
├── checkpoints/               # Persisted local head checkpoints
└── results/                   # SQLite metrics database
```
