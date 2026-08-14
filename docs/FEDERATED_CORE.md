# FedMedDx Federated Learning Core (FedRep + FedBN)

FedMedDx implements **Personalized Federated Learning (pFL)** combining **FedRep** (Federated Representation Learning) and **FedBN** (Federated Batch Normalization) over a unified Chest X-Ray feature representation.

---

## 1. Architectural Principles

### A. Decoupled Model Architecture
*   **Shared Backbone**: Layers `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4` of ResNet-18 (excluding Batch Normalization stats and final `fc` head).
*   **Local Head ($W_{fc}$)**: Unique task classification layers kept 100% local on each client PC (`checkpoints/client_{client_id}_head.pth`).
*   **Local Batch Norm (FedBN)**: Batch normalization layers are kept local to adapt to individual dataset contrast levels without causing drift.

### B. Communication Flow
1.  **Server Broadcast**: Central server broadcasts the aggregated backbone parameters.
2.  **Local Head Optimization (Phase 1)**: Client freezes the backbone and trains the local classification head.
3.  **Local Backbone Optimization (Phase 2)**: Client unfreezes the backbone and computes gradients on local CXR scans.
4.  **Client Upload**: Client uploads only the updated backbone weights to the server.
5.  **Server FedAvg**: Server computes weighted average of backbone weights across active clients.

---

## 2. Distributed Execution via Local Wi-Fi / LAN

### Coordinator (Jalpan)
1. Find your local IP address:
   ```powershell
   ipconfig
   ```
2. Start the Flower Server:
   ```bash
   python -m federated.server --port 8080 --rounds 20 --min_clients 4
   ```

### Distributed Clients (Priyanka, Gargee, Smit, Hirva)
Clients run pointing to the coordinator's current local IP address:
```bash
# Priyanka (COVID-19 Radiography)
python -m federated.client --server <SERVER_IP>:8080 --modality covid --hospital_id 0

# Gargee (Pneumonia)
python -m federated.client --server <SERVER_IP>:8080 --modality pneumonia --hospital_id 1

# Smit (Tuberculosis)
python -m federated.client --server <SERVER_IP>:8080 --modality tb --hospital_id 2

# Hirva (Pneumothorax)
python -m federated.client --server <SERVER_IP>:8080 --modality pneumothorax --hospital_id 3
```

---

## 3. Local Simulation Engine

To test end-to-end on a single machine across virtual hospital partitions:
```bash
python -m federated.run_fedavg --modality dummy --num_hospitals 2 --rounds 2
```
