# FedMedDx SQLite Experiment Database Schema

The SQLite database (`results/fedmeddx_experiments.db`) logs all federated execution runs, hospital client partitions, round-by-round training loss/accuracy, global validation benchmarks, and strategy hyperparameter sweeps.

---

## Entity Relationship & Table Schemas

### 1. `experiment_runs`
Stores metadata for each federated simulation run.

| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | TEXT PRIMARY KEY | Unique UUID or timestamp run ID |
| `modality` | TEXT | Modality config (`cxr`, `skin`, `mri`, `retina`, `dummy`) |
| `strategy` | TEXT | Strategy name (`FedAvg`, `FedProx`, `DP`, `SecAgg`) |
| `alpha` | REAL | Dirichlet non-IID concentration parameter |
| `num_hospitals` | INTEGER | Number of simulated hospital clients |
| `rounds` | INTEGER | Total federated communication rounds |
| `created_at` | TIMESTAMP | Experiment launch timestamp |

---

### 2. `round_metrics`
Stores round-by-round metric progression.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | Record ID |
| `run_id` | TEXT | Foreign key to `experiment_runs.run_id` |
| `round_num` | INTEGER | Current round number (1 to N) |
| `hospital_id` | INTEGER | Client hospital ID (-1 for global server aggregation) |
| `loss` | REAL | Validation or training loss |
| `accuracy` | REAL | Classification accuracy (0.0 to 1.0) |
| `f1_macro` | REAL | Macro-averaged F1 score |
| `auc_ovr` | REAL | One-vs-Rest ROC AUC score |

---

### 3. `hospital_distributions`
Stores hospital data partition class distributions.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | Record ID |
| `run_id` | TEXT | Foreign key to `experiment_runs.run_id` |
| `hospital_id` | INTEGER | Client hospital ID |
| `class_idx` | INTEGER | Class index |
| `sample_count` | INTEGER | Number of training samples assigned to hospital |
