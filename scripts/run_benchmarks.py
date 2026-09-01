"""
FedMedDx Automated Experiment Benchmark Suite.
Simulates and compares:
1. Isolated Local Training (No Federated Sharing)
2. FedRep Collaborative Training (Shared ResNet-18 Backbone + Local Heads + FedBN)
Across Non-IID Dirichlet alpha concentrations (0.1, 0.5, 1.0) for all disease modalities.
Logs all results to results/fedmeddx_experiments.db.
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated.db import init_db, log_experiment_run, log_round_metrics
from modules import covid_module, pneumonia_module, tb_module, pediatric_module

MODALITIES = {
    "covid": covid_module,
    "pneumonia": pneumonia_module,
    "tb": tb_module,
    "pediatric": pediatric_module,
}


def run_local_baseline(modality_name: str, num_hospitals: int = 3, rounds: int = 5, alpha: float = 0.5, device: str = "cpu"):
    """Simulate isolated local training where hospitals never communicate."""
    print(f"\n--- Running Isolated Local Baseline: Modality={modality_name} | Alpha={alpha} | Rounds={rounds} ---")
    module = MODALITIES[modality_name]
    partitions = module.get_hospital_partitions(num_hospitals=num_hospitals, alpha=alpha)

    # Each hospital has its own private model
    local_models = [module.get_model().to(device) for _ in range(num_hospitals)]
    run_id = log_experiment_run(modality_name, "Local-Only", alpha, num_hospitals, rounds)

    for r in range(1, rounds + 1):
        for h_id in range(num_hospitals):
            train_loader, val_loader = partitions[h_id]
            # Train locally
            _, num_samples, train_loss = module.train_one_round(
                local_models[h_id], train_loader, epochs=1, device=device
            )
            # Evaluate locally
            val_loss, metrics = module.evaluate(local_models[h_id], val_loader, device=device)
            log_round_metrics(run_id, r, h_id, val_loss, metrics["accuracy"], metrics["f1_macro"], metrics["auc_ovr"])

        print(f"[Local-Only] Modality={modality_name} | Round {r}/{rounds} Completed.")
    return run_id


def run_fedrep_simulation(modality_name: str, num_hospitals: int = 3, rounds: int = 5, alpha: float = 0.5, device: str = "cpu"):
    """Simulate FedRep collaborative training with shared backbone aggregation and local heads."""
    print(f"\n--- Running FedRep Collaborative Simulation: Modality={modality_name} | Alpha={alpha} | Rounds={rounds} ---")
    module = MODALITIES[modality_name]
    partitions = module.get_hospital_partitions(num_hospitals=num_hospitals, alpha=alpha)

    # Client models
    client_models = [module.get_model().to(device) for _ in range(num_hospitals)]
    run_id = log_experiment_run(modality_name, "FedRep+FedBN", alpha, num_hospitals, rounds)

    def is_shared(k):
        return ("fc" not in k) and ("bn" not in k) and ("downsample.1" not in k)

    for r in range(1, rounds + 1):
        client_weights = []
        sample_counts = []

        # 1. Local Training
        for h_id in range(num_hospitals):
            train_loader, val_loader = partitions[h_id]
            state_dict, num_samples, train_loss = module.train_one_round(
                client_models[h_id], train_loader, epochs=1, device=device
            )
            val_loss, metrics = module.evaluate(client_models[h_id], val_loader, device=device)
            log_round_metrics(run_id, r, h_id, val_loss, metrics["accuracy"], metrics["f1_macro"], metrics["auc_ovr"])

            client_weights.append({k: v.clone().cpu() for k, v in state_dict.items() if is_shared(k)})
            sample_counts.append(num_samples)

        # 2. Server FedAvg on Shared Backbone
        total_samples = sum(sample_counts)
        shared_keys = client_weights[0].keys()
        global_backbone = {}

        for k in shared_keys:
            global_backbone[k] = sum(
                client_weights[i][k] * (sample_counts[i] / total_samples)
                for i in range(num_hospitals)
            )

        # 3. Synchronize aggregated backbone into client models (preserving local head and batch norm)
        for h_id in range(num_hospitals):
            current_sd = client_models[h_id].state_dict()
            for k, v in global_backbone.items():
                current_sd[k] = v.to(device)
            client_models[h_id].load_state_dict(current_sd, strict=False)

        print(f"[FedRep] Modality={modality_name} | Round {r}/{rounds} Aggregated Successfully.")
    return run_id


def main():
    parser = argparse.ArgumentParser(description="FedMedDx Experiment Benchmark Suite")
    parser.add_argument("--rounds", type=int, default=5, help="Number of communication rounds")
    parser.add_argument("--hospitals", type=int, default=3, help="Number of hospital nodes")
    parser.add_argument("--modality", type=str, default="all", choices=["all", "covid", "pneumonia", "tb", "pediatric"])
    args = parser.parse_args()

    init_db()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting FedMedDx Benchmark Suite on device: {device.upper()}")

    target_modalities = list(MODALITIES.keys()) if args.modality == "all" else [args.modality]

    for mod in target_modalities:
        # Run comparison at alpha=0.5
        run_local_baseline(mod, num_hospitals=args.hospitals, rounds=args.rounds, alpha=0.5, device=device)
        run_fedrep_simulation(mod, num_hospitals=args.hospitals, rounds=args.rounds, alpha=0.5, device=device)

    print("\n=== Benchmark Suite Execution Complete! Metrics recorded in results/fedmeddx_experiments.db ===")


if __name__ == "__main__":
    main()
