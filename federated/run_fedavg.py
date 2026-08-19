"""
FedMedDx Multi-Hospital Local Simulation Runner (FedRep + FedBN).
Simulates federated training across virtual hospital clients sequentially on a single machine,
logging metrics to SQLite without requiring heavy Ray dependencies.
"""

import argparse
import importlib
import torch
from typing import Dict, List, Tuple
from federated.client_wrapper import FedRepClient
from federated.db import init_db, log_experiment_run, log_round_metrics

def main():
    parser = argparse.ArgumentParser(description="FedMedDx Local Federated Simulation Engine")
    parser.add_argument("--modality", type=str, default="dummy", choices=["covid", "pneumonia", "tb", "pediatric", "dummy"], help="Disease module to simulate")
    parser.add_argument("--num_hospitals", type=int, default=3, help="Number of virtual hospital partitions (default: 3)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID concentration parameter (default: 0.5)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds (default: 3)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing simulation: modality={args.modality}, hospitals={args.num_hospitals}, rounds={args.rounds}, device={device}")

    # Load module
    if args.modality == "dummy":
        module = importlib.import_module("federated.dummy_module")
    else:
        module = importlib.import_module(f"modules.{args.modality}_module")

    # Generate partitions
    partitions = module.get_hospital_partitions(num_hospitals=args.num_hospitals, alpha=args.alpha)
    
    # Initialize DB
    init_db()
    run_id = log_experiment_run(
        modality=args.modality,
        strategy="FedRep",
        alpha=args.alpha,
        num_hospitals=args.num_hospitals,
        rounds=args.rounds,
    )

    # Initialize client models
    clients = []
    for h_id in range(args.num_hospitals):
        model = module.get_model()
        train_loader, val_loader = partitions[h_id]
        c = FedRepClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            module_contract=module,
            client_id=f"sim_{args.modality}_{h_id}",
            device=device
        )
        clients.append(c)

    # Global backbone initialization
    global_backbone_params = clients[0].get_parameters({})

    print("\n--- Starting FedRep Simulation Rounds ---")
    for r in range(1, args.rounds + 1):
        print(f"\n[Round {r}/{args.rounds}]")
        client_updates = []
        total_samples = 0
        round_accuracies = []

        # Local Training (FedRep: local head + local backbone)
        for h_id, client in enumerate(clients):
            updated_params, num_samples, fit_res = client.fit(global_backbone_params, {"epochs": "1", "round": str(r)})
            loss, num_val, eval_metrics = client.evaluate(global_backbone_params, {})
            
            client_updates.append((updated_params, num_samples))
            total_samples += num_samples
            acc = eval_metrics.get("accuracy", 0.0)
            round_accuracies.append(acc)

            # Log to DB
            log_round_metrics(
                run_id=run_id,
                round_num=r,
                hospital_id=h_id,
                loss=loss,
                accuracy=acc,
                f1_macro=eval_metrics.get("f1_macro", acc),
                auc_ovr=eval_metrics.get("auc_ovr", acc),
            )
            print(f"  Hospital {h_id} -> Val Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}% | Samples: {num_samples}")

        # Server Aggregation (FedAvg on shared backbone only)
        num_params = len(global_backbone_params)
        aggregated_backbone = []
        for p_idx in range(num_params):
            weighted_param = sum(
                (samples / total_samples) * client_params[p_idx]
                for client_params, samples in client_updates
            )
            aggregated_backbone.append(weighted_param)

        global_backbone_params = aggregated_backbone
        avg_round_acc = sum(round_accuracies) / len(round_accuracies)
        print(f"[Round {r} Summary] Average Hospital Accuracy: {avg_round_acc * 100:.2f}%")

    print(f"\nSimulation run '{run_id}' finished successfully and logged to results/fedmeddx_experiments.db!")

if __name__ == "__main__":
    main()
