import argparse
import importlib
import os
import torch
import flwr as fl
from federated.client_wrapper import FlowerNumPyClient
from federated.db import log_experiment_run, log_round_metrics

def main():
    parser = argparse.ArgumentParser(description="Run FedAvg simulation for FedMedDx modality.")
    parser.add_argument("--modality", type=str, default="dummy", choices=["cxr", "skin", "mri", "retina", "dummy"],
                        help="Modality module name to simulate.")
    parser.add_argument("--num_hospitals", type=int, default=3, help="Number of simulated hospital clients.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter.")
    parser.add_argument("--rounds", type=int, default=3, help="Total federated communication rounds.")
    args = parser.parse_args()

    print(f"Starting FedAvg Simulation: modality={args.modality}, hospitals={args.num_hospitals}, alpha={args.alpha}, rounds={args.rounds}")
    
    # 1. Dynamically import selected modality module
    if args.modality == "dummy":
        module = importlib.import_module("federated.dummy_module")
    else:
        module = importlib.import_module(f"modules.{args.modality}_module")

    # 2. Generate hospital data partitions
    partitions = module.get_hospital_partitions(num_hospitals=args.num_hospitals, alpha=args.alpha)
    
    # 3. Log experiment run in SQLite DB
    run_id = log_experiment_run(args.modality, "FedAvg", args.alpha, args.num_hospitals, args.rounds)
    print(f"Logged experiment run_id: {run_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # 4. Define ClientFn for Flower Simulation
    def client_fn(cid: str) -> fl.client.Client:
        hospital_idx = int(cid)
        train_loader, val_loader = partitions[hospital_idx]
        numpy_client = FlowerNumPyClient(module, train_loader, val_loader, device=device)
        return numpy_client.to_client()

    # 5. Define custom evaluate metric aggregation function
    def evaluate_metrics_aggregation(metrics_list):
        if not metrics_list:
            return {}
        total_examples = sum(num_examples for num_examples, _ in metrics_list)
        aggregated_metrics = {}
        first_metrics = metrics_list[0][1]
        for key in first_metrics.keys():
            weighted_val = sum(num_examples * m[key] for num_examples, m in metrics_list if key in m)
            aggregated_metrics[key] = weighted_val / max(1, total_examples)
        return aggregated_metrics

    # 6. Instantiate FedAvg Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_hospitals,
        min_evaluate_clients=args.num_hospitals,
        min_available_clients=args.num_hospitals,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    # 7. Start Flower Virtual Client Engine Simulation
    client_resources = {"num_gpus": 0.25 if torch.cuda.is_available() else 0.0, "num_cpus": 1}
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_hospitals,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    print("FedAvg Simulation completed cleanly!")
    print("Metrics History:", history)

if __name__ == "__main__":
    main()
