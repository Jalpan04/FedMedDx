"""
FedMedDx Standalone Flower Server (FedRep Backbone Aggregation).
Runs on the coordinator's machine (Jalpan) to orchestrate federated rounds over LAN.
"""

import argparse
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics reported by distributed clients."""
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_acc = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics)
    avg_acc = weighted_acc / max(1, total_examples)
    
    print(f"\n[Global Server Round Summary] Aggregated Hospital Test Accuracy: {avg_acc * 100:.2f}%")
    return {"hospital_avg_accuracy": avg_acc}

def main():
    parser = argparse.ArgumentParser(description="FedMedDx Central Flower Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated communication rounds (default: 20)")
    parser.add_argument("--min_clients", type=int, default=4, help="Minimum clients required to start aggregation (default: 4)")
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=lambda server_round: {"epochs": 1, "round": server_round},
    )

    server_address = f"0.0.0.0:{args.port}"
    print(f"Starting FedMedDx FedRep Server on {server_address} for {args.rounds} rounds (waiting for {args.min_clients} clients)...")
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
