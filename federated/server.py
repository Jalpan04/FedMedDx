import argparse
import flwr as fl

def main():
    parser = argparse.ArgumentParser(description="Start distributed Flower Server for FedMedDx.")
    parser.add_argument("--port", type=int, default=8080, help="Local port to bind server to.")
    parser.add_argument("--rounds", type=int, default=20, help="Total communication rounds.")
    parser.add_argument("--min_clients", type=int, default=4, help="Minimum clients required to start rounds.")
    args = parser.parse_args()

    print(f"Starting Flower Server on port {args.port} for {args.rounds} rounds...")

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

    # FedAvg Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    # Start Server
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
