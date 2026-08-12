import argparse
import importlib
import torch
import flwr as fl
from federated.client_wrapper import FlowerNumPyClient

def main():
    parser = argparse.ArgumentParser(description="Start distributed Flower Client for FedMedDx.")
    parser.add_argument("--server", type=str, required=True, help="ngrok TCP address (e.g. 0.tcp.ngrok.io:12345)")
    parser.add_argument("--modality", type=str, required=True, choices=["cxr", "skin", "mri", "retina", "dummy"],
                        help="Modality module name to load.")
    parser.add_argument("--hospital_id", type=int, default=0, help="Local hospital partition index (simulated subset).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for split.")
    args = parser.parse_args()

    print(f"Connecting client to server at {args.server} using modality: {args.modality}...")

    # Load selected modality module
    if args.modality == "dummy":
        module = importlib.import_module("federated.dummy_module")
    else:
        module = importlib.import_module(f"modules.{args.modality}_module")

    # Load local data partition
    # (For testing distributed networks, we split dataset locally and select partition index)
    partitions = module.get_hospital_partitions(num_hospitals=3, alpha=args.alpha)
    train_loader, val_loader = partitions[args.hospital_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # Initialize NumPyClient wrapper
    client = FlowerNumPyClient(module, train_loader, val_loader, device=device)

    # Start NumPy client connection
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )

if __name__ == "__main__":
    main()
