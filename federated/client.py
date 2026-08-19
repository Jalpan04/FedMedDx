"""
FedMedDx Standalone Distributed Client.
Connects a local disease module to the central Flower coordinator via LAN gRPC.
"""

import argparse
import importlib
import torch
import flwr as fl
from federated.client_wrapper import FedRepClient

def main():
    parser = argparse.ArgumentParser(description="FedMedDx Distributed Node Client")
    parser.add_argument("--server", type=str, required=True, help="Server address (e.g. 192.168.1.50:8080 or 10.246.11.202:8080)")
    parser.add_argument("--modality", type=str, required=True, choices=["covid", "pneumonia", "tb", "pediatric", "dummy"], help="Modality task to execute")
    parser.add_argument("--hospital_id", type=int, default=0, help="Local hospital partition index (default: 0)")
    parser.add_argument("--client_id", type=str, default=None, help="Unique client ID for checkpoint persistence (defaults to modality_hospital_id)")
    args = parser.parse_args()

    client_id = args.client_id or f"{args.modality}_{args.hospital_id}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {args.modality} client node on device: {device} (Client ID: {client_id})")

    # Dynamically load the selected modality module
    if args.modality == "dummy":
        module = importlib.import_module("federated.dummy_module")
    else:
        module = importlib.import_module(f"modules.{args.modality}_module")

    model = module.get_model()
    partitions = module.get_hospital_partitions(num_hospitals=max(1, args.hospital_id + 1), alpha=0.5)
    train_loader, val_loader = partitions[min(args.hospital_id, len(partitions) - 1)]

    client = FedRepClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        module_contract=module,
        client_id=client_id,
        device=device,
    )

    print(f"Connecting to FedMedDx Coordinator at {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )

if __name__ == "__main__":
    main()
