import argparse
import os
from models.images_pipeline import ImageTagDataset, create_dataloaders
from models.secure_client import SecureClient
from models.secure_server import SecureServer
import flwr as fl
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--client_id', type=int, default=0)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--server_address', type=str, default='[::]:8080')
    args = parser.parse_args()
    
    # Initialize data pipeline
    dataset = ImageTagDataset(args.data_dir)
    
    if args.mode == 'server':
        # Start server
        server = SecureServer(
            num_clusters=3,
            min_fit_clients=args.num_clients,
            min_available_clients=args.num_clients
        )
        
        # Define strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )
        
        # Start Flower server
        fl.server.start_server(
            server_address=args.server_address,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds)
        )
        
    elif args.mode == 'client':
        # Create dataloaders for federated learning
        client_loaders = create_dataloaders(dataset, args.num_clients)
        client_data = client_loaders[args.client_id]
        
        # Initialize secure client
        client = SecureClient(
            client_id=args.client_id,
            dataset=dataset,
            num_tags=len(dataset.tag_to_idx),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Start Flower client
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )

if __name__ == "__main__":
    main()