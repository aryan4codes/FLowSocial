import argparse
from fl_model.server import main as server_main
from fl_model.client import TextFLClient
import flwr as fl
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['server', 'client'], required=True, help="Start as server or client")
    parser.add_argument('--server_address', type=str, default="localhost:8080")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    if args.mode == "server":
        print("Starting federated learning server…")
        server_main()
    
    elif args.mode == "client":
        print("Starting federated learning client…")
        client = TextFLClient(device=args.device)
        fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()