# FL server logic (aggregator)

import flwr as fl
from fl_model.utils import evaluate_aggregate

def main():
    # We use Flower's FedAvg with a custom evaluation function
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,   # change as needed
        min_available_clients=2,
        evaluate_fn=evaluate_aggregate  # custom evaluation of aggregated model
    )
    
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()