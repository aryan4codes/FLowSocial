import flwr as fl
from typing import List, Tuple, Dict
import numpy as np
import torch
from collections import defaultdict
from cryptography.fernet import Fernet

class SecureServer(fl.server.Server):
    def __init__(self, 
                 num_clusters: int = 3,
                 min_fit_clients: int = 5,
                 min_available_clients: int = 5):
        self.num_clusters = num_clusters
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        
        self.client_keys = {}
        
        # Initialize cluster centroids
        self.cluster_centroids = None
    
    def aggregate_fit(self, results, failures):
        """Aggregate encrypted model updates from clients"""
        if not results:
            return None
            
        # Decrypt and aggregate parameters
        aggregated_params = None
        weights = []
        
        for client_result in results:
            client_id = client_result.id
            encrypted_params = client_result.parameters
            
            # Decrypt parameters (in practice, use secure key exchange)
            if client_id in self.client_keys:
                cipher_suite = Fernet(self.client_keys[client_id])
                decrypted_params = [
                    np.frombuffer(cipher_suite.decrypt(param), dtype=np.float32)
                    for param in encrypted_params
                ]
                
                if aggregated_params is None:
                    aggregated_params = decrypted_params
                else:
                    for i in range(len(aggregated_params)):
                        aggregated_params[i] += decrypted_params[i]
                        
                weights.append(client_result.num_examples)
        
        # Average the parameters
        total_weight = sum(weights)
        for i in range(len(aggregated_params)):
            aggregated_params[i] /= total_weight
            
        return aggregated_params
    
    def configure_fit(self, rnd: int, parameters, client_manager):
        """Configure the next round of training"""
        config = {
            "round": rnd,
            "num_clusters": self.num_clusters,
        }
        
        if parameters is not None:
            config["parameters"] = parameters
            
        # Sample clients for next round
        sample_size = int(client_manager.num_available() * 0.75)
        clients = client_manager.sample(
            num_clients=max(sample_size, self.min_fit_clients),
            min_num_clients=self.min_available_clients,
        )
        
        return clients, config
    
    def aggregate_evaluate(self, results, failures):
        """Aggregate evaluation results from clients"""
        if not results:
            return None
            
        # Aggregate metrics (e.g., clustering quality)
        aggregated_metrics = defaultdict(float)
        total_examples = 0
        
        for client_result in results:
            metrics = client_result.metrics
            num_examples = client_result.num_examples
            
            for metric, value in metrics.items():
                aggregated_metrics[metric] += value * num_examples
            total_examples += num_examples
            
        return {
            metric: value / total_examples
            for metric, value in aggregated_metrics.items()
        }