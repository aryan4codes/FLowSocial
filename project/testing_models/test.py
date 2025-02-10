import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import json
from cryptography.fernet import Fernet
import tensorflow_privacy as tf_privacy

class SecureImageRecommender(nn.Module):
    def __init__(self, embedding_dim=64, num_clusters=3):
        """
        Neural network for secure image recommendations
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        
        # Embedding layers
        self.tag_embedding = nn.Embedding(1000, embedding_dim)  # Assuming 1000 possible tags
        
        # Recommendation layers
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        
        # Clustering layer
        self.cluster_assignment = nn.Linear(64, num_clusters)

    def forward(self, x):
        # Encode input
        encoded = self.encoder(x)
        # Generate cluster probabilities
        cluster_probs = torch.softmax(self.cluster_assignment(encoded), dim=1)
        # Decode for reconstruction
        decoded = self.decoder(encoded)
        return encoded, cluster_probs, decoded

class SecureClient(fl.client.NumPyClient):
    def __init__(self, device_id: int, embedding_dim: int = 64):
        self.device_id = device_id
        self.model = SecureImageRecommender(embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.local_data = []
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # DP noise mechanism
        self.noise_multiplier = 1.1
        self.l2_norm_clip = 1.0
        self.dp_optimizer = tf_privacy.DPAdamGaussianOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=1,
            learning_rate=0.001
        )

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data with differential privacy"""
        self.set_parameters(parameters)
        
        # Convert local data to tensors
        train_data = torch.tensor([item[0] for item in self.local_data])
        train_labels = torch.tensor([item[1] for item in self.local_data])
        
        # Apply DP training
        for epoch in range(5):
            self.model.train()
            
            # Add DP noise to gradients
            with tf_privacy.PrivacyAccountant():
                encoded, cluster_probs, decoded = self.model(train_data)
                
                # Compute losses
                reconstruction_loss = nn.MSELoss()(decoded, train_data)
                rating_loss = nn.BCEWithLogitsLoss()(cluster_probs, train_labels)
                total_loss = reconstruction_loss + rating_loss
                
                # DP-SGD step
                self.dp_optimizer.zero_grad()
                total_loss.backward()
                self.dp_optimizer.step()
        
        # Encrypt parameters before sending
        encrypted_params = self.encrypt_parameters(self.get_parameters(config))
        
        return encrypted_params, len(self.local_data), {}

    def encrypt_parameters(self, parameters):
        """Encrypt model parameters"""
        encrypted_params = []
        for param in parameters:
            param_bytes = param.tobytes()
            encrypted_param = self.cipher_suite.encrypt(param_bytes)
            encrypted_params.append(encrypted_param)
        return encrypted_params

    def decrypt_parameters(self, encrypted_parameters):
        """Decrypt model parameters"""
        decrypted_params = []
        for encrypted_param in encrypted_parameters:
            decrypted_bytes = self.cipher_suite.decrypt(encrypted_param)
            param = np.frombuffer(decrypted_bytes, dtype=np.float32)
            decrypted_params.append(param)
        return decrypted_params

class SecureServer(fl.server.Server):
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.global_model = SecureImageRecommender(num_clusters=num_clusters)
        self.cluster_profiles = {}

    def aggregate_fit(self, results, failures):
        """Aggregate encrypted model updates from clients"""
        # Decrypt and aggregate parameters
        aggregated_params = None
        num_clients = len(results)
        
        for client_result in results:
            encrypted_params = client_result.parameters
            # Each client would need to share their decryption key securely
            # In practice, this would use a secure key exchange protocol
            decrypted_params = self.decrypt_parameters(encrypted_params)
            
            if aggregated_params is None:
                aggregated_params = decrypted_params
            else:
                for i in range(len(aggregated_params)):
                    aggregated_params[i] += decrypted_params[i]
        
        # Average the parameters
        for i in range(len(aggregated_params)):
            aggregated_params[i] /= num_clients
            
        return aggregated_params

def main():
    # Initialize server and clients
    server = SecureServer(num_clusters=3)
    clients = [SecureClient(i) for i in range(5)]
    
    # Sample data (in practice, this would be your Open Images dataset)
    sample_images = [
        (torch.randn(64), torch.ones(3)),  # Positive example
        (torch.randn(64), torch.zeros(3))  # Negative example
    ]
    
    # Distribute data to clients
    for client in clients:
        client.local_data = sample_images
    
    # Define Flower strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )
    
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3},
        strategy=strategy
    )

if __name__ == "__main__":
    main()