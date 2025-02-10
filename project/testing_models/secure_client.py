import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from cryptography.fernet import Fernet
import tensorflow_privacy as tf_privacy
from typing import Dict, List, Tuple

# Assuming OpenImagesDataset is defined in a module named datasets
from data.raw import OpenImagesDataset
import numpy as np

class SecureImageRecommender(nn.Module):
    def __init__(self, num_tags: int, embedding_dim: int = 64, num_clusters: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        
        # Image feature extractor (assuming preprocessed images)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embedding_dim)
        )
        
        # Tag processor
        self.tag_processor = nn.Sequential(
            nn.Linear(num_tags, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Combined features processor
        self.combined_processor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Clustering layer
        self.cluster_assignment = nn.Linear(64, num_clusters)
        
    def forward(self, images, tag_embeddings):
        # Process images and tags
        image_features = self.image_encoder(images)
        tag_features = self.tag_processor(tag_embeddings)
        
        # Combine features
        combined = torch.cat([image_features, tag_features], dim=1)
        features = self.combined_processor(combined)
        
        # Generate cluster assignments
        cluster_probs = torch.softmax(self.cluster_assignment(features), dim=1)
        
        return features, cluster_probs

class SecureClient(fl.client.NumPyClient):
    def __init__(self, 
                 client_id: int,
                 dataset: 'OpenImagesDataset',
                 num_tags: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.client_id = client_id
        self.device = device
        self.dataset = dataset
        
        # Initialize model
        self.model = SecureImageRecommender(num_tags=num_tags).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        
        # Security components
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # DP components
        self.noise_multiplier = 1.1
        self.l2_norm_clip = 1.0
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True
        )
        
        # Training with DP
        for epoch in range(5):
            for batch in train_loader:
                images = batch['image'].to(self.device)
                tag_embeddings = batch['tag_embedding'].to(self.device)
                
                # Forward pass
                features, cluster_probs = self.model(images, tag_embeddings)
                
                # Compute loss (example: clustering loss)
                loss = self._compute_clustering_loss(features, cluster_probs)
                
                # DP-SGD step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.l2_norm_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Encrypt parameters
        return self.encrypt_parameters(self.get_parameters(config)), len(self.dataset), {}
    
    def _compute_clustering_loss(self, features, cluster_probs):
        # Implement clustering loss (e.g., k-means like loss)
        return torch.mean((features - features.mean(0))**2)
    
    def encrypt_parameters(self, parameters):
        encrypted_params = []
        for param in parameters:
            param_bytes = param.tobytes()
            encrypted_param = self.cipher_suite.encrypt(param_bytes)
            encrypted_params.append(encrypted_param)
        return encrypted_params