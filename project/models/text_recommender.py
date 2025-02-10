import math
import torch
import torch.nn as nn
from transformers import DistilBertModel

class ClusteringAttention(nn.Module):
    def __init__(self, hidden_size: int, num_clusters: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_size = hidden_size
        self.multihead = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        # Initialize cluster centroids as learnable parameters.
        self.cluster_centroids = nn.Parameter(torch.randn(num_clusters, hidden_size))
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch, hidden_size)
        
        Returns:
            fused: Fused representation of shape (batch, hidden_size)
            cluster_weights: Attention weights for cluster assignments (batch, num_clusters)
        """
        # MultiheadAttention expects (seq_len, batch, hidden_size); we treat each representation as a sequence of 1.
        x_seq = x.unsqueeze(0)
        attn_output, _ = self.multihead(query=x_seq, key=x_seq, value=x_seq)
        attn_output = attn_output.squeeze(0)  # (batch, hidden_size)
        
        # Compute similarity scores with the cluster centroids.
        # Using scaled-dot product similarity manually.
        scores = torch.matmul(attn_output, self.cluster_centroids.t())  # (batch, num_clusters)
        scores = scores / math.sqrt(self.hidden_size)
        cluster_weights = torch.softmax(scores, dim=1)  # (batch, num_clusters)
        
        # Generate a cluster context for each instance.
        cluster_context = torch.matmul(cluster_weights, self.cluster_centroids)  # (batch, hidden_size)
        
        # Fuse the attention output with cluster context.
        fused = attn_output + cluster_context
        return fused, cluster_weights

class TextRecommender(nn.Module):
    def __init__(self, num_recommendations: int, num_clusters: int = 3, dropout: float = 0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size
        self.cluster_attention = ClusteringAttention(hidden_size=self.hidden_size, num_clusters=num_clusters, dropout=dropout)
        self.classifier = nn.Linear(self.hidden_size, num_recommendations)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Use CLS token representation
        pooled_output = self.dropout(pooled_output)
        
        # Apply clustering-based attention.
        fused, cluster_weights = self.cluster_attention(pooled_output)
        
        # Generate recommendations from the fused representation.
        recommendations = self.classifier(fused)
        return recommendations, cluster_weights