# # Clustering functions using scikit-learn/FAISS

import numpy as np
from sklearn.cluster import KMeans

def cluster_embeddings(embeddings: np.ndarray, num_clusters: int = 3) -> np.ndarray:
    """
    Perform KMeans clustering on the provided embeddings.
    
    Args:
        embeddings: A NumPy array of shape (batch_size, embedding_dim).
        num_clusters: Number of clusters to form.
    
    Returns:
        cluster_labels: A NumPy array with cluster assignments.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels