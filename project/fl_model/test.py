import torch
from models.text_recommender import TextRecommender
from models.clustering import cluster_embeddings

def main():
    # Instantiate the text recommender model
    model = TextRecommender(num_recommendations=5)
    # Create dummy input tensors
    input_ids = torch.randint(0, 10000, (1, 32))
    attention_mask = torch.ones((1, 32))
    output = model(input_ids, attention_mask)
    print("Model output:", output.tolist())

    # Test clustering logic on the model output
    import numpy as np
    clusters = cluster_embeddings(output.detach().numpy(), num_clusters=2)
    print("Cluster assignments:", clusters)

if __name__ == "__main__":
    main()