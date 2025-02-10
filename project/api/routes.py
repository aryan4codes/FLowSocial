from flask import Flask, request, jsonify
from models.text_recommender import TextRecommender
from models.clustering import cluster_embeddings
import torch

app = Flask(__name__)

# Initialize the recommender model (for inference only)
model = TextRecommender(num_recommendations=5)

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Expects JSON {"input_ids": [...], "attention_mask": [...]}.
    Returns text recommendation scores and clusters.
    """
    data = request.get_json()
    input_ids = torch.tensor(data.get("input_ids")).long()
    attention_mask = torch.tensor(data.get("attention_mask")).long()
    
    # Get recommendations from model
    recommendations = model(input_ids, attention_mask)
    # For demonstration, simulate clustering by splitting recommendations into 2 clusters
    clusters = cluster_embeddings(recommendations.detach().numpy(), num_clusters=2)
    
    return jsonify({
        "recommendations": recommendations.detach().tolist(),
        "clusters": clusters.tolist()
    })