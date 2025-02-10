# Utility functions (e.g. encryption, key management)

import torch
from models.text_recommender import TextRecommender

def evaluate_aggregate(server_round, parameters, config):
    """
    Custom evaluation function to evaluate aggregated model parameters.
    For demonstration, we just create a new model instance and compute 
    a dummy loss on random data.
    """
    model = TextRecommender(num_recommendations=5)
    # Load NumPy parameters into the model
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    # Generate dummy validation data
    input_ids = torch.randint(0, 10000, (1, 32))
    attention_mask = torch.ones((1, 32))
    target = torch.rand(1, 5)
    
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        loss = torch.nn.MSELoss()(output, target)
    print(f"Round {server_round} evaluation loss: {loss.item()}")
    return float(loss.item())