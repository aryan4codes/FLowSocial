# FL client with text-specific model (BERT-based)

import flwr as fl
import torch
import numpy as np
from models_v2.text_recommender import TextRecommender

class TextFLClient(fl.client.NumPyClient):
    def __init__(self, device="cpu"):
        self.device = device
        self.model = TextRecommender(num_recommendations=5).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Simulated local dataset: random tensors representing tokenized inputs
        self.local_data = self.generate_local_data()

    def generate_local_data(self):
        # For the prototype, generate random data mimicking tokenized text
        # In practice, load data from your database
        data = []
        for _ in range(10):
            input_ids = torch.randint(0, 10000, (1, 32)).to(self.device)
            attention_mask = torch.ones((1, 32)).to(self.device)
            target = torch.rand(1, 5).to(self.device)
            data.append((input_ids, attention_mask, target))
        return data

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epoch_loss = 0.0
        for input_ids, attention_mask, target in self.local_data:
            self.optimizer.zero_grad()
            output = self.model(input_ids, attention_mask)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return self.get_parameters(config), len(self.local_data), {"loss": epoch_loss / len(self.local_data)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, target in self.local_data:
                output = self.model(input_ids, attention_mask)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return float(total_loss / len(self.local_data)), len(self.local_data), {"loss": total_loss / len(self.local_data)}

if __name__ == "__main__":
    # For testing this file standalone, start the Flower client
    client = TextFLClient(device="cpu")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)