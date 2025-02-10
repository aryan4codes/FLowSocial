import torch.nn as nn
from transformers import DistilBertModel

class TextRecommender(nn.Module):
    def __init__(self, num_recommendations: int, dropout: float = 0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_recommendations)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        recommendations = self.classifier(pooled_output)
        return recommendations