from urllib.request import urlopen
from PIL import Image
import timm
import torch
import json

# Load the image
url = "testimage.jpeg"
image = Image.open(url)

# Load the pretrained model
model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True)
model = model.eval()

# Get model-specific transforms
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Process the image
input_tensor = transforms(image).unsqueeze(0)  # Unsqueeze single image into a batch of 1

# Predict with the model
output = model(input_tensor)
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

# Load ImageNet class labels
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_labels = json.loads(urlopen(imagenet_labels_url).read().decode("utf-8"))

# Map indices to labels and print results
for prob, idx in zip(top5_probabilities[0], top5_class_indices[0]):
    print(f"Class: {imagenet_labels[idx]}, Probability: {prob.item():.2f}%")
