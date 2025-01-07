from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests

url = "testimage.jpeg"
image = Image.open(url)

feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-xx-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small")

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
