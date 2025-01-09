import fiftyone as fo

# Load 5,000 images with animal-related tags
dataset = fo.zoo.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections", "classifications"],
    classes=["Cat", "Dog", "Bird", "Horse"],
    max_samples=5000,
)
dataset.export(
    export_dir="data/raw/",
    dataset_type=fo.types.ImageDirectory,
)
