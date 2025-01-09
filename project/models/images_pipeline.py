# import pandas as pd
# import numpy as np
# import torch
# from PIL import Image
# import os
# from typing import List, Tuple, Dict
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

# class OpenImagesDataset(Dataset):
#     def __init__(self, root_dir: str, transform=None):
#         """
#         Initialize dataset
#         root_dir: Path containing data folder with all files
#         """
#         self.root_dir = root_dir
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                               std=[0.229, 0.224, 0.225])
#         ])
        
#         # Load metadata
#         self.classes_df = pd.read_csv(os.path.join(root_dir, 'metadata/classes.csv'),
#                                     names=['LabelName', 'DisplayName'])
#         self.classifications_df = pd.read_csv(os.path.join(root_dir, 'labels/classifications.csv'))
#         self.detections_df = pd.read_csv(os.path.join(root_dir, 'labels/detections.csv'))
        
#         # Create label mapping
#         self.label_map = dict(zip(self.classes_df['LabelName'], 
#                                  self.classes_df['DisplayName']))
        
#         # Get all image IDs
#         self.image_ids = [f.split('.')[0] for f in os.listdir(os.path.join(root_dir, 'raw'))
#                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        
#         # Create tag vocabulary
#         self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.classes_df['DisplayName'])}
#         self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        
#     def __len__(self):
#         return len(self.image_ids)
    
#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
        
#         # Load image
#         image_path = os.path.join(self.root_dir, 'raw', f"{image_id}.jpg")
#         image = Image.open(image_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
        
#         # Get tags
#         image_classifications = self.classifications_df[
#             self.classifications_df['ImageID'] == image_id
#         ]
        
#         # Convert labels to display names
#         tags = [self.label_map.get(row['LabelName'], row['LabelName'])
#                 for _, row in image_classifications.iterrows()]
        
#         # Create tag embedding (one-hot)
#         tag_embedding = torch.zeros(len(self.tag_to_idx))
#         for tag in tags:
#             if tag in self.tag_to_idx:
#                 tag_embedding[self.tag_to_idx[tag]] = 1
        
#         return {
#             'image_id': image_id,
#             'image': image,
#             'tags': tags,
#             'tag_embedding': tag_embedding
#         }

# class ImagePipeline:
#     def __init__(self, root_dir: str, batch_size: int = 32):
#         self.root_dir = root_dir
#         self.batch_size = batch_size
#         self.dataset = OpenImagesDataset(root_dir)
        
#     def get_dataloader(self, num_workers: int = 4):
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=num_workers
#         )
    
#     def split_data_for_clients(self, num_clients: int) -> List[List[Dict]]:
#         """Split dataset into chunks for each client"""
#         all_indices = list(range(len(self.dataset)))
#         np.random.shuffle(all_indices)
        
#         # Split indices for each client
#         client_indices = np.array_split(all_indices, num_clients)
        
#         client_data = []
#         for indices in client_indices:
#             client_batch = []
#             for idx in indices:
#                 client_batch.append(self.dataset[idx])
#             client_data.append(client_batch)
            
#         return client_data

# if __name__ == "__main__":
#     # Test pipeline
#     pipeline = ImagePipeline("path/to/data")
#     dataloader = pipeline.get_dataloader()
    
#     # Print sample batch
#     for batch in dataloader:
#         print(f"Batch size: {batch['image'].size()}")
#         print(f"Number of tags: {len(batch['tags'])}")
#         break


import pandas as pd
import numpy as np
import torch
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List

class ImageTagDataset(Dataset):
    def __init__(self, root_dir: str):
        """
        Initialize dataset focused on images and their tags
        root_dir: Path to the data folder containing raw/, labels/, metadata/
        """
        self.root_dir = root_dir
        
        # Load only necessary data
        print("Loading dataset files...")
        self.classes_df = pd.read_csv(
            os.path.join(root_dir, 'metadata/classes.csv'), 
            header=None, 
            names=['LabelName', 'DisplayName']
        )
        self.classifications_df = pd.read_csv(
            os.path.join(root_dir, 'labels/classifications.csv')
        )
        
        # Create label mapping
        self.label_map = dict(zip(
            self.classes_df['LabelName'], 
            self.classes_df['DisplayName']
        ))
        
        # Get available image IDs (only those that exist in raw folder)
        print("Scanning raw image folder...")
        available_images = set(f.split('.')[0] for f in os.listdir(os.path.join(root_dir, 'raw')))
        
        # Filter classifications to only include available images
        self.classifications_df = self.classifications_df[
            self.classifications_df['ImageID'].isin(available_images)
        ]
        
        # Get unique image IDs after filtering
        self.image_ids = self.classifications_df['ImageID'].unique()
        print(f"Found {len(self.image_ids)} images with classifications")
        
        # Create tag vocabulary
        unique_tags = set(
            self.label_map[label] 
            for label in self.classifications_df['LabelName'].unique() 
            if label in self.label_map
        )
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}
        print(f"Vocabulary size: {len(self.tag_to_idx)} tags")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load and preprocess image
        try:
            image_path = os.path.join(self.root_dir, 'raw', f"{image_id}.jpg")
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            # Return a zero tensor if image loading fails
            image_tensor = torch.zeros(3, 224, 224)
        
        # Ensure the tensor is correctly sized
        if image_tensor.size() != (3, 224, 224):
            print(f"Resizing mismatch for image {image_id}. Resizing...")
            image_tensor = transforms.Resize((224, 224))(image_tensor)
            image_tensor = transforms.ToTensor()(image_tensor)

        # Get tags for this image
        image_tags = self.get_image_tags(image_id)
        
        # Create tag embedding (multi-hot encoding)
        tag_embedding = torch.zeros(len(self.tag_to_idx))
        for tag in image_tags:
            if tag in self.tag_to_idx:
                tag_embedding[self.tag_to_idx[tag]] = 1.0
        
        return {
            'image_id': image_id,
            'image': image_tensor,
            'tags': image_tags,
            'tag_embedding': tag_embedding
        }

    def get_image_tags(self, image_id: str) -> List[str]:
        """Get all tags for a specific image"""
        image_classifications = self.classifications_df[
            self.classifications_df['ImageID'] == image_id
        ]
        
        tags = []
        for _, row in image_classifications.iterrows():
            if row['LabelName'] in self.label_map:
                tag = self.label_map[row['LabelName']]
                confidence = row['Confidence']
                # Only include tags with high confidence
                if confidence > 0.5:
                    tags.append(tag)
        
        return tags

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length tag lists
    """
    # Initialize empty lists for each key
    image_ids = []
    images = []
    tags_list = []
    tag_embeddings = []
    
    # Collect items separately
    for item in batch:
        image_ids.append(item['image_id'])
        images.append(item['image'])
        tags_list.append(item['tags'])
        tag_embeddings.append(item['tag_embedding'])
    
    # Stack tensors and create return dictionary
    return {
        'image_id': image_ids,  # Keep as list
        'image': torch.stack(images),  # Stack into batch
        'tags': tags_list,  # Keep as list of lists
        'tag_embedding': torch.stack(tag_embeddings)  # Stack into batch
    }

def create_dataloaders(dataset: ImageTagDataset, 
                      num_clients: int,
                      batch_size: int = 32) -> List[DataLoader]:
    """Split dataset into multiple dataloaders for federated learning"""
    
    # Calculate size of each client's dataset
    total_size = len(dataset)
    client_size = total_size // num_clients
    
    client_dataloaders = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else total_size
        
        # Create subset indices
        indices = list(range(start_idx, end_idx))
        
        # Create subset using torch's SubsetRandomSampler
        client_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(indices),
            num_workers=2,
            pin_memory=True,
            collate_fn=custom_collate_fn  # Add custom collate function
        )
        
        client_dataloaders.append(client_loader)
    
    return client_dataloaders

# Example usage
if __name__ == "__main__":
    # Test dataset
    dataset = ImageTagDataset("../data")
    print("\nTesting dataset access...")
    
    # Get one sample
    sample = dataset[0]
    print(f"\nSample image ID: {sample['image_id']}")
    print(f"Image tensor shape: {sample['image'].shape}")
    print(f"Number of tags: {len(sample['tags'])}")
    print(f"Tags: {sample['tags']}")
    print(f"Tag embedding shape: {sample['tag_embedding'].shape}")
    
    # Test dataloader creation
    print("\nCreating client dataloaders...")
    client_loaders = create_dataloaders(dataset, num_clients=5)
    print(f"Created {len(client_loaders)} client dataloaders")
    
    # Test first client's dataloader
    print("\nTesting first client's dataloader...")
    first_batch = next(iter(client_loaders[0]))
    print(f"Batch size: {len(first_batch['image_id'])}")
    print(f"First batch image shape: {first_batch['image'].shape}")