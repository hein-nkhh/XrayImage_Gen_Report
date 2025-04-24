from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from models.feature_extractor import FeatureExtractor

class ImageDataset(Dataset):
    def __init__(self, image_paths1, image_paths2):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.feature_extractor = FeatureExtractor()
    
    def __len__(self):
        return len(self.image_paths1)
    
    def __getitem__(self, idx):
        image1 = self.load_image(self.image_paths1[idx])
        image2 = self.load_image(self.image_paths2[idx])
        if image1 is not None and image2 is not None:
            f1 = self.feature_extractor.extract(image1)
            f2 = self.feature_extractor.extract(image2)
            return np.concatenate([f1, f2], axis=-1)
        return None
    
    def load_image(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((224, 224))
            return np.array(image)
        except FileNotFoundError:
            return None

def extract_features_for_datasets(image_paths1, image_paths2, batch_size=32):
    dataset = ImageDataset(image_paths1, image_paths2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    for batch in dataloader:
        batch_features = batch.numpy()
        features.append(batch_features)
    return np.concatenate(features, axis=0)
