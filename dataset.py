import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class XrayReportDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform_front = None, transform_lateral = None, max_length=153):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform_front = transform_front
        self.transform_lateral = transform_lateral
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image1_name = os.path.basename(row['Image1'])
        image2_name = os.path.basename(row['Image2'])

        image1_path = os.path.join(self.image_dir, image1_name)
        image2_path = os.path.join(self.image_dir, image2_name)

        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        if self.transform_lateral and self.transform_front:
            # Apply transformations to images
            image1 = self.transform_front(image1)
            image2 = self.transform_lateral(image2)
        
        # Return report if available (for training/eval)
        if 'Report' in row:
            report = row['Report']
            return {
                'front': image1,
                'lateral': image2,
                'report': report  # Truncate to max_length
            }
        else:
            return {
                'front': image1,
                'lateral': image2,
                'report': None  # For inference, no report available
            }
            
    @staticmethod
    def get_transform_front():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])
    def get_transform_lateral():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.498, 0.462, 0.413], std=[0.234, 0.229, 0.221])
        ])