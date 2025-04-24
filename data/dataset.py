import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
from transformers import SwinModel, AutoImageProcessor
from config import DEVICE, IMAGE_DIR
from models.feature_extractor import FeatureExtractor
import torch
from torch.utils.data import Dataset, DataLoader

# Hàm load ảnh từ đường dẫn
def load_image(img_name, base_path=IMAGE_DIR):
    path = os.path.join(base_path, os.path.basename(img_name.strip()))
    try:
        image = Image.open(path).convert("RGB")
        image = image.resize((224, 224))
        return np.array(image)
    except FileNotFoundError:
        print(f"❌ Image not found: {path}")
        return None

# Dataset cho việc trích xuất đặc trưng từ ảnh
class ImageFeatureDataset(Dataset):
    def __init__(self, image_paths1, image_paths2, feature_extractor):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.image_paths1)
    
    def __getitem__(self, idx):
        image1 = load_image(self.image_paths1[idx])
        image2 = load_image(self.image_paths2[idx])
        
        if image1 is not None and image2 is not None:
            f1 = self.feature_extractor.extract(image1)
            f2 = self.feature_extractor.extract(image2)
            return np.concatenate([f1, f2], axis=-1)
        return None

# Hàm trích xuất đặc trưng với Dataloader
def extract_features_with_dataloader(image_paths1, image_paths2, batch_size=32):
    feature_extractor = FeatureExtractor()
    dataset = ImageFeatureDataset(image_paths1, image_paths2, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        # Xử lý batch, đảm bảo dữ liệu trên CPU
        batch_features = batch
        features.append(batch_features)
    
    return np.concatenate(features, axis=0)

# Hàm trích xuất đặc trưng cho các datasets (train, cv, test)
def extract_features_for_datasets(train_df, cv_df, test_df):
    # Khởi tạo FeatureExtractor
    feature_extractor = FeatureExtractor()
    
    # Đảm bảo bạn có thể gọi extract_features_with_dataloader trên các DataFrame
    def extract_all(df, desc):
        image_paths1 = df['Image1'].tolist()
        image_paths2 = df['Image2'].tolist()
        return extract_features_with_dataloader(image_paths1, image_paths2, batch_size=32)
    
    return (
        extract_all(train_df, "Train"),
        extract_all(cv_df, "Validation"),
        extract_all(test_df, "Test")
    )
