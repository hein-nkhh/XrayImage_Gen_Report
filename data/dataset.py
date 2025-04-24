import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from config import DEVICE
from data.image_dataset import XrayImagePairDataset
from models.feature_extractor import SwinFeatureExtractor

def clean_text(text):
    import re
    text = str(text)
    text = re.sub(r'[^\w\s.]', '', text)
    return ' '.join(text.split())

def extract_features_batchwise(df, image_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = XrayImagePairDataset(df, image_dir=image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    extractor = SwinFeatureExtractor()
    features = []

    for img1_batch, img2_batch in tqdm(loader, desc="🔍 Extracting image features (batchwise)"):
        batch_feats = []
        for img1, img2 in zip(img1_batch, img2_batch):
            img1_np = (img1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img2_np = (img2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            f1 = extractor.extract(img1_np)
            f2 = extractor.extract(img2_np)
            combined = np.concatenate([f1, f2], axis=-1)
            batch_feats.append(combined)

        batch_array = np.array(batch_feats)  # ✅ fix performance warning
        features.append(torch.from_numpy(batch_array))

    return torch.cat(features, dim=0).numpy()

def extract_features_for_datasets(train_df, val_df, test_df, image_dir):
    X_train = extract_features_batchwise(train_df, image_dir)
    X_val = extract_features_batchwise(val_df, image_dir)
    X_test = extract_features_batchwise(test_df, image_dir)
    return X_train, X_val, X_test
