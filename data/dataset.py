import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
from transformers import SwinModel, AutoImageProcessor
from config import DEVICE, IMAGE_DIR
from models.feature_extractor import SwinFeatureExtractor

def load_image(img_name, base_path=IMAGE_DIR):
    path = os.path.join(base_path, os.path.basename(img_name.strip()))
    try:
        image = Image.open(path).convert("RGB")
        image = image.resize((224, 224))
        return np.array(image)
    except FileNotFoundError:
        print(f"❌ Image not found: {path}")
        return None

def extract_img_feature(image1_path, image2_path):
    feature_extractor = SwinFeatureExtractor()
    
    def extract_single(image_path):
        image = load_image(image_path)
        if image is None: return None
        return feature_extractor.extract(image)
    
    f1 = extract_single(image1_path)
    f2 = extract_single(image2_path)
    if f1 is not None and f2 is not None:
        return np.concatenate([f1, f2], axis=-1)
    return None

def extract_features_for_datasets(train_df, cv_df, test_df):
    def extract_all(df, desc):
        features = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=desc):
            feat = extract_img_feature(row['Image1'], row['Image2'])
            if feat is not None:
                features.append(feat)
        return np.array(features)

    return (
        extract_all(train_df, "Train"),
        extract_all(cv_df, "Validation"),
        extract_all(test_df, "Test")
    )

def clean_text(text):
    import re
    text = str(text)
    text = re.sub(r'[^\w\s.]', '', text)
    return ' '.join(text.split())
