import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
from transformers import SwinModel, AutoImageProcessor
from config import DEVICE, IMAGE_DIR

# Load Swin Transformer và extractor
swin_model = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224').to(DEVICE)
swin_model.eval()
feature_extractor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')

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
    def extract_single(image_path):
        image = load_image(image_path)
        if image is None: return None
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output = swin_model(**inputs).last_hidden_state[:, 0, :]
        return output.squeeze(0).cpu().numpy()
    
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
