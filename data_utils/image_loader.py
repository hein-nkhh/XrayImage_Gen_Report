import os
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, SwinModel

class ImageFeatureExtractor:
    def __init__(self, model_name='microsoft/swin-base-patch4-window7-224', device='cpu'):
        self.device = device
        self.model = SwinModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    def load_image(self, img_path, size=(224, 224)):
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(size)
            return np.asarray(image)
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy ảnh: {img_path}")
            return None

    def extract_feature(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # CLS token

    def extract_from_paths(self, path1, path2):
        img1 = self.load_image(path1)
        img2 = self.load_image(path2)
        if img1 is not None and img2 is not None:
            feat1 = self.extract_feature(img1)
            feat2 = self.extract_feature(img2)
            return np.concatenate([feat1, feat2], axis=-1)
        return None