import torch
from transformers import SwinModel, AutoImageProcessor
from config import DEVICE

class FeatureExtractor:
    def __init__(self):
        self.model = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224').to(DEVICE)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224', use_fast=True)

    def extract(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        return outputs[:, 0, :].squeeze(0).cpu().numpy()  # [CLS] token
