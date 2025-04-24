import torch
from transformers import SwinModel, AutoImageProcessor
from config import DEVICE

class SwinFeatureExtractor:
    def __init__(self):
        self.model = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224').to(DEVICE)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')

    def extract(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        return outputs[:, 0, :].squeeze(0).cpu().numpy()  # [CLS] token
