from torch.utils.data import Dataset
from PIL import Image
import os

class XrayImagePairDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1_path = os.path.join(self.image_dir, os.path.basename(row["Image1"]))
        img2_path = os.path.join(self.image_dir, os.path.basename(row["Image2"]))

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2
