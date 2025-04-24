import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
IMAGE_DIR = "/kaggle/input/image-features-attention/xray_images"
CHECKPOINT_DIR = "/kaggle/working/checkpoint"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/best_model.pt"


# Model config

MLP_HIDDEN_DIM = 1024
MAX_SEQ_LEN = 150
BIOBART_MODEL_NAME = "GanjinZero/biobart-base"

# Training config
BATCH_SIZE = 16
EPOCHS = 2
LR_MLP = 1e-4
LR_BART = 5e-5
PATIENCE = 8
