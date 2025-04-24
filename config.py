import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
IMAGE_DIR = "/kaggle/working/xray_images"
CHECKPOINT_PATH = "vit_biobart_best_model.pt"

# Model config
MLP_INPUT_DIM = 2048
MLP_HIDDEN_DIM = 1024
MAX_SEQ_LEN = 150
BIOBART_MODEL_NAME = "GanjinZero/biobart-base"

# Training config
BATCH_SIZE = 16
EPOCHS = 20
LR_MLP = 1e-4
LR_BART = 5e-5
PATIENCE = 8
