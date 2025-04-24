import pandas as pd
import torch

from config import DEVICE, CHECKPOINT_PATH, CHECKPOINT_DIR
from data.dataset import extract_features_for_datasets, clean_text
from data.dataloader import get_dataloaders
from models.mlp import MLP
from models.report_generator import ReportGenerator
from train.trainer import train_model
from train.evaluator import evaluate_model
import os


    
def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created directory: {CHECKPOINT_DIR}")
    else:
        print(f"Directory already exists: {CHECKPOINT_DIR}")
    
    # --- 1. Load dữ liệu CSV ---
    train_df = pd.read_csv("/kaggle/input/data-split-csv/train.csv")
    val_df = pd.read_csv("/kaggle/input/data-split-csv/val.csv")
    test_df = pd.read_csv("/kaggle/input/data-split-csv/test.csv")

    # --- 2. Làm sạch văn bản ---
    y_train = [clean_text(r) for r in train_df["Report"]]
    y_val = [clean_text(r) for r in val_df["Report"]]
    y_test = [clean_text(r) for r in test_df["Report"]]

    # --- 3. Trích xuất đặc trưng ảnh ---
    print("🔍 Extracting image features...")
    X_train, X_val, X_test = extract_features_for_datasets(train_df, val_df, test_df)

    # --- 4. Chuẩn bị DataLoader ---
    print("📦 Tokenizing and preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    # --- 5. Khởi tạo mô hình ---
    print("🧠 Initializing models...")
    generator = ReportGenerator()
    d_model = generator.model.config.d_model
    mlp = MLP(output_dim=d_model).to(DEVICE)

    # --- 6. Train mô hình ---
    print("🚀 Starting training loop...")
    train_model(mlp, generator, train_loader, val_loader)

    # --- 7. Load checkpoint tốt nhất (nếu có) ---
    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0:
        torch.cuda.empty_cache()

    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Loading best checkpoint from {CHECKPOINT_PATH}...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        mlp.load_state_dict(state_dict['mlp'])
        generator.model.load_state_dict(state_dict['biobart'])

    # --- 8. Evaluate ---
    print("📊 Evaluating on test set...")
    evaluate_model(mlp, generator, test_loader)

if __name__ == "__main__":
    main()
