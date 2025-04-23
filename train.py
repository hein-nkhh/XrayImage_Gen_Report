import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from dataset import XrayReportDataset
from model import XrayReportModel
from config import Config
from evaluate import evaluate_model

torch.manual_seed(Config.seed)

def custom_collate(batch):
    front_imgs = [item['front'] for item in batch]
    lateral_imgs = [item['lateral'] for item in batch]
    reports = [item['report'] for item in batch]
    return {
        'front': front_imgs,
        'lateral': lateral_imgs,
        'report': reports
    }
    
# Dataset & DataLoader
train_dataset = XrayReportDataset(Config.train_csv, Config.image_dir,
                                   transform_front=XrayReportDataset.get_transform_front(),
                                   transform_lateral=XrayReportDataset.get_transform_lateral())

val_dataset = XrayReportDataset(Config.cv_csv, Config.image_dir,
                                 transform_front=XrayReportDataset.get_transform_front(),
                                 transform_lateral=XrayReportDataset.get_transform_lateral())

test_dataset = XrayReportDataset(Config.test_csv, Config.image_dir,
                                 transform_front=XrayReportDataset.get_transform_front(),
                                 transform_lateral=XrayReportDataset.get_transform_lateral())

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size,
                          shuffle=True, collate_fn=custom_collate)

val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,
                        shuffle=False, collate_fn=custom_collate)

test_loader = DataLoader(test_dataset, batch_size=Config.batch_size,
                        shuffle=False, collate_fn=custom_collate)

# Tạo thư mục chứa model nếu chưa tồn tại
if not os.path.exists(Config.output_dir):
    os.makedirs(Config.output_dir)
    
if not os.path.exists(Config.best_model_path):
    os.makedirs(Config.best_model_path)

# Model
model = XrayReportModel(Config).to(Config.device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=Config.lr)
num_training_steps = len(train_loader) * Config.epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=Config.warmup_steps,
                             num_training_steps=num_training_steps)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score - self.delta:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training loop with Early Stopping
early_stopping = EarlyStopping(patience=Config.patience, delta=0.001)

best_bleu1 = 0
for epoch in range(Config.epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        front = batch['front']
        lateral = batch['lateral']
        report = batch['report']  # raw text list

        outputs = model(front, lateral, report)
        loss = outputs['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} Loss: {avg_loss:.4f}")

    # Evaluate on validation set
    if (epoch + 1) % Config.eval_every_n_epochs == 0:
        metrics = evaluate_model(model=model, dataloader=val_loader, num_examples=5, test=False)
        print(f"Validation metrics: {metrics}")

        # Save model with best BLEU-1 score
        if metrics['bleu1'] > best_bleu1:
            best_bleu1 = metrics['bleu1']
            torch.save(model.state_dict(), Config.best_model_path)
            print(f"✅ Saved best model (BLEU-1 = {best_bleu1:.4f}) at {Config.best_model_path}")

        # Check early stopping
        early_stopping(metrics['bleu1'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            model.load_state_dict(early_stopping.best_model)
            break  # Stop training

# Load the best model for evaluation
model.load_state_dict(torch.load(Config.best_model_path))

# Evaluate on the test set
print("\nEvaluating on test set...")
test_metrics = evaluate_model(model=model, dataloader=test_loader, num_examples=5, test=True)
print(f"Test metrics: {test_metrics}")
