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

val_dataset = XrayReportDataset(Config.test_csv, Config.image_dir,
                                 transform_front=XrayReportDataset.get_transform_front(),
                                 transform_lateral=XrayReportDataset.get_transform_lateral())

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size,
                          shuffle=True, collate_fn=custom_collate)

val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,
                        shuffle=False, collate_fn=custom_collate)

if not os.path.exists(Config.output_dir):
    os.makedirs(Config.output_dir)

# Model
model = XrayReportModel(Config).to(Config.device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=Config.lr)
num_training_steps = len(train_loader) * Config.epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=Config.warmup_steps,
                             num_training_steps=num_training_steps)

# Loss: handled inside model (CrossEntropy + ignore pad) → không cần tạo ngoài

# Training loop
best_bleu4 = 0
for epoch in range(Config.epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        front = batch['front']
        lateral = batch['lateral']
        report = batch['report']  # raw text list

        outputs = model(front, lateral, report)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} Loss: {avg_loss:.4f}")

    # 🔍 Evaluate on validation set
    if (epoch + 1) % Config.eval_every_n_epochs == 0:
        metrics = evaluate_model(model, val_loader, Config.device)
        print(f"Validation metrics: {metrics}")

        if metrics['bleu4'] > best_bleu4:
            best_bleu4 = metrics['bleu4']
            torch.save(model.state_dict(), Config.best_model_path)
            print(f"✅ Saved best model (BLEU-4 = {best_bleu4:.4f}) at {Config.best_model_path}")
