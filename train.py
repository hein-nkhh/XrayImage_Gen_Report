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

# Dataset & DataLoader
train_dataset = XrayReportDataset(Config.train_csv, Config.image_dir,
                                   transform_front=XrayReportDataset.get_transform_front(),
                                   transform_lateral=XrayReportDataset.get_transform_lateral())

val_dataset = XrayReportDataset(Config.cv_csv, Config.image_dir,
                                 transform_front=XrayReportDataset.get_transform_front(),
                                 transform_lateral=XrayReportDataset.get_transform_lateral())

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

# Model
model = XrayReportModel(Config).to(Config.device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=Config.lr)
num_training_steps = len(train_loader) * Config.epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=Config.warmup_steps,
                             num_training_steps=num_training_steps)

# Loss
criterion = nn.CrossEntropyLoss(ignore_index=model.biogpt.tokenizer.pad_token_id)

# Training loop
best_bleu4 = 0
for epoch in range(Config.epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        front = batch['front'].to(Config.device)
        lateral = batch['lateral'].to(Config.device)
        report = batch['report']

        encoding = model.biogpt.encode_text(report, max_length=Config.max_len)
        input_ids = encoding['input_ids'].to(Config.device)
        labels = input_ids.clone()

        outputs = model(front, lateral, report, labels=labels)
        loss = outputs.loss

        loss.backward()
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
