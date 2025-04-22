import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from dataset import XrayReportDataset
from model import XrayReportModel
from evaluate import evaluate_all

def train():
    torch.manual_seed(Config.seed)

    dataset = XrayReportDataset(
        csv_file=Config.train_csv,
        image_dir=Config.image_dir,
        transform_front=XrayReportDataset.get_transform_front(),
        transform_lateral=XrayReportDataset.get_transform_lateral()
    )
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    model = XrayReportModel(Config).to(Config.device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.lr)

    model.train()
    for epoch in range(Config.epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            front = batch['front'].to(Config.device)
            lateral = batch['lateral'].to(Config.device)
            report = batch['report']

            encoding = model.biogpt.encode_text(report, max_length=Config.max_len)
            labels = encoding['input_ids'].to(Config.device)

            outputs = model(front, lateral, report, labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        
        os.makedirs(Config.output_dir, exist_ok=True)
        torch.save(model.state_dict(), Config.best_model_path)

if __name__ == '__main__':
    train()
