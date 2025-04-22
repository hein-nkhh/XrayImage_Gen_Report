# file config.py
import os
import torch

class Config:
    # Data paths
    train_csv = '/kaggle/input/data-split-csv/Train_Data.csv'
    cv_csv = '/kaggle/input/data-split-csv/CV_Data.csv'
    test_csv = '/kaggle/input/data-split-csv/Test_Data.csv'
    image_dir = '/kaggle/input/image-features-attention/xray_images'
    
    # Training hyperparameters
    batch_size = 8
    epochs = 15
    lr = 3e-5
    warmup_steps = 500
    max_len = 153
    dropout_rate = 0.1
    gradient_accumulation_steps = 1
    seed = 42

    # Model configuration
    vision_encoder_name = 'swin_base_patch4_window7_224'
    vision_output_dim = 1024
    cross_attn_dim = 1024
    cross_attn_heads = 8
    text_decoder_model = 'microsoft/biogpt'
    max_position_embeddings = 512
    hidden_size = 768

    # Generation parameters
    max_gen_length = 150
    num_beams = 4
    repetition_penalty = 1.2
    length_penalty = 1.0

    # Logging & Evaluation
    log_every_n_steps = 10
    eval_every_n_epochs = 1
    patience = 3

    # Model saving
    output_dir = '/kaggle/working/checkpoints'
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    save_every_n_epochs = 1

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
#file dataset.py
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class XrayReportDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform_front = None, transform_lateral = None, max_length=153):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform_front = transform_front
        self.transform_lateral = transform_lateral
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image1_name = os.path.basename(row['Image1'])
        image2_name = os.path.basename(row['Image2'])

        image1_path = os.path.join(self.image_dir, image1_name)
        image2_path = os.path.join(self.image_dir, image2_name)

        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        if self.transform_lateral and self.transform_front:
            # Apply transformations to images
            image1 = self.transform_front(image1)
            image2 = self.transform_lateral(image2)
        
        # Return report if available (for training/eval)
        if 'Report' in row:
            report = row['Report']
            return {
                'front': image1,
                'lateral': image2,
                'report': report  # Truncate to max_length
            }
        else:
            return {
                'front': image1,
                'lateral': image2,
                'report': None  # For inference, no report available
            }
            
    @staticmethod
    def get_transform_front():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])
    def get_transform_lateral():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.498, 0.462, 0.413], std=[0.234, 0.229, 0.221])
        ])
        

    
#file model.py
import torch
import torch.nn as nn
from timm import create_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)

class DualViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Base encoder cho từng view
        self.front_encoder = create_model(config.vision_encoder_name, 
                                        pretrained=True, 
                                        num_classes=0)
        self.lateral_encoder = create_model(config.vision_encoder_name, 
                                        pretrained=True, 
                                        num_classes=0)
        
        # Projection layers
        self.front_proj = ProjectionHead(self.front_encoder.num_features, 
                                        config.vision_output_dim)
        self.lateral_proj = ProjectionHead(self.lateral_encoder.num_features,
                                        config.vision_output_dim)
        
        # Cross-view attention
        self.cross_attn = EnhancedCrossAttention(hidden_dim=config.cross_attn_dim,
                                    num_heads=config.cross_attn_heads)
        
        self.output_dim = 2 * config.cross_attn_dim
    
    def forward(self, front, lateral):
        # Encode features
        front_feat = self.front_encoder.forward_features(front)
        lateral_feat = self.lateral_encoder.forward_features(lateral)
        
        # Flatten spatial dims
        front_feat = front_feat.view(front_feat.size(0), -1, front_feat.size(-1))   # [B, N, C]
        lateral_feat = lateral_feat.view(lateral_feat.size(0), -1, lateral_feat.size(-1))
        
        # Projection
        front_proj = self.front_proj(front_feat)
        lateral_proj = self.lateral_proj(lateral_feat)
        
        # Cross-view attention
        fused_feat = self.cross_attn(front_proj, lateral_proj)
        return fused_feat

class EnhancedCrossAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=8):
        super().__init__()
        self.front_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.lateral_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim*2, 4*hidden_dim*2),
            nn.GELU(),
            nn.Linear(4*hidden_dim*2, hidden_dim*2),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(hidden_dim*2)
        
    def forward(self, front, lateral):
        # Attention giữa các view
        attn_front, _ = self.front_attn(front, lateral, lateral)
        attn_lateral, _ = self.lateral_attn(lateral, front, front)
        
        # Kết hợp features
        combined = torch.cat([attn_front, attn_lateral], dim=-1)
        
        # Feed forward
        fused = self.norm(combined + self.ffn(combined))
        return fused

class TextDecoder(nn.Module):
    def __init__(self, model_name='microsoft/biogpt'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.embedding_dim = self.model.get_input_embeddings().weight.shape[1]
        
        # Add special tokens if needed
        special_tokens = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def encode_text(self, texts, max_length=153):
        """Encode text to input_ids and attention_mask"""
        if not isinstance(texts, list):
            texts = [texts]
            
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
        
    def decode(self, token_ids):
        """Decode token IDs to text"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        """Forward pass of text decoder"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels
        )

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class XrayReportModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder, cross_attention, config):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.cross_attention = cross_attention
        self.config = config

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_encoder.output_dim, text_decoder.model.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(text_decoder.model.config.hidden_size)
        )

        self.pos_encoder = PositionalEncoding(text_decoder.model.config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(text_decoder.model.config.hidden_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, front_images, lateral_images, text=None, labels=None):
        device = front_images.device

        vision_embeds = self.vision_encoder(front_images, lateral_images)
        vision_embeds = self.vision_proj(vision_embeds)
        vision_embeds = vision_embeds + self.pos_encoder(vision_embeds)
        vision_embeds = self.layer_norm(vision_embeds)
        vision_embeds = self.dropout(vision_embeds)

        if text is None:
            raise ValueError("Text input is required for training mode")

        input_ids, attention_mask = self.text_decoder.encode_text(text)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        text_embeds = text_embeds + self.pos_encoder(text_embeds)

        fused = torch.cat([vision_embeds, text_embeds], dim=1)
        fused = self.layer_norm(fused)

        vision_mask = torch.ones((attention_mask.size(0), vision_embeds.size(1)), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Fix: Pad labels with -100 to match fused sequence length
        if labels is None:
            labels = input_ids

        vision_pad = torch.full((labels.size(0), vision_embeds.size(1)), -100, dtype=labels.dtype, device=labels.device)
        padded_labels = torch.cat([vision_pad, labels], dim=1)

        return self.text_decoder(
            inputs_embeds=fused,
            attention_mask=full_attention_mask,
            labels=padded_labels
        )

        
#file trainer.py
import os
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import Config
from model import DualViewEncoder, TextDecoder, EnhancedCrossAttention, XrayReportModel
from dataset import XrayReportDataset
from utils import collate_fn

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch  in progress_bar:
        front = batch['front'].to(device)
        lateral = batch['lateral'].to(device)
        reports = batch['report']
        
        # Forward pass
        outputs = model(front_images=front, lateral_images=lateral, text=reports)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def train_model(config=Config):
    """Main training function"""
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device)
    
    # Create datasets and dataloaders
    transform_front = XrayReportDataset.get_transform_front()
    transform_lateral = XrayReportDataset.get_transform_lateral()
    train_ds = XrayReportDataset(config.train_csv, config.image_dir, transform_front=transform_front, transform_lateral = transform_lateral, max_length=config.max_len)
    val_ds = XrayReportDataset(config.cv_csv, config.image_dir, transform_front=transform_front, transform_lateral = transform_lateral, max_length=config.max_len)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model components
    vision_encoder = DualViewEncoder(config=config)
    
    text_decoder = TextDecoder(model_name=config.text_decoder_model)
    cross_attention = EnhancedCrossAttention(hidden_dim=config.cross_attn_dim, num_heads=config.cross_attn_heads)
    
    # Create full model
    model = XrayReportModel(vision_encoder, text_decoder, cross_attention, config).to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps, 
        num_training_steps=total_steps
    )

    print(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Save checkpoint
        ckpt_path = os.path.join(config.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Training complete!")
    return model

#main.py
import argparse
from config import Config
from XrayImage_Gen_report.train import train_model

def main():
    parser = argparse.ArgumentParser(description='X-ray Report Generation Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Operation mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (for testing)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train the model
        model = train_model()
        
if __name__ == '__main__':
    main()