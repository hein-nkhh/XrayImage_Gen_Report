import os
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import Config
from model import VisionEncoder, TextDecoder, CrossAttention, XrayReportModel
from dataset import XrayReportDataset
from utils import collate_fn
from metrics import calculate_metrics

def adjust_encoder_channels(vision_encoder, in_chans=6):
    """
    Adjust the encoder's first conv layer to accept 6 channels (2 X-ray images)
    """
    orig_proj = vision_encoder.encoder.patch_embed.proj
    new_proj = nn.Conv2d(
        in_channels=in_chans,
        out_channels=orig_proj.out_channels,
        kernel_size=orig_proj.kernel_size,
        stride=orig_proj.stride,
        padding=orig_proj.padding,
        bias=(orig_proj.bias is not None)
    )
    
    # Initialize new conv layer with weights from original
    with torch.no_grad():
        # Copy weights for first 3 channels
        new_proj.weight[:, :3, :, :].copy_(orig_proj.weight)
        # Copy weights again for the next 3 channels
        new_proj.weight[:, 3:, :, :].copy_(orig_proj.weight)
        if orig_proj.bias is not None:
            new_proj.bias.copy_(orig_proj.bias)
            
    vision_encoder.encoder.patch_embed.proj = new_proj
    return vision_encoder

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for images, reports in progress_bar:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images, reports)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate model on validation/test data"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for images, reports in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # Get loss
            outputs = model(images, reports)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Generate predictions
            preds = model.generate_report(images)
            
            all_preds.extend(preds)
            all_refs.extend(reports)
    
    print("\nSamples (Prediction | Reference):")
    for i in range(min(3, len(all_preds))):
        print(f"Pred: {all_preds[i][:100]}...")
        print(f"Ref:  {all_refs[i][:100]}...")
        print("---")
        
    # Calculate evaluation metrics
    metrics = calculate_metrics(all_preds, all_refs)
    metrics["loss"] = total_loss / len(dataloader)
    
    return metrics

def train_model(config=Config):
    """Main training function"""
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device)
    
    # Create datasets and dataloaders
    transform = XrayReportDataset.get_transform()
    train_ds = XrayReportDataset(config.train_csv, config.image_dir, transform=transform, max_length=config.max_len)
    val_ds = XrayReportDataset(config.cv_csv, config.image_dir, transform=transform, max_length=config.max_len)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model components
    vision_encoder = VisionEncoder(model_name=config.vision_encoder_name, output_dim=config.vision_output_dim)
    vision_encoder = adjust_encoder_channels(vision_encoder, in_chans=6)
    
    text_decoder = TextDecoder(model_name=config.text_decoder_model)
    cross_attention = CrossAttention(hidden_dim=config.cross_attn_dim, num_heads=config.cross_attn_heads)
    
    # Create full model
    model = XrayReportModel(vision_encoder, text_decoder, cross_attention).to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    best_bleu4 = -1.0
    patience = 3 
    patience_counter = 0 

    print(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log results
        print(f"Epoch {epoch}/{config.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  BLEU-1: {val_metrics['bleu1']:.4f}")
        print(f"  BLEU-4: {val_metrics['bleu4']:.4f}")
        print(f"  METEOR: {val_metrics['meteor']:.4f}")
        print(f"  ROUGE-L: {val_metrics['rouge_l']:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(config.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        
        
        # Save best model
        if val_metrics["bleu4"] > best_bleu4:
            best_bleu4 = val_metrics["bleu4"]
            torch.save(model.state_dict(), config.best_model_path)
            print(f"  New best model saved! BLEU-4: {best_bleu4:.4f}")
            patience_counter = 0  # Reset bộ đếm
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs. (Patience: {patience})")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs!")
            break
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Training complete!")
    return model