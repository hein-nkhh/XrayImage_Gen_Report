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
    
    # Model saving
    output_dir = '/kaggle/working/checkpoints'
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    
    # Model configuration
    vision_encoder_name = 'swin_base_patch4_window7_224'
    vision_output_dim = 1024
    cross_attn_dim = 1024
    cross_attn_heads = 8
    text_decoder_model = 'microsoft/biogpt'
    
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
        
#file inference.py
import torch
from torch.utils.data import DataLoader
from config import Config
from model import DualViewEncoder, TextDecoder, EnhancedCrossAttention, XrayReportModel
from dataset import XrayReportDataset
from utils import collate_fn
from PIL import Image
from metrics import calculate_metrics
from tqdm import tqdm



def load_model(checkpoint_path, config=Config):
    """Load trained model from checkpoint with updated architecture"""
    device = torch.device(config.device)
    
    # Khởi tạo các thành phần với cấu trúc mới
    vision_encoder = DualViewEncoder(config)
    text_decoder = TextDecoder(model_name=config.text_decoder_model)
    cross_attention = EnhancedCrossAttention(
        hidden_dim=config.cross_attn_dim,
        num_heads=config.cross_attn_heads
    )
    
    # Tạo model với config
    model = XrayReportModel(
        vision_encoder=vision_encoder,
        text_decoder=text_decoder,
        cross_attention=cross_attention,
        config=config
    )
    
    # Load weights và config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def run_inference(model, image_pairs, config=Config):
    """
    Run inference on new image pairs (front, lateral)
    
    Args:
        model: Loaded XrayReportModel
        image_pairs: List of tuples [(front_path, lateral_path), ...]
        config: Configuration
        
    Returns:
        List of generated reports
    """
    device = model.device
    transform = XrayReportDataset.get_transform()
    
    front_images = []
    lateral_images = []
    
    for front_path, lateral_path in image_pairs:
        # Xử lý từng view riêng
        front_img = transform(Image.open(front_path).convert('RGB'))
        lateral_img = transform(Image.open(lateral_path).convert('RGB'))
        
        front_images.append(front_img)
        lateral_images.append(lateral_img)
    
    # Tạo batch riêng cho từng view
    front_batch = torch.stack(front_images, dim=0).to(device)
    lateral_batch = torch.stack(lateral_images, dim=0).to(device)
    
    # Generation
    with torch.no_grad():
        reports = model.generate_report(
            front_images=front_batch,
            lateral_images=lateral_batch,
            **config.generation_params  # Sử dụng tham số từ config
        )
    
    return reports

def evaluate_test_set(model, config=Config):
    """Evaluate model on test set với xử lý multi-view"""
    device = model.device
    
    # Tạo dataset và dataloader mới
    class TestDataset(XrayReportDataset):
        def __getitem__(self, idx):
            data = super().__getitem__(idx)
            return {
                'front': data['front'],
                'lateral': data['lateral'],
                'report': data['report']
            }
    
    test_ds = TestDataset(config.test_csv, config.image_dir, 
                         transform=XrayReportDataset.get_transform())
    
    test_loader = DataLoader(test_ds, 
                            batch_size=config.batch_size,
                            collate_fn = collate_fn)
    
    all_preds = []
    all_refs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            front = batch['front'].to(device)
            lateral = batch['lateral'].to(device)
            
            preds = model.generate_report(front, lateral)
            all_preds.extend(preds)
            all_refs.extend(batch['report'])
    
    # Tính metrics
    metrics = calculate_metrics(all_preds, all_refs)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics, all_preds, all_refs

#file metrics.py
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('wordnet')
nltk.download('punkt')

def calculate_metrics(predictions, references):
    """
    Calculate NLG evaluation metrics: BLEU-1,2,3,4, METEOR, and ROUGE-L
    
    Args:
        predictions: List of generated report strings
        references: List of ground truth report strings
    
    Returns:
        Dictionary of metrics
    """
    # Tokenize predictions and references
    smooth = SmoothingFunction().method1
    pred_tokens = [word_tokenize(p.lower()) for p in predictions]
    ref_tokens = [[word_tokenize(r.lower())] for r in references]
    
    # BLEU Metrics
    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # METEOR
    m_scores = [meteor_score([word_tokenize(r)], word_tokenize(p)) for r, p in zip(references, predictions)]
    meteor = np.mean(m_scores)
    
    # ROUGE-L
    def lcs(x, y):
        m, n = len(x), len(y)
        L = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                L[i][j] = L[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(L[i - 1][j], L[i][j - 1])
        return L[m][n]
    
    rouge_scores = []
    for r, p in zip(references, predictions):
        rt, pt = word_tokenize(r), word_tokenize(p)
        l = lcs(rt, pt)
        rec = l / len(rt) if rt else 0
        prec = l / len(pt) if pt else 0
        rouge_scores.append((2 * rec * prec / (rec + prec)) if rec + prec else 0)
    
    rouge_l = np.mean(rouge_scores)
    
    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge_l": rouge_l
    }
    
#file model.py
import torch
import torch.nn as nn
from timm import create_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.cross_attn = CrossAttention(hidden_dim=config.cross_attn_dim,
                                    num_heads=config.cross_attn_heads)
    
    def forward(self, front, lateral):
        # Encode features
        front_feat = self.front_encoder.forward_features(front)
        lateral_feat = self.lateral_encoder.forward_features(lateral)
        
        # Projection
        front_proj = self.front_proj(front_feat)
        lateral_proj = self.lateral_proj(lateral_feat)
        
        # Cross-view attention
        fused_feat = self.cross_attn(front_proj, lateral_proj)
        return fused_feat

class EnhancedCrossAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=8):
        super().__init__()
        self.front_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.lateral_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, front, lateral):
        # Attention giữa các view
        attn_front, _ = self.front_attn(front, lateral, lateral)
        attn_lateral, _ = self.lateral_attn(lateral, front, front)
        
        # Kết hợp features
        combined = torch.cat([attn_front, attn_lateral], dim=1)
        
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

    def generate(self, input_ids=None, attention_mask=None, inputs_embeds=None,
             max_length=150, max_new_tokens=None, **kwargs):
        """Generate text from inputs"""
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either inputs_embeds or input_ids must be provided")

        with torch.no_grad():
            # When using inputs_embeds, we need to make sure max_new_tokens is set properly
            if inputs_embeds is not None and input_ids is None:
                # Use max_new_tokens instead of max_length when using inputs_embeds
                if max_new_tokens is None:
                    max_new_tokens = max_length

                # Create a dummy input_ids tensor to help with generation
                batch_size = inputs_embeds.size(0)
                bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
                input_ids = torch.tensor([[bos_token_id]] * batch_size, device=inputs_embeds.device)
                
                # For inputs_embeds case, we'll use max_new_tokens instead of max_length
                output_ids = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
            else:
                # For input_ids case, we can use max_length as originally intended
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    max_length=max_length,
                    **kwargs
                )
                
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

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
        
        # Vision to text projection
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_encoder.output_dim, text_decoder.model.config.hidden_size),
            nn.GELU(),
            nn.Layer(text_decoder.model.config.hidden_size)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(text_decoder.model.config.hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(text_decoder.model.config.hidden_size)
        
        # Initialize weights
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
        """
        Enhanced forward pass with dual-view processing
        """
        device = front_images.device
        
        # Encode vision features
        vision_embeds = self.vision_encoder(front_images, lateral_images)
        vision_embeds = self.vision_proj(vision_embeds)
        
        # Add positional encoding
        vision_embeds = vision_embeds + self.pos_encoder(vision_embeds)
        vision_embeds = self.layer_norm(vision_embeds)
        vision_embeds = self.dropout(vision_embeds)

        if text is None:
            raise ValueError("Text input is required for training mode")
            
        # Process text inputs
        input_ids, attention_mask = self.text_decoder.encode_text(text)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get text embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        text_embeds = text_embeds + self.pos_encoder(text_embeds)
        
        # Multimodal fusion
        fused = self.cross_attention(
            query=text_embeds,
            key=vision_embeds,
            value=vision_embeds,
            key_padding_mask=None
        )
        fused = self.layer_norm(fused + text_embeds)  # Residual connection
        
        return self.text_decoder(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            labels=labels if labels is not None else input_ids
        )

    def generate_report(self, front_images, lateral_images, prompt_text=None, **kwargs):
        """
        Enhanced generation with medical prompts
        """
        device = front_images.device
        batch_size = front_images.shape[0]
        
        # Encode vision features
        vision_embeds = self.vision_encoder(front_images, lateral_images)
        vision_embeds = self.vision_proj(vision_embeds)
        vision_embeds = vision_embeds + self.pos_encoder(vision_embeds)
        
        # Create medical prompts
        if prompt_text is None:
            prompt_text = ["Findings: Impression:"] * batch_size
            
        # Encode prompts
        input_ids, attention_mask = self.text_decoder.encode_text(prompt_text)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get prompt embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        text_embeds = text_embeds + self.pos_encoder(text_embeds)
        
        # Multimodal fusion
        fused = self.cross_attention(
            query=text_embeds,
            key=vision_embeds,
            value=vision_embeds,
            key_padding_mask=None
        )
        fused = self.layer_norm(fused + text_embeds)  # Residual connection
        
        # Generation parameters
        gen_kwargs = {
            'inputs_embeds': fused,
            'attention_mask': attention_mask,
            'max_length': self.config.max_gen_length,
            'num_beams': self.config.num_beams,
            'repetition_penalty': self.config.repetition_penalty,
            'length_penalty': self.config.length_penalty,
            'early_stopping': True,
            'pad_token_id': self.text_decoder.tokenizer.eos_token_id
        }
        gen_kwargs.update(kwargs)
        
        generated_ids = self.text_decoder.model.generate(**gen_kwargs)
        return self.text_decoder.decode(generated_ids)
        
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
    for batch  in progress_bar:
        front = batch['front'].to(device)
        lateral = batch['lateral'].to(device)
        reports = batch['report']
        
        # Forward pass
        outputs = model(front=front, lateral=lateral, text=reports)
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
    transform_front = XrayReportDataset.get_transform_front()
    transform_lateral = XrayReportDataset.get_transform_lateral()
    train_ds = XrayReportDataset(config.train_csv, config.image_dir, transform_front=transform_front, transform_lateral = transform_lateral, max_length=config.max_len)
    val_ds = XrayReportDataset(config.cv_csv, config.image_dir, transform_front=transform_front, transform_lateral = transform_lateral, max_length=config.max_len)
    
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
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
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