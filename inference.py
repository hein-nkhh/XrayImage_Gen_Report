import torch
from torch.utils.data import DataLoader
from config import Config
from model import VisionEncoder, TextDecoder, CrossAttention, XrayReportModel
from dataset import XrayReportDataset
from utils import collate_fn

def load_model(checkpoint_path, config=Config):
    """Load trained model from checkpoint"""
    device = torch.device(config.device)
    
    # Initialize model components
    vision_encoder = VisionEncoder(model_name=config.vision_encoder_name, output_dim=config.vision_output_dim)
    text_decoder = TextDecoder(model_name=config.text_decoder_model)
    cross_attention = CrossAttention(hidden_dim=config.cross_attn_dim, num_heads=config.cross_attn_heads)
    
    # Create model
    model = XrayReportModel(vision_encoder, text_decoder, cross_attention)
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def run_inference(model, image_paths, config=Config):
    """
    Run inference on new images
    
    Args:
        model: Loaded XrayReportModel
        image_paths: List of tuples [(frontal_image_path, lateral_image_path), ...]
        config: Configuration
        
    Returns:
        List of generated reports
    """
    device = torch.device(config.device)
    transform = XrayReportDataset.get_transform()
    
    images = []
    for front_path, lateral_path in image_paths:
        # Load and process images
        front_img = transform(Image.open(front_path).convert('RGB'))
        lateral_img = transform(Image.open(lateral_path).convert('RGB'))
        combined_img = torch.cat([front_img, lateral_img], dim=0)
        images.append(combined_img)
    
    # Stack images into batch
    image_batch = torch.stack(images, dim=0).to(device)
    
    # Generate reports
    with torch.no_grad():
        reports = model.generate_report(
            image_batch,
            prompt_text="The chest x-ray shows",  # Optional starting prompt
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    
    return reports

def evaluate_test_set(model, config=Config):
    """Evaluate model on test set"""
    device = torch.device(config.device)
    
    # Create test dataset and dataloader
    transform = XrayReportDataset.get_transform()
    test_ds = XrayReportDataset(config.test_csv, config.image_dir, transform=transform, max_length=config.max_len)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Run evaluation
    model.eval()
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for images, reports in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            preds = model.generate_report(images)
            
            all_preds.extend(preds)
            all_refs.extend(reports)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_refs)
    
    print("Test Set Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    return metrics, all_preds, all_refs