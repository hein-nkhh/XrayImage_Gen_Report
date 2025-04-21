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