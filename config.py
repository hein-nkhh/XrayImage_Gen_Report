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