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
    vision_dim = 1024
    vision_hidden_size = 1024

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
