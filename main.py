import argparse
from config import Config
from trainer import train_model
from inference import load_model, evaluate_test_set

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
        
    elif args.mode == 'test':
        # Load model from checkpoint
        checkpoint_path = args.checkpoint or Config.best_model_path
        model = load_model(checkpoint_path)
        
        # Evaluate on test set
        metrics, _, _ = evaluate_test_set(model)
        
if __name__ == '__main__':
    main()