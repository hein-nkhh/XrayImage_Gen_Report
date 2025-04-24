import os
import torch
from models.mlp import MLP
from models.biobart_wrapper import BioBARTWrapper
from preprocessing.dataset_builder import normalize_features, to_tensor, build_dataset
from preprocessing.report_cleaning import clean_text
from training.train import train_step
from training.validate import validate
from training.evaluate import generate_and_decode
from utils.metrics_helper import calculate_bleu, calculate_meteor, calculate_rouge_scores
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mlp = MLP(config['mlp_input_dim'], config['mlp_hidden_dim'], config['mlp_output_dim']).to(device)
    biobart = BioBARTWrapper(config['biobart_model_name'], device)

    X_train = normalize_features(config['X_train'])
    y_train = biobart.tokenize_reports([clean_text(txt) for txt in config['y_train']])
    train_dataset = build_dataset(to_tensor(X_train), y_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': mlp.parameters(), 'lr': config['lr_mlp']},
        {'params': biobart.model.parameters(), 'lr': config['lr_bart']}
    ])

    for epoch in range(config['epochs']):
        mlp.train(), biobart.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = batch
            loss = train_step(mlp, biobart, optimizer, x, y, device)
            total_loss += loss
        val_loss = validate(biobart, mlp, train_loader, device)
        print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f, } Validation Loss: {val_loss:.4f}")

    # Evaluation
    generated, references = generate_and_decode(biobart, mlp, train_loader, device)
    bleu_scores = calculate_bleu(references, generated)
    meteor = calculate_meteor(references, generated)
    rouge_scores = calculate_rouge_scores(references, generated)

    print("\nEvaluation Results:")
    print("BLEU:", bleu_scores)
    print("METEOR:", meteor)
    print("ROUGE:", rouge_scores)


if __name__ == '__main__':
    main()
