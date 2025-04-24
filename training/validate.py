import torch
from tqdm import tqdm

def validate(model, mlp, val_loader, device):
    model.model.eval()
    mlp.eval()
    total_loss, count = 0, 0
    with torch.no_grad():
        for input_batch, label_batch in val_loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            embeddings = mlp(input_batch)
            loss = model.get_loss(embeddings, label_batch)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float('inf')