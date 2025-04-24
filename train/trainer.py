import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from config import LR_MLP, LR_BART, CHECKPOINT_DIR, CHECKPOINT_PATH, EPOCHS, PATIENCE, DEVICE

    
def train_step(mlp, generator, batch, optimizer, tokenizer):
    mlp.train()
    generator.model.train()

    inputs, labels = batch
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()
    embeddings = mlp(inputs)

    attention_mask = (labels != tokenizer.pad_token_id).long().to(DEVICE)
    loss = generator.get_loss(embeddings, labels, attention_mask)

    loss.backward()
    clip_grad_norm_(list(mlp.parameters()) + list(generator.model.parameters()), max_norm=1.0)
    optimizer.step()
    return loss.item()

def validate(mlp, generator, val_loader, tokenizer):
    mlp.eval()
    generator.model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            embeddings = mlp(inputs)
            mask = (labels != tokenizer.pad_token_id).long().to(DEVICE)
            loss = generator.get_loss(embeddings, labels, mask)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_model(mlp, generator, train_loader, val_loader):
    optimizer = AdamW([
        {'params': mlp.parameters(), 'lr': LR_MLP},
        {'params': generator.model.parameters(), 'lr': LR_BART}
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        mlp.train()
        generator.model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            loss = train_step(mlp, generator, batch, optimizer, generator.tokenizer)
            total_loss += loss
            progress.set_postfix({"train_loss": f"{loss:.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate(mlp, generator, val_loader, generator.tokenizer)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            
            if os.path.exists(CHECKPOINT_PATH) and os.path.isfile(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)
                
            torch.save({
                'mlp': mlp.state_dict(),
                'biobart': generator.model.state_dict()
            }, CHECKPOINT_PATH)
            print(f"✅ Saved checkpoint to {CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break
