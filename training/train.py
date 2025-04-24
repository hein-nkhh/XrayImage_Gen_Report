import torch
from torch.nn.utils import clip_grad_norm_

def train_step(mlp, model, optimizer, input_batch, label_batch, device):
    mlp.train()
    model.model.train()

    input_batch = input_batch.to(device)
    label_batch = label_batch.to(device)
    optimizer.zero_grad()

    embeddings = mlp(input_batch)
    loss = model.get_loss(embeddings, label_batch)
    loss.backward()
    clip_grad_norm_(list(mlp.parameters()) + list(model.model.parameters()), max_norm=1.0)
    optimizer.step()
    return loss.item()