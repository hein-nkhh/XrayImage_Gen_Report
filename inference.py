# inference.py

import torch

def run_inference(model, dataloader, device):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in dataloader:
            front = batch['front'].to(device)
            lateral = batch['lateral'].to(device)
            report = batch['report']  # list of strings

            # Generate reports
            generated_ids = model.generate(front, lateral, max_length=150)
            generated_texts = model.decode(generated_ids)

            # Save
            predictions.extend(generated_texts)
            references.extend(report)

    return predictions, references
