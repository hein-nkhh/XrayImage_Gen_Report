from tqdm import tqdm
import torch 

def generate_and_decode(model, mlp, test_loader, device):
    model.model.eval()
    mlp.eval()
    generated, references = [], []
    with torch.no_grad():
        for input_batch, label_batch in tqdm(test_loader, desc='Evaluating'):
            input_batch = input_batch.to(device)
            embeddings = mlp(input_batch)
            outputs = model.generate(embeddings)
            gen_texts = model.decode(outputs)
            ref_texts = model.decode(label_batch)
            generated.extend([g.strip() for g in gen_texts])
            references.extend([r.strip() for r in ref_texts])
    return generated, references