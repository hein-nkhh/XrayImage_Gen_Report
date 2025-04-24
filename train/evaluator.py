import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm

def evaluate_model(mlp, generator, test_loader):
    mlp.eval()
    generator.model.eval()
    tokenizer = generator.tokenizer

    predictions, references = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(generator.model.device)
            labels = labels.cpu()
            embeddings = mlp(inputs)
            generated = generator.generate(embeddings)

            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend([g.strip() for g in generated])
            references.extend([r.strip() for r in decoded_refs])

    # Tokenize for BLEU/METEOR
    pred_tokens = [word_tokenize(p.lower()) for p in predictions]
    ref_tokens = [[word_tokenize(r.lower())] for r in references]

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    meteor = np.mean([meteor_score([r[0]], p) for r, p in zip(ref_tokens, pred_tokens)])

    print(f"\n📊 Evaluation Scores:")
    print(f"BLEU-1:  {bleu1:.4f}")
    print(f"BLEU-2:  {bleu2:.4f}")
    print(f"BLEU-3:  {bleu3:.4f}")
    print(f"BLEU-4:  {bleu4:.4f}")
    print(f"METEOR:  {meteor:.4f}")

    print("\n🔍 Sample Predictions:")
    for i in range(min(3, len(predictions))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Generated: {predictions[i]}")
        print(f"Reference: {references[i]}")
