import torch
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

def evaluate_model(mlp, generator, test_loader, tokenizer, device):
    """Đánh giá mô hình sinh báo cáo bằng BLEU, METEOR và ROUGE."""

    mlp.eval()
    generator.model.eval()

    generated_texts = []
    reference_texts = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            embedded = mlp(inputs)
            embedded_seq = embedded.unsqueeze(1)

            try:
                output_ids = generator.model.generate(
                    inputs_embeds=embedded_seq,
                    max_length=153,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                batch_gen = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                batch_ref = tokenizer.batch_decode(labels, skip_special_tokens=True)

                for gen, ref in zip(batch_gen, batch_ref):
                    generated_texts.append(gen.strip())
                    reference_texts.append(ref.strip())

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("❌ CUDA OOM — skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                raise e

    if not generated_texts:
        print("⚠️ Không có văn bản nào được tạo ra.")
        return

    # --- BLEU ---
    smooth = SmoothingFunction().method1
    gen_tok = [word_tokenize(t.lower()) for t in generated_texts]
    ref_tok = [[word_tokenize(t.lower())] for t in reference_texts]

    bleu1 = corpus_bleu(ref_tok, gen_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tok, gen_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tok, gen_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tok, gen_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    meteor = np.mean([meteor_score([r[0]], g) for r, g in zip(ref_tok, gen_tok)])

    # --- ROUGE ---
    rouge_scores = {f"rouge{n}": [] for n in range(1, 5)}
    rouge_scores["rougeL"] = []

    for ref, gen in zip(ref_tok, gen_tok):
        for n in range(1, 5):
            rouge_scores[f"rouge{n}"].append(calculate_rouge_n(ref[0], gen, n))
        rouge_scores["rougeL"].append(calculate_rouge_l(ref[0], gen))

    # --- Print Metrics ---
    print("\n📊 Evaluation Metrics:")
    print(f"BLEU-1:  {bleu1:.4f}")
    print(f"BLEU-2:  {bleu2:.4f}")
    print(f"BLEU-3:  {bleu3:.4f}")
    print(f"BLEU-4:  {bleu4:.4f}")
    print(f"METEOR:  {meteor:.4f}")
    print(f"ROUGE-1: {np.mean(rouge_scores['rouge1']):.4f}")
    print(f"ROUGE-2: {np.mean(rouge_scores['rouge2']):.4f}")
    print(f"ROUGE-3: {np.mean(rouge_scores['rouge3']):.4f}")
    print(f"ROUGE-4: {np.mean(rouge_scores['rouge4']):.4f}")
    print(f"ROUGE-L: {np.mean(rouge_scores['rougeL']):.4f}")

    print("\n🔍 Example Generations:")
    for i in range(min(5, len(generated_texts))):
        print(f"\n--- Example {i+1} ---")
        print(f"Generated: {generated_texts[i]}")
        print(f"Reference: {reference_texts[i]}")

# --- Helper Functions for ROUGE ---

def calculate_rouge_n(ref_tokens, gen_tokens, n=1):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    ref_ngrams = get_ngrams(ref_tokens, n)
    gen_ngrams = get_ngrams(gen_tokens, n)

    if not ref_ngrams or not gen_ngrams: return 0.0

    ref_counts = {ng: ref_ngrams.count(ng) for ng in set(ref_ngrams)}
    gen_counts = {ng: gen_ngrams.count(ng) for ng in set(gen_ngrams)}
    
    overlap = sum(min(gen_counts.get(ng, 0), ref_counts.get(ng, 0)) for ng in gen_counts)

    precision = overlap / len(gen_ngrams)
    recall = overlap / len(ref_ngrams)
    if precision + recall == 0: return 0.0
    return (2 * precision * recall) / (precision + recall)

def calculate_rouge_l(ref_tokens, gen_tokens):
    m, n = len(ref_tokens), len(gen_tokens)
    if m == 0 or n == 0: return 0.0

    L = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == gen_tokens[j]:
                L[i+1][j+1] = L[i][j] + 1
            else:
                L[i+1][j+1] = max(L[i][j+1], L[i+1][j])
    lcs_len = L[m][n]

    recall = lcs_len / m
    precision = lcs_len / n
    return (2 * recall * precision) / (recall + precision) if recall + precision > 0 else 0.0
