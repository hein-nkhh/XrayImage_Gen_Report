from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')

def calculate_bleu(reference_texts, generated_texts):
    refs = [[word_tokenize(ref.lower())] for ref in reference_texts]
    hyps = [word_tokenize(hyp.lower()) for hyp in generated_texts]
    smooth = SmoothingFunction().method1
    return {
        'bleu1': corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth),
        'bleu2': corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        'bleu3': corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
        'bleu4': corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }


def calculate_meteor(reference_texts, generated_texts):
    return np.mean([meteor_score([ref], gen) for ref, gen in zip(reference_texts, generated_texts)])


def calculate_rouge_n(ref_tokens, gen_tokens, n=1):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    ref_ngrams = get_ngrams(ref_tokens, n)
    gen_ngrams = get_ngrams(gen_tokens, n)
    ref_counts = {ng: ref_ngrams.count(ng) for ng in set(ref_ngrams)}
    gen_counts = {ng: gen_ngrams.count(ng) for ng in set(gen_ngrams)}
    overlap = sum(min(ref_counts.get(ng, 0), gen_counts.get(ng, 0)) for ng in gen_counts)
    precision = overlap / len(gen_ngrams) if gen_ngrams else 0.0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def calculate_rouge_l(ref_tokens, gen_tokens):
    m, n = len(ref_tokens), len(gen_tokens)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == gen_tokens[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i][j + 1], L[i + 1][j])
    lcs_len = L[m][n]
    precision = lcs_len / n if n else 0.0
    recall = lcs_len / m if m else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def calculate_rouge_scores(reference_texts, generated_texts):
    ref_tokens = [word_tokenize(ref.lower()) for ref in reference_texts]
    gen_tokens = [word_tokenize(gen.lower()) for gen in generated_texts]
    r1, r2, r3, r4, rl = [], [], [], [], []
    for ref, gen in zip(ref_tokens, gen_tokens):
        r1.append(calculate_rouge_n(ref, gen, n=1))
        r2.append(calculate_rouge_n(ref, gen, n=2))
        r3.append(calculate_rouge_n(ref, gen, n=3))
        r4.append(calculate_rouge_n(ref, gen, n=4))
        rl.append(calculate_rouge_l(ref, gen))
    return {
        'rouge1': np.mean(r1),
        'rouge2': np.mean(r2),
        'rouge3': np.mean(r3),
        'rouge4': np.mean(r4),
        'rougeL': np.mean(rl),
    }