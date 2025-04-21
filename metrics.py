from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('wordnet')
nltk.download('punkt')

def calculate_metrics(predictions, references):
    """
    Calculate NLG evaluation metrics: BLEU-1,2,3,4, METEOR, and ROUGE-L
    
    Args:
        predictions: List of generated report strings
        references: List of ground truth report strings
    
    Returns:
        Dictionary of metrics
    """
    # Tokenize predictions and references
    smooth = SmoothingFunction().method1
    pred_tokens = [word_tokenize(p.lower()) for p in predictions]
    ref_tokens = [[word_tokenize(r.lower())] for r in references]
    
    # BLEU Metrics
    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # METEOR
    m_scores = [meteor_score([word_tokenize(r)], word_tokenize(p)) for r, p in zip(references, predictions)]
    meteor = np.mean(m_scores)
    
    # ROUGE-L
    def lcs(x, y):
        m, n = len(x), len(y)
        L = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                L[i][j] = L[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(L[i - 1][j], L[i][j - 1])
        return L[m][n]
    
    rouge_scores = []
    for r, p in zip(references, predictions):
        rt, pt = word_tokenize(r), word_tokenize(p)
        l = lcs(rt, pt)
        rec = l / len(rt) if rt else 0
        prec = l / len(pt) if pt else 0
        rouge_scores.append((2 * rec * prec / (rec + prec)) if rec + prec else 0)
    
    rouge_l = np.mean(rouge_scores)
    
    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge_l": rouge_l
    }