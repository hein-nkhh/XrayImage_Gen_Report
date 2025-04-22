import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import torch
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('punkt')
# BLEU với smoothing
def compute_bleu_scores(references, predictions):
    smoothie = SmoothingFunction().method4
    scores = {
        'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []
    }

    for ref, pred in zip(references, predictions):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()

        scores['bleu1'].append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        scores['bleu2'].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        scores['bleu3'].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        scores['bleu4'].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
    
    return {k: sum(v)/len(v) for k, v in scores.items()}


def compute_meteor_score(references, predictions):
    meteor = evaluate.load("meteor")
    return meteor.compute(predictions=predictions, references=references)['meteor']


def compute_rouge_l_score(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
    return sum(scores) / len(scores)


def evaluate_all(preds, refs):
    bleu = compute_bleu_scores(refs, preds)
    meteor = compute_meteor_score(refs, preds)
    rouge_l = compute_rouge_l_score(refs, preds)

    results = {
        **bleu,
        'meteor': meteor,
        'rouge_l': rouge_l
    }
    return results

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            front = batch['front'].to(device)
            lateral = batch['lateral'].to(device)
            ref_texts = batch['report']

            # Sinh báo cáo từ model
            generated_ids = model.generate(front, lateral, max_length=153)
            generated_texts = model.decode(generated_ids)

            predictions.extend(generated_texts)
            references.extend(ref_texts)

    return evaluate_all(predictions, references)