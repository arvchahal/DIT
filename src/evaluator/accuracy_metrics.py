from typing import List, Tuple
import bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def bertscore_score(candidates: List[str], references: List[str]) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute BertScore for each candidate-reference pair.
    Idea: Use BERT embeddings to compute similarity at the token level.
    Returns (precision, recall, f1) lists.
    """
    P, R, F1 = bert_score.score(candidates, references, lang="en", rescale_with_baseline=True)
    return list(P), list(R), list(F1)

def bleu_score(candidate: str, reference: str) -> float:
    """
    Compute BLEU score for a candidate and reference with smoothing.
    Idea: Use n-gram overlap with smoothing to avoid zero scores.
    Returns a BLEU score.
    """
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)

def rouge_score_fn(candidate: str, reference: str) -> dict:
    """
    Compute ROUGE scores for a candidate and reference.
    Idea: Measure overlap of n-grams and longest common subsequence.
    Returns a dictionary with ROUGE-1 and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)
