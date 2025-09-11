import pytest
from evaluator.accuracy_metrics import bertscore_score, bleu_score, rouge_score_fn

def test_bertscore_score():
    candidates = ["The cat sat on the mat."]
    references = ["A cat is sitting on a mat."]
    P, R, F1 = bertscore_score(candidates, references)
    assert len(P) == 1 and len(R) == 1 and len(F1) == 1
    assert 0 <= F1[0] <= 1

def test_bleu_score():
    candidate = "The cat sat on the mat."
    reference = "A cat is sitting on a mat."
    score = bleu_score(candidate, reference)
    assert 0 <= score <= 1

def test_rouge_score_fn():
    candidate = "The cat sat on the mat."
    reference = "A cat is sitting on a mat."
    scores = rouge_score_fn(candidate, reference)
    assert "rouge1" in scores and "rougeL" in scores
    assert 0 <= scores["rouge1"].fmeasure <= 1
