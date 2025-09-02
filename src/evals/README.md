# Evaluation

## Metrics
1. Efficiency: Speed vs Concision of Response
2. Calibration of model: Confidence of Response

## Methods
- Automatic Semantic Similarity
    - N-gram overlap metrics: BLEU, ROUGE, METEOR (if ground truth, but there is not)
    - BERTScore, Sentence-BERT cosine similarity (meaing based scoring)
    - Task-specific Scores
        - Factual QA Exact Match (EM), F1 overlap on entities/keywords
- LLM-as-a-judge (on correctness, coverage, and style)?
- Human evaluation for final benchmarking/sensitive tasks