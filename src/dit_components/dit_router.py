from routers.router import Router
from transformers import pipeline
from collections import defaultdict
from dit_components import DitModel

class DITRouter:
    def __init__(self, expert_table: dict[str, DitModel]):
        self.expert_table = expert_table
        #  zero-shot model for quick classification
        self.zs = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")

    def route(self, query: str) -> str:
        candidates = list(self.expert_table.keys())
        res = self.zs(query, candidate_labels=candidates)
        return res["labels"][0]