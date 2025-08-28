from routers import Router
from collections import deque
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EmbeddingRouter(Router):
    """
    Embedding Router will route by embeddding the label for the agents as anchors 
    for the queries. An embeddings score will then be calculated for the query (TBD)
    and the embedding with the closest match to the anchor is the agent the query is
    sent to.
    """
    def __init__(self, *, experts: list[str]):
        super().__init__(experts= experts)
        self.tokenizer, self.encoder = self._init_encoder()
        self.agent_anchors_MRU_cache: deque[str] = deque(self.experts)
        self.anchor_embeddings: Dict[str, np.ndarray] = {}
        self._init_anchor_embeddings()

    def _init_encoder(self):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        return tokenizer, model

    def _get_embedding(self, text) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        vec = embeddings.squeeze().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)
    
    def _init_anchor_embeddings(self):
        """Compute and cache embeddings for each expert label once."""
        for expert in self.experts:
            self.anchor_embeddings[expert] = self._get_embedding(expert)
        return
    
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b)) # Assumption: a and b are normalized
    
    def _touch_expert(self, expert: str) -> None:
        """Move expert to the front (most recently used) of the deque."""
        try:
            self.agent_anchors_MRU_cache.remove(expert)
            self.agent_anchors_MRU_cache.appendleft(expert)
        except ValueError:
            pass

    def route(self, query: str) -> str:
        """
        Choose the expert with the highest cosien similarity to the query embedding.
        Update the MRU cache so the chosen expert moves to the front.
        """
        if not query:
            # Round Robin on fallback
            expert = self.agent_anchors_MRU_cache.popleft()
            self.agent_anchors_MRU_cache.append(expert)
            return expert

        q_vec = self._get_embedding(query)

        best_expert, best_score = None, -1.0
        for expert in self.agent_anchors_MRU_cache:
            score = self._cosine_sim(q_vec, self.anchor_embeddings[expert])
            if score > best_score:
                best_expert, best_score = expert, score

        self._touch_expert(best_expert)
        return best_expert
