from routers import Router
from collections import deque


class EmbeddingRouter(Router):
    """
    Embedding Router will route by embeddding the label for the agents as anchors 
    for the queries. An embeddings score will then be calculated for the query (TBD)
    and the embedding with the closest match to the anchor is the agent the query is
    sent to.
    """
    def __init__(self, *, expert: list[str]):
        super().__init__(experts= experts)
        self.encoder = self._init_encoder()
        self.agent_anchors_cache = None # TODO: implement

    def _init_encoder(self):
        pass

    def _get_embedding_label(self, expert_label):
        pass

    def _get_embedding_query(self, query):
        pass
    
    def route(self, query: str) -> str:
        pass