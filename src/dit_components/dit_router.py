from routers.router import Router
from transformers import pipeline
from collections import defaultdict
from dit_components import DitExpert


class DITRouter:
    def __init__(self, expert_table: dict[str, DitExpert], router: Router):
        self.expert_table = expert_table
        self.router: Router = router

    def route(self, query: str) -> str:
        return self.router.route(query)
