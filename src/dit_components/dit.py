from dit_components.dit_expert import DitModel
from dit_components.dit_router import DITRouter
class DIT:
    def __init__(self, experts: dict[str, DitModel]):
        self.table = experts
        self.router = DITRouter(self.table)

    def exec(self, query: str):
        key = self.router.route(query)
        return self.table[key].run_model(query)