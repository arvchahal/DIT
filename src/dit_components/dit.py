from dit_components import DitExpert,DITRouter
from routers.router import Router
class DIT:
    def __init__(self, experts: dict[str, DitExpert],router:Router):
        self.table = experts
        self.router = DITRouter(self.table,router:Router)

    def exec(self, query: str):
        key = self.router.route(query)
        return self.table[key].run_model(query)
