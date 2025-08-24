from __future__ import annotations
from typing import Dict
from routers.router import Router
from .dit_expert import DitExpert
from .dit_router import DitRouter


class DIT:
    def __init__(self, experts: Dict[str, DitExpert], router: Router):
        self.table = experts
        self._router = router  # keep reference
        self.router_adapter = DitRouter(experts, router)

    @property
    def router(self) -> Router:
        return self._router

    @router.setter
    def router(self, new_router: Router):
        """
        Swap routing strategy at runtime.
        Existing expert table is reused.
        """
        self._router = new_router
        self.router_adapter = DitRouter(self.table, new_router)

    def exec(self, query: str):
        key = self.router_adapter.route(query)
        return self.table[key].run_model(query)
