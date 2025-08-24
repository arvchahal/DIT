from routers import Router
import itertools


class SimpleRouter(Router):
    """
    This is a simple router class that will be a basic implementation that uses
    a naive way to route to MoE models
    """

    def __init__(self, *, experts: list[str]):
        super().__init__(experts=experts)
        self._rr = itertools.cycle(self.experts)

    def route(self, query: str) -> str:
        return next(self._rr)
