from routers import Router
import itertools

class SimpleRouter(Router):
    """
    SimpleRouter performs naive round-robin routing across experts.

    Updated to accept `mapping_expert_to_descriptors` for compatibility
    with the experiment harness, though it does not use it.
    """

    def __init__(self, *, experts: list[str], mapping_expert_to_descriptors: dict[str, list[str]] | None = None, **kwargs):
        super().__init__(experts=experts)
        self._rr = itertools.cycle(self.experts)
        # compatibility field (ignored)
        self.mapping_expert_to_descriptors = mapping_expert_to_descriptors

    def route(self, query: str) -> str:
        return next(self._rr)
