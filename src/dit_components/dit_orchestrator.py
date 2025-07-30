import time
from dit_components import DIT, DITRouter


class DITOrchestrator:
    """Runs a batch of queries through a DIT instance, respecting a time budget."""

    def __init__(self, dit: DIT):
        self.dit = dit

    def run(self, queries: list[str], total_time: int):
        if not queries:
            raise ValueError("Query list is empty")
        start = time.time()
        out = []
        for q in queries:
            if time.time() - start > total_time:
                break
            out.append(self.dit.exec(q))
        return out
