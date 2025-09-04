from routers import Router

class DomainRouter(Router):
    """
    Domain Router uses domain information to route queries to the appropriate expert.
    Prior domain knowledge for counts is necessary.
    """
    def __init__(self, *, experts: list[str], mapping_expert_to_descriptors: dict[str, list[str]]):
        super().__init__(experts=experts)
        self._init_domains(mapping_expert_to_descriptors)

    def _init_domains(self, mapping_expert_to_descriptors: dict[str, list[str]]):
        self.domains = {}
        self.ambiguous_descriptors = set()
        for expert, descriptors in mapping_expert_to_descriptors.items():
            for descriptor in descriptors:
                if descriptor in self.domains or descriptor in self.ambiguous_descriptors:
                    self.domains.pop(descriptor, None)
                    self.ambiguous_descriptors.add(descriptor)
                else:
                    self.domains[descriptor] = expert

    def route(self, query: str) -> str:
        """
        For each word in query, check if in domain. The expert with the most number of queries 
        gets picked. Otherwise, fallback to a random expert.
        """
        tallies = {expert: 0 for expert in self.experts}
        for word in query.split():
            if word in self.ambiguous_descriptors:
                continue
            if word in self.domains:
                tallies[self.domains[word]] += 1

        best_expert = max(tallies, key=tallies.get, default=None)
        return best_expert if best_expert else super().route(query)
