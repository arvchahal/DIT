from routers import Router

class DomainSimplifiedRouter(Router):
    """
    Domain Simplified Router is a simplified variation of the Domain Router. Where instead of 
    creating a tally, we match with the first word that is associated with a domain. If 
    there are no changes, it is randomly selected.
    """
    def __init__(self, *, experts: list[str], mapping_expert_to_descriptors: dict[str, list[str]]):
        super().__init__(experts=experts)
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
        for word in query.split():
            if word in self.domains:
                return self.domains[word]
        return super().fallback()
