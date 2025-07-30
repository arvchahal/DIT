from abc import ABC, abstractmethod


class Router(ABC):
    """
    Base router class
    experts will be a list of strings that will be what our different models specialize in ie finance
    """

    def __init__(self, experts: list[str]):
        self.experts = experts
        self.total_models = len(experts)

    @abstractmethod
    def route(self, query: str) -> str:
        pass
