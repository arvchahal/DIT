from abc import ABC, abstractmethod

class Router(ABC):
    """
    Base router class
    experts will be a list of strings that will be what our different models specialize in ie finance
    """

    def init(self,experts:list[str]):
        self.experts = experts
        self.total_models = len(experts)
    
    @abstractmethod
    def route(self):
        pass
