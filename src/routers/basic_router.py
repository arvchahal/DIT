from routers.router import Router
class SimpleRouter(Router):
    """
    This is a simple router class that will be a basic implementation that uses 
    a naive way to route to MoE models
    """
    def init(self,experts:list[str]):
        super.init(experts=experts)
