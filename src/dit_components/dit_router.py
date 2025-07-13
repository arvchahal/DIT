from routers.router import Router
class DITRouter:
    """
    Implements a routing object allows us to vary different types of routers and still run the code
    """
    def init(self, router:Router):
        self.router = router
