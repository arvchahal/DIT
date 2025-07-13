import time
from dit_components import DIT, DITRouter
class DITOrchestrator():
    """
    The acutal inference engine that will bring all components together
    """
    def init(self,router:DITRouter,table:DIT, ):
        self.router = router
        self.table = table

    def run(self, queries:list[str], total_time:int):
        if len(queries)<1:
            raise ValueError("Queries object must not be empty")
        start_time = time.time()
        i =0
        responses = []
        while i< len(queries) and int(time.time-start_time) <total_time:
            _response = self.table.exec(query=queries[i])
            responses.append(_response)
        return response
