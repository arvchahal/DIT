from transformers import pipeline
class DitModel():
    """
    main class for holding the transformer objects
    way to abstractly hold transformer models without actually having a direct reference to them
    in a table.
    
    model param should be the loaded model object 
     
    """
    def init(self, model:any = None):
        self.model  = model | None 

    #either input the model object that is laoded or the link to load the model
    @property
    def load_model(self,model:any=None,task:str=None,model_name:str=None) ->bool:
        if model:
            self.model = model

        else:
            if not (task and model):
                raise ValueError("Either a model object or task and model_name must be provided.")
            try:
                self.model = pipeline(task,model_name)
            except Exception as e:
                raise ValueError(f"Error loading model: {e}")
        return True
    
    def run_model(self,query):
        return self.model(query)


            
            
        