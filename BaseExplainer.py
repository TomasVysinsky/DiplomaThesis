class BaseExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, inputs, target):
        raise NotImplementedError