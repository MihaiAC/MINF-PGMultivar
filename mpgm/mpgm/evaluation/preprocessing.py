import numpy as np

class Preprocessor():
    def __init__(self):
        pass

    def preprocess(self, data:np.ndarray):
        pass

class ClampMax(Preprocessor):
    def __init__(self, max_value:int):
        super().__init__()
        self.max_value = max_value

    def preprocess(self, data:np.ndarray):
        data[data > self.max_value] = self.max_value
