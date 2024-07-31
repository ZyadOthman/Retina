from .module import Module
 
from abc import ABC

class Loss(Module, ABC):
    "Abstract class for loss functions"

    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        raise NotImplementedError
            
    def __call__(self, *inputs):
        return self.forward(*inputs)
    

class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, target):
        assert target.shape == predictions.shape, \
            "Labels and predictions shape does not match: {} and {}".format(target.shape, predictions.shape)
        
        cost = ((predictions - target) ** 2).sum() / predictions.numel
        return cost