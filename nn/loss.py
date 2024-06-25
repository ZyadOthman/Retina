from .module import Module
 
class MSELoss(Module):
    def __init__(self):
      pass

    def forward(self, predictions, labels):
        assert labels.shape == predictions.shape, \
            "Labels and predictions shape does not match: {} and {}".format(labels.shape, predictions.shape)
        
        result = ((predictions - labels) ** 2)
        result = result.sum()
        return result / predictions.numel

    def __call__(self, *inputs):
        return self.forward(*inputs)