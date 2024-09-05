import torch.nn as nn
import torch
from kan import KAN
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from cvxopt import matrix, solvers


# For setting up basic neural network to train on the CBF dataset
class FCNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nOut, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nCls = nOut
        self.mean = mean
        self.std = std
        self.device = device
        self.bn = bn

        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nOut).double()
    
    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = self.fc21(x)  # No activation function here for regression
        
        return x21

    
# def kanParams():
#     def __init__(self, width, **kwargs):

#         self.width = width

#         if 'opt' in kwargs:
#             self.opt = kwargs.pop('opt')
#         if 'device' in kwargs:
#             self.device = kwargs.pop('device')
#         if 'grid' in kwargs:
#             self.grid = kwargs.pop('grid')
#         if 'k' in kwargs:
#             self.k = kwargs.pop('k')
#         if 'steps' in kwargs:
#             self.steps = kwargs.pop('steps')   

#     def set_params(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)
    
#     def get_params(self, **kwargs):
#         params = []
#         for key in kwargs:
#             if key not in self.__dict__:
#                 raise ValueError("Invalid parameter")
#             params.append(getattr(self, key))
        
        
    
# For setting up basic CBF
class CBF:
    def __init__(self, dims, function):
        super().__init__()
        self.dims = dims
        self.function = function

    def get_function(self):
        return self.function
    
    def set_function(self, function):
        self.function = function
    
    def evaluate(self, x):

        assert x.shape[1] == self.dims, "Input dimension does not match the CBF dimension"

        return self.function(x)
    
