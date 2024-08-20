from kan import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import models


## Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

## Define ground truth CBF
R = 5
CBF = models.CBF(dims=2, function=lambda x: x[0]**2 + x[1]**2 - R**2)

## Create dataset
dataset = create_dataset(CBF.get_function(), nVars = CBF.dims, device=device)

## Create params
width = [CBF.dims, 1]
params = {'width' : width, 
          'opt' : 'LBFGS', 
          'device' : device, 
          'grid' : 3,
          'k' : 3, 
          'steps' : 20}

## Create KAN
kan = KAN(width=params['width'], grid=params['grid'], k = params['k'], device=params['device'])