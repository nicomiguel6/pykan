#usr/bin/env python3

from kan import *
from my_classes import Dataset_list as Dataset
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
CBF = models.CBF(dims=[2,1], function=lambda x: (x[:,[0]])**2 + (x[:,[1]])**2 - R**2)

## Create dataset
dataset = create_dataset(CBF.get_function(), n_var = CBF.dims[0], device=device)

## Create params
width = [CBF.dims[0], 5, CBF.dims[1]] # [ dims_in, L [of width N], dims_out]
params = {'width' : width, 
          'opt' : 'LBFGS', 
          'device' : device, 
          'grid' : 3,
          'k' : 3, 
          'steps' : 200}

## Create KAN
kan = KAN(width=params['width'], grid=params['grid'], k = params['k'], device=params['device'])

## Train
results = kan.fit(dataset, opt=params['opt'], steps=params['steps'], lr = 1e-3)

train_losses = []
test_losses = []

train_losses += results['train_loss']
test_losses += results['test_loss']

plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.ylabel('RMSE')
plt.xlabel('step')
plt.yscale('log')
plt.show()


## Create dataset for FCNN
CBF_fun = CBF.get_function()
num_samples = 1000
train_data = np.zeros((num_samples,2))
train_labels = np.zeros((num_samples,1))

# Generate training points with labels
train_data = np.zeros((num_samples, 2))
train_labels = np.zeros((num_samples, 1))
for iter in range(num_samples):
    x = np.random.rand(1, 2) * 100
    y = CBF_fun(x)
    # print(f'x: {x}, y: {y}')

    train_data[iter, :] = x
    train_labels[iter] = y

# Generate testing points with labels
test_data = np.zeros((num_samples, 2))
test_labels = np.zeros((num_samples, 1))
for iter in range(num_samples):
    x = np.random.rand(1, 2) * num_samples
    y = CBF_fun(x)
    # print(f'x: {x}, y: {y}')

    test_data[iter, :] = x
    test_labels[iter] = y

train_data = np.double(train_data)
train_labels = np.double(train_labels)
test_data = np.double(test_data)
test_labels = np.double(test_labels)
    
# Parameters
params = {'batch_size': 50,
          'shuffle': True}

# Generators
training_set = Dataset(train_data, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset(test_data, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_set, **params)

## Create FCNN (equivalent is to have L hidden layers of width N)
model = models.FCNet(nFeatures=CBF.dims[0], nHidden1=5, nOut=CBF.dims[1], mean=0, std=1, device=device, bn=False)


# Initialize the optimizer.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()



epochs = 10
train_losses, test_losses = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses = train(train_dataloader, model, loss_fn, optimizer, train_losses, test_losses)
    #test_losses = test(test_dataloader, model, loss_fn, test_losses)
print("Training Done!")

torch.save(model.state_dict(), "model_fc.pth")
print("Saved PyTorch Model State to model_xx.pth")

print(np.shape(train_losses))
print(np.shape(test_losses))

plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.ylabel('RMSE')
plt.xlabel('step')
plt.yscale('log')
plt.show()
