#!/usr/bin/python

import numpy as np
import torch
import cvxopt as cvx
from util import *
import matplotlib.pyplot as plt
import gymnasium as gym
from kan import *
from verification import *
import os
import json

def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

## Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

## Define gym Inverted Pendulum env
a = 0.25
b = 0.5
env = InvertedPendulum(a=a, b=b, Kp=0.6, Kd=0.6, render_mode='human')

## Define CBF
CBF = lambda x: 1 - ((x[0]**2)/a**2) - ((x[1]**2)/b**2) - ((x[0]*x[1])/(a*b))

env.set_CBF(CBF)

## Define alpha
gamma = 1
alpha = lambda h, u_ref: gamma*h
env.set_alpha(alpha)

## Create config dict that stores all model types and parameter counts
config_network = {
    'KAN': {'params': [[2,[1,1],1], [2,[2,2],1], [2,2,2,1], [2,3,3,1]], 'num_params': [], 'save_path': []},
    'NN': {'params': [[2,8,8,1], [2,10,10,1], [2,8,8,8,1], [2,10,10,10,1], [2,16,16,1], [2,16,16,16,1]], 'num_params': [], 'save_path': []},
    'grid': [3, 5, 10, 20, 50]
}

## Create results dict that stores all loss values for each parameter
results_dict = {
    'KAN': {str(params) + '_' + str(grid): {} for params in config_network['KAN']['params'] for grid in config_network['grid']},
    'NN': {str(params): {} for params in config_network['NN']['params']}
}

## ----------------- Training ----------------- ##
## KAN

# set up dataset
kan_CBF = lambda x: 1 - ((x[:,[0]]**2)/a**2) - ((x[:,[1]]**2)/b**2) - ((x[:,[0]]*x[:,[1]])/(a*b))
kan_dataset = create_dataset(kan_CBF, n_var=2, train_num=1000, test_num=1000, device=device)

data_range_theta = torch.linspace(-np.pi, np.pi, steps = 1001)[:,None]
data_range_theta_dot = torch.linspace(-8.0, 8.0, steps = 1001)[:,None]
data_range = torch.cat((data_range_theta, data_range_theta_dot), dim=1).to(device)

# Model and training loop
for param in config_network['KAN']['params']:
    save_path_list = []
    for grid in config_network['grid']:

        ## calculate parameters 
        # (this is done by multiplying layer n and n-1 and adding to layer n-1 times n-2, etc and then multiplying the entire sum by the number of grid points)
        num_params = 0
        for i in range(len(param)-1):
            num_params += np.sum(param[i])*np.sum(param[i+1])
        num_params *= grid
        config_network['KAN']['num_params'].append(num_params)

        if grid == 3:
            # create model
            kanModel = KAN(width=param, grid=3, k=3, device=device, seed=10)
            kanModel.update_grid_from_samples(data_range)
        else:
            kanModel = kanModel.refine(grid)
            kanModel.update_grid_from_samples(data_range)

        # train model
        results = kanModel.fit(dataset=kan_dataset, opt='LBFGS', steps=200)

        # save model to ./KAN_models. check if filename exists, create if not
        filename = './kan_models/' + 'kan[' + str([i for i in param]) + ']' + '_' + str(grid)
        kanModel.saveckpt(filename)
        save_path_list.append(filename)

        # save results to config_network
        results_dict['KAN'][str(param) + '_' + str(grid)] = results
    config_network['KAN']['save_path'].append(save_path_list)


## NN
# Set up dataset
train_data = np.double(kan_dataset['train_input'])
train_labels = np.double(kan_dataset['train_label'])
test_data = np.double(kan_dataset['test_input'])
test_labels = np.double(kan_dataset['test_label'])

# Parameters
params = {'batch_size': 1000,
          'shuffle': True}

# Generators
training_set = Dataset_list(train_data, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset_list(test_data, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_set, **params)

results_dict_NN = {
    'NN': {str(params): {} for params in config_network['NN']['params']}
}

# Model and training loop
for param in config_network['NN']['params']:
    ## calculate parameters
    num_params = 0
    for i in range(len(param)-1):
        num_params += param[i]*param[i+1] + param[i+1]
    #config_network['NN']['num_params'].append(num_params)

    # check if filepath already exists
    filename = "./NN_model_" + '[' + str([i for i in param]) + ']' + ".pth"

    # create model
    model = FCNet(width=param, mean=0, std=1, device=device, bn=False)
    model.set_dataloaders(train_dataloader, test_dataloader)

    # Initialize the optimizer.
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    epochs = 1000
    cumul_train_losses, cumul_test_losses = [], []
    train_losses, test_losses = [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses = model.train_model(model, loss_fn, optimizer, train_losses)
        test_losses = model.test(model, loss_fn, test_losses)
        cumul_train_losses.append(train_losses)
        cumul_test_losses.append(test_losses)
    print("Training Done!")

    results_dict_NN['NN'][str(param)] = {'train': cumul_train_losses, 'test': cumul_test_losses}

    # save model to ./NN_models
    filename = "./NN_model_" + '[' + str([i for i in param]) + ']' + ".pth"
    torch.save(model.state_dict(), filename)
    config_network['NN']['save_path'].append(filename)

config_network_converted = convert_numpy_types(config_network)
results_dict_NN_converted = convert_numpy_types(results_dict_NN)

# # Save config_network to a JSON file
# with open('config_network.json', 'w') as json_file:
#     json.dump(config_network_converted, json_file, indent=4)

# # Save results_dict to a JSON file
# with open('results_dict.json', 'w') as json_file:
#     json.dump(results_dict_converted, json_file, indent=4)

# Save results_dict to a JSON file
with open('results_NN_dict.json', 'w') as json_file:
    json.dump(results_dict_NN_converted, json_file, indent=4)





