#!/usr/bin/python

import numpy as np
import torch
import cvxopt as cvx
from util import *
import matplotlib.pyplot as plt
from matplotlib import cm
import gymnasium as gym
from kan import *

## Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


## initial dataset (always starts in an area where the norm of the state vector is less than 2)
train_input, train_label = generate_dictionary(n_samples=500, random_state=2024)
test_input, test_label = generate_dictionary(n_samples=500, random_state=2025)

dataset = {}
dtype = torch.get_default_dtype()
dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
dataset['train_label'] = torch.from_numpy(train_label[:,None]).type(dtype).to(device)
dataset['test_label'] = torch.from_numpy(test_label[:,None]).type(dtype).to(device)

# ## Define gym Inverted Pendulum env
env = InvertedPendulum(a=0.25, b=0.5, Kp=0.6, Kd=0.6, render_mode='human', mode='NN')

## Define proposed ground truth CBF
a = 0.25
b = 0.5
CBF = lambda x: 1 - ((x[0]**2)/a**2) - ((x[1]**2)/b**2) - ((x[0]*x[1])/(a*b))

env.set_CBF(CBF)

## Define alpha
gamma = 1
alpha = lambda h, u_ref: gamma*h
env.set_alpha(alpha)

train_data = np.double(dataset['train_input'])
train_labels = np.double(dataset['train_label'])
test_data = np.double(dataset['test_input'])
test_labels = np.double(dataset['test_label'])
    
# Parameters
params = {'batch_size': 50,
          'shuffle': True}

# Generators
training_set = Dataset_list(train_data, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset_list(test_data, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_set, **params)

## Create FCNN (equivalent is to have L hidden layers of width N)
model = FCNet(width=[2,16,16,1], mean=0, std=1, device=device, bn=False)
model.set_dataloaders(train_dataloader, test_dataloader)
model.set_environment(env)

# Initialize the optimizer.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# epochs = 20
# train_losses, test_losses = [], []
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_losses = model.train_model(model, optimizer, train_losses, env)
#     #test_losses = model.test(test_dataloader, model, loss_fn, test_losses)
# print("Training Done!")

model = model.double()
model.load_state_dict(torch.load('model_fc.pth'))

# torch.save(model.state_dict(), "model_fc.pth")
# print("Saved PyTorch Model State to model_xx.pth")
env.set_NN_CBF(model)

## simulation loop
num_steps = 300
NN_states = []
NN_actions = []
NN_actual_cbf_value = []
state = env.reset()
for step in range(num_steps):
    action = env.ref_controller(state)
    #action = np.random.uniform(-10.0, 10.0)
    next_state, _, _, done, _ = env.step(action)
    NN_states.append(state)
    NN_actions.append(action)
    NN_actual_cbf_value.append(env.h(state, action))
    state = next_state 
    if done:
        break
    #env.render()
    if step == num_steps - 1:
        print("Inverted Pendulum did not fall in {} steps".format(num_steps))

NN_states = np.array(NN_states)
NN_actions = np.array(NN_actions)
NN_actual_cbf_value = np.array(NN_actual_cbf_value)

## Plot safe set vs trajectory
plt.figure()
## Create 2D grid of states
x = np.linspace(-3.0, 3.0, 1000)
y = np.linspace(-3.0, 3.0, 1000)
X, Y = np.meshgrid(x, y)
states_grid = np.column_stack((X.flatten(), Y.flatten()))


# ## Calculate CBF values for each state in the grid
# cbf_values = []
# for pair in states_grid:
#     #h_temp = env.h(pair, 0)
#     h_temp = np.linalg.norm(pair)
#     cbf_values.append(h_temp)
# cbf_values = np.array(cbf_values)

# for pair in states_grid:
#     #h_temp = env.h(pair, 0)
#     h_temp = np.linalg.norm(pair)
#     cbf_values.append(h_temp)
# cbf_values = np.array(cbf_values)

# Plot circular region for all states where the norm of the state vector is less than 2
plt.scatter(states_grid[np.linalg.norm(states_grid, axis=1) < 2.0][:,0], states_grid[np.linalg.norm(states_grid, axis=1) < 2.0][:,1], marker='.', c='gold', label='Safe Set', alpha=0.3)


# Plot state trajectory
plt.plot(NN_states[:,0], NN_states[:,1], label='NN controller')

plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot{\theta}$')
plt.legend()
plt.title('Safe Set vs Trajectory')
plt.show()



## Calculate CBF values for each state in the grid
cbf_values = np.linalg.norm(states_grid, axis=1)

states_grid_torch = torch.tensor(states_grid, dtype=torch.float64).to(device)

#fig, ax = plt.subplots()
## 3d plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
nn_values = model(states_grid_torch, 1).detach().cpu().numpy()
## sort out all states that are not in the safe set
nn_states_grid = states_grid[np.where(nn_values > 0.0)]

kan_values = nn_values.reshape(X.shape)


#ax.contourf(X, Y, cbf_values.reshape(X.shape), cmap='Blues')
num_levels = 100  # Increase this number for finer color gradations
#surf = ax.contourf(X, Y, kan_values, cmap=cm.coolwarm, antialiased=False)
surf = ax.plot_surface(X, Y, kan_values, cmap=cm.coolwarm, antialiased=False)
#cbar_ticks = np.linspace(-5.0, 5.0, 21)  # 21 intervals from -5.0 to 5.0

#plt.scatter(kan_states_grid[:, 0], kan_states_grid[:, 1], marker='.', c='red', label='KAN Safe Set', alpha=0.3)
#ax.set_zlim(-0.1, 0.05)
fig.colorbar(surf)
ax.set_title('Safe Set vs Trajectory')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')
plt.show()





