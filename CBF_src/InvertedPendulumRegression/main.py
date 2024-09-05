#!/usr/bin/python

import numpy as np
import torch
import cvxopt as cvx
from util import *
import matplotlib.pyplot as plt
import gymnasium as gym
from kan import *

## Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

## Define gym Inverted Pendulum env
env = InvertedPendulum(a=1, b=1, Kp=0.6, Kd=0.6, render_mode='human')

## Define CBF
a = 1
b = 1
CBF = lambda x: 1 - ((x[0]**2)/a**2) - ((x[1]**2)/b**2) - ((x[0]*x[1])/(a*b))

env.set_CBF(CBF)

## Define alpha
gamma = 1
alpha = lambda h, u_ref: gamma*h
env.set_alpha(alpha)

## test basic controller with CBF

## simulation parameters
num_steps = 300
CBF_states = []
CBF_actions = []
CBF_actual_cbf_value = []

## simulation loop
state = env.reset() # Always in CW+ reference frame
for step in range(num_steps):
    #action = env.ref_controller(state)
    #action = np.random.uniform(-10.0, 10.0)
    action = 0.0
    next_state, _, _, done, _ = env.step(action)
    CBF_states.append(state)
    CBF_actions.append(action)
    CBF_actual_cbf_value.append(env.h(state, action))
    state = next_state 
    if done:
        break
    #env.render()
    if step == num_steps - 1:
        print("Inverted Pendulum did not fall in {} steps".format(num_steps))

env.close()

# Convert lists to numpy arrays for easier manipulation
CBF_states = np.array(CBF_states)
CBF_actions = np.array(CBF_actions)
CBF_actual_cbf_value = np.array(CBF_actual_cbf_value)

## Initialize KAN
env = InvertedPendulum(a=1, b=1, Kp=0.6, Kd=0.6, render_mode='human', mode = 'KAN')
env.set_CBF(CBF)

# ## Define alpha
# gamma = 1
# alpha = lambda h, u_ref: gamma*h
env.set_alpha(alpha)

## Create dataset [this will generate training points, but does not show the model the actual CBF]
## Define KAN CBF
a = 1
b = 1
kan_CBF = lambda x: 1 - ((x[:,[0]]**2)/a**2) - ((x[:,[1]]**2)/b**2) - ((x[:,[0]]*x[:,[1]])/(a*b))

model_exists = True
kanModel = KAN(width=[2, [3,3], 1], grid=3, k=3, device=device)
kan_dataset = create_dataset(kan_CBF, n_var=2, device=device)

if model_exists: # load existing model
    kanModel = KAN.loadckpt('./model_saved/' + '0.9')
    results = None
else: #train new model
    results = kanModel.fit(dataset=kan_dataset, opt='LBFGS', steps=200, lr=1e-3)


# ## Symbolification
# lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','tan','abs']
# kanModel.auto_symbolic(lib=lib)
# formula = kanModel.symbolic_formula()[0][0]
# ex_round(formula, 4)
# print(formula)

## Assign KAN to the environment
env.set_KAN_CBF(kanModel)

## test KAN controller with CBF
## simulation parameters
num_steps = 300
KAN_states = []
KAN_actions = []
KAN_actual_cbf_value = []

## simulation loop
state = env.reset()
for step in range(num_steps):
    #action = env.ref_controller(state)
    # choose random action and convert to float
    #action = np.random.uniform(-10.0, 10.0)
    action = 0.0
    next_state, _, _, done, _ = env.step(action)
    KAN_states.append(state)
    KAN_actions.append(action)
    KAN_actual_cbf_value.append(env.h(state, action))
    state = next_state 
    if done:
        break
    #env.render()
    if step == num_steps - 1:
        print("Inverted Pendulum did not fall in {} steps".format(num_steps))

#kanModel.plot_results(results)
env.close()

# Convert lists to numpy arrays for easier manipulation
KAN_states = np.array(KAN_states)
KAN_actions = np.array(KAN_actions)
KAN_actual_cbf_value = np.array(KAN_actual_cbf_value)

# # Plot results (optional)
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(states[:, 0], label='Angle')
# plt.plot(states[:, 1], label='Angular Velocity')
# plt.legend()
# plt.title('States over Time')

# plt.subplot(2, 1, 2)
# plt.plot(actions, label='Actions')
# plt.legend()
# plt.title('Actions over Time')

# plt.show()

## Plot safe set vs trajectory
plt.figure()
## Create 2D grid of states
x = np.linspace(-0.5, 0.5, 500)
y = np.linspace(-0.7, 0.7, 500)
X, Y = np.meshgrid(x, y)
states_grid = np.column_stack((X.flatten(), Y.flatten()))

## Calculate CBF values for each state in the grid
cbf_values = []
for pair in states_grid:
    h_temp = env.h(pair, 0)
    cbf_values.append(h_temp)
cbf_values = np.array(cbf_values)

# # ## Calculate KAN values for each state in the grid
# # kan_values = []
# # for pair in states_grid:
# #     x_tensor = torch.tensor(pair, dtype=torch.float32, requires_grad=True).to(device).unsqueeze(0)
# #     kan_values.append(kanModel(x_tensor).detach().cpu().numpy())
# # kan_values = np.array(kan_values)


# plt.scatter(states_grid[:, 0], states_grid[:, 1], marker='.', c=np.where(cbf_values > 0, 'gold', 'white'), label='Safe Set' if np.any(cbf_values > 0) else None, alpha=0.3)
# #plt.scatter(states_grid[:, 0], states_grid[:, 1], marker='.', c=np.where(kan_values > 0, 'red', 'white'), label='KAN Safe Set' if np.any(cbf_values > 0) else None, alpha=0.1)
# ## Plot safe set vs trajectory
# plt.plot(CBF_states[:, 0], CBF_states[:, 1], color='blue', linestyle='dotted', label='CBF Trajectory')
# plt.plot(KAN_states[:, 0], KAN_states[:, 1], color='red', linestyle='dotted', label='KAN Trajectory')
# plt.xlim(-0.4, 0.4)
# plt.ylim(-0.8, 0.8)
# plt.legend()
# plt.title('Safe Set vs Trajectory')
# plt.show()

# ## Plot CBF values vs time
# plt.figure()
# plt.plot(CBF_actual_cbf_value, label='CBF')
# plt.plot(KAN_actual_cbf_value, label='KAN')
# plt.legend()
# plt.title('CBF Values over Time')
# plt.show()



# # # Plot results (optional)
# # plt.figure()
# # # Plot states
# # plt.subplot(2, 1, 1)
# # for i in range(len(states)):
# #     color = 'red' if actual_cbf_value[i] < 0 else 'blue'
# #     plt.scatter(i, states[i, 0], color=color, label='Angle' if i == 0 else "")
# #     plt.scatter(i, states[i, 1], color=color, label='Angular Velocity' if i == 0 else "")
# # plt.legend()
# # plt.title('States over Time')

# # # Plot actions
# # plt.subplot(2, 1, 2)
# # plt.plot(actions, label='Actions')
# # plt.legend()
# # plt.title('Actions over Time')

# # plt.show()

# ## Initialize InvertedPendulum for NN
env = InvertedPendulum(a=1, b=1, Kp=0.6, Kd=0.6, render_mode='human', mode = 'NN')
env.set_CBF(CBF)
env.set_alpha(alpha)

## Create dataset for FCNN
num_samples = 1000
train_data = np.zeros((num_samples,2))
train_labels = np.zeros((num_samples,1))

# Generate training points with labels
train_data = np.zeros((num_samples, 2))
train_labels = np.zeros((num_samples, 1))

# Generate testing points with labels
test_data = np.zeros((num_samples, 2))
test_labels = np.zeros((num_samples, 1))

train_data = np.double(kan_dataset['train_input'])
train_labels = np.double(kan_dataset['train_label'])
test_data = np.double(kan_dataset['test_input'])
test_labels = np.double(kan_dataset['test_label'])
    
# Parameters
params = {'batch_size': 50,
          'shuffle': True}

# Generators
training_set = Dataset_list(train_data, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset_list(test_data, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_set, **params)

## Create FCNN (equivalent is to have L hidden layers of width N)
model = FCNet(width=[2,4,1], mean=0, std=1, device=device, bn=False)
model.set_dataloaders(train_dataloader, test_dataloader)

# Initialize the optimizer.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

epochs = 10
train_losses, test_losses = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses = model.train_model(model, loss_fn, optimizer, train_losses)
    #test_losses = model.test(test_dataloader, model, loss_fn, test_losses)
print("Training Done!")

torch.save(model.state_dict(), "model_fc.pth")
print("Saved PyTorch Model State to model_xx.pth")
env.set_NN_CBF(model)

## simulation loop
num_steps = 300
NN_states = []
NN_actions = []
NN_actual_cbf_value = []
state = env.reset()
for step in range(num_steps):
    #action = env.ref_controller(state)
    #action = np.random.uniform(-10.0, 10.0)
    action = 0.0
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
x = np.linspace(-0.5, 0.5, 500)
y = np.linspace(-0.7, 0.7, 500)
X, Y = np.meshgrid(x, y)
states_grid = np.column_stack((X.flatten(), Y.flatten()))

## Calculate CBF values for each state in the grid
cbf_values = []
for pair in states_grid:
    h_temp = env.h(pair, 0)
    cbf_values.append(h_temp)
cbf_values = np.array(cbf_values)



# ## Calculate KAN values for each state in the grid
# kan_values = []
# for pair in states_grid:
#     x_tensor = torch.tensor(pair, dtype=torch.float32, requires_grad=True).to(device).unsqueeze(0)
#     kan_values.append(kanModel(x_tensor).detach().cpu().numpy())
# kan_values = np.array(kan_values)


plt.scatter(states_grid[:, 0], states_grid[:, 1], marker='.', c=np.where(cbf_values > 0, 'gold', 'white'), label='Safe Set' if np.any(cbf_values > 0) else None, alpha=0.3)
#plt.scatter(states_grid[:, 0], states_grid[:, 1], marker='.', c=np.where(kan_values > 0, 'red', 'white'), label='KAN Safe Set' if np.any(cbf_values > 0) else None, alpha=0.1)
## Plot safe set vs trajectory
plt.plot(CBF_states[:, 0], CBF_states[:, 1], color='red', linestyle='dotted', label='CBF Trajectory')
plt.plot(NN_states[:, 0], NN_states[:, 1], color='blue', linestyle='dotted', label='NN Trajectory')
plt.plot(KAN_states[:, 0], KAN_states[:, 1], color='green', linestyle='dotted', label='KAN Trajectory')
plt.xlim(-0.4, 0.4)
plt.ylim(-0.8, 0.8)
plt.legend()
plt.title('Safe Set vs Trajectory')
plt.show()

## Plot CBF values vs time
plt.figure()
plt.plot(CBF_actual_cbf_value, label='CBF')
plt.plot(NN_actual_cbf_value, label='NN')
plt.plot(KAN_actual_cbf_value, label='KAN')
plt.legend()
plt.title('CBF Values over Time')
plt.show()

# print(np.shape(train_losses))
# print(np.shape(test_losses))

# plt.plot(train_losses)
# plt.plot(test_losses)
# plt.legend(['train', 'test'])
# plt.ylabel('RMSE')
# plt.xlabel('step')
# plt.yscale('log')
# plt.show()




