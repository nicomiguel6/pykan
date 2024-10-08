import numpy as np
import torch
import torch.nn.functional as F
import cvxopt as cvx
from CBF_src import *
import matplotlib.pyplot as plt
import gymnasium as gym
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from kan import *
from typing import Optional

## Define wrapper class

class InvertedPendulum(PendulumEnv):
    def __init__(self, a, b, Kp, Kd, render_mode: Optional[str] = None, g=10.0, device='cpu', mode='CBF'):
        super().__init__(render_mode='human', g=g)
        self.a = a
        self.b = b
        self.Kp = Kp
        self.Kd = Kd
        self.device = device
        self.mode = mode

        # Define the new observation space
        high = np.array([2*np.pi, 8.0], dtype=np.float32)
        low = np.array([0, -8.0], dtype=np.float32)
        self.high = high
        self.low = low

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        mode = self.mode

        #self.state[0] = 2*pi - self.state[0]
        #self.state[1] = -self.state[1]

        ## check psi
        psi = self.psi(self.state, action)

        if psi > 0:
            # safe action
            return self.step_cw(np.array(action))
        else:
            # unsafe action
            u_act = self.CBF_QP(self.state, action)
            return self.step_cw(np.array(u_act))
        
    def step_cw(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (2 * g / (2 * l) * np.sin(th) + 1.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        # if self.render_mode == "human":
        #     self.render()
        return self._get_obs(), -costs, False, False, {}
    
    def safe_states(self, num_points):
        x = np.linspace(-1.0, 1.0, num_points)
        y = np.linspace(-1.0, 1.0, num_points)
        X, Y = np.meshgrid(x, y)
        states_grid = np.column_stack((X.flatten(), Y.flatten()))

        # only return states that are safe
        safe_states = []
        for state in states_grid:
            if self.h(state, 0) > 0 and np.linalg.norm(state) > 0.4:
                safe_states.append(state)
        self.safe_states = np.array(safe_states)
    
    def reset(self):
        # Choose from random safe state np array
        safe_states = self.safe_states
        self.state = safe_states[np.random.randint(0, len(safe_states))]

        #self.state = np.array([np.random.uniform(-2*np.pi, 2*np.pi), np.random.uniform(-8.0, 8.0)])
        #self.state = np.array([-0.1,0.5])
        return self.state
    
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot], dtype=np.float32)
    
    ## Dynamics
    def f(self, x):
        l = self.l
        g = self.g

        theta = x[0]
        theta_dot = x[1]

        f = np.array([theta_dot, 2*g/(2*l)*np.sin(theta)])

        return f

    def g_dyn(self):
        m = self.m
        l = self.l

        g_dyn = np.array([0, 1/(m*l**2)])

        return g_dyn
    
    ## High level controller
    def ref_controller(self, x):
        u_ref = (self.m*self.l**2)*(-(self.g/self.l)*np.sin(x[0]) - self.Kp*x[0] - self.Kd*x[1])
        return u_ref
    
    def alpha(self, h, u_ref):
        # h = self.h(x, u_ref)
        alpha_fun = self.get_alpha()
        alpha_val = alpha_fun(h, u_ref)

        return alpha_val
    
    ## CBF methods
    def h(self, x, u_ref):
        CBF = self.get_CBF()

        return CBF(x)
    
    def dhdx(self, x, u_ref):
        theta = x[0]
        theta_dot = x[1]

        dhdx = np.array([(-2 / self.a**2) * theta - (theta_dot / (self.a * self.b)), 
                         (-2 / self.b**2) * theta_dot - (theta / (self.a * self.b)) ])

        return dhdx
    
    def psi(self, x, u_ref):
        mode = self.mode

        if mode == 'CBF':
            h = self.h(x, u_ref)
            Lfh = self.dhdx(x, u_ref) @ self.f(x)
            Lgh = self.dhdx(x, u_ref) @ self.g_dyn()
        elif mode == 'NN':
            x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True).to(self.device).unsqueeze(0)
            nn_model = self.get_NN_CBF()
            h = nn_model(x_tensor, 1)
            dhdx = torch.autograd.grad(h, x_tensor, create_graph=True)[0]
            dhdx = dhdx.detach().numpy()

            h = h.detach().numpy()
            Lfh = np.array(dhdx @ self.f(x))
            Lgh = np.array(dhdx @ self.g_dyn())
        elif mode == 'KAN':
            x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device).unsqueeze(0)
            kan_model = self.get_KAN_CBF()
            #h = self.predict(x_tensor, kan_model)
            h = kan_model(x_tensor)
            # Compute the gradient of h with respect to x
            dhdx = torch.autograd.grad(h, x_tensor, create_graph=True)[0]
            dhdx = dhdx.detach().numpy()

            h = h.detach().numpy()
            Lfh = np.array(dhdx @ self.f(x))
            Lgh = np.array(dhdx @ self.g_dyn())

        alpha = np.array(self.alpha(h, u_ref)).item()

        psi = Lfh + Lgh*np.array(u_ref) + alpha

        return psi.item()
    
    def CBF_QP(self, x, u_ref):
        mode = self.mode

        cvx.solvers.options['show_progress'] = False

        ## Calculate CBF and lie derivatives [depends on mode]
        if mode == 'CBF':
            h = self.h(x, u_ref)
            Lfh = self.dhdx(x, u_ref) @ self.f(x)
            Lgh = self.dhdx(x, u_ref) @ self.g_dyn()
            alpha = self.alpha(h, u_ref)
        elif mode == 'NN':
            x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True).to(self.device).unsqueeze(0)
            nn_model = self.get_NN_CBF()
            h = nn_model(x_tensor, 1)
            dhdx = torch.autograd.grad(h, x_tensor, create_graph=True)[0]
            dhdx = dhdx.detach().numpy()

            h = h.detach().numpy()
            Lfh = np.array(dhdx @ self.f(x))
            Lgh = np.array(dhdx @ self.g_dyn())

            alpha = self.alpha(h, u_ref)
            Lfh = Lfh.item()
            Lgh = Lgh.item()
            alpha = alpha.item()
        elif mode == 'KAN':
            x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device).unsqueeze(0)
            kan_model = self.get_KAN_CBF()
            #h = self.predict(x_tensor, kan_model)
            h = kan_model(x_tensor)
            # Compute the gradient of h with respect to x
            dhdx = torch.autograd.grad(h, x_tensor, create_graph=True)[0]
            dhdx = dhdx.detach().numpy()

            h = h.detach().numpy()
            Lfh = dhdx @ self.f(x)
            Lgh = dhdx @ self.g_dyn()

            alpha = self.alpha(h, u_ref)
            Lgh = Lgh[0]
            Lfh = Lfh[0]
            alpha = alpha.item()

        
        psi = self.psi(x, u_ref)

        ## Inequality constraints
        G = cvx.matrix([-Lgh])
        h_qp = cvx.matrix([Lfh - alpha])

        ## Equality constraints
        A = cvx.matrix([0.0])
        b = cvx.matrix([0.0])

        # Cost matrices
        Q = cvx.matrix([1.0])
        p = cvx.matrix((-u_ref))

        sol = cvx.solvers.qp(Q, p, G, h_qp, None, None)

        u_act = sol['x'][0]

        return u_act
    
    def set_CBF(self, CBF):
        self.CBF = CBF

    def get_CBF(self):
        return self.CBF

    ## learned NN CBF (NN model)
    def set_NN_CBF(self, nn_model):
        self.nn_cbf = nn_model

    def get_NN_CBF(self):
        return self.nn_cbf
    
    ## learned KAN CBF (KAN model)
    def set_KAN_CBF(self, kan_model):
        self.kan_cbf = kan_model
    
    def get_KAN_CBF(self):
        return self.kan_cbf
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def get_alpha(self):
        return self.alpha
    
    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode
    
    def predict(self, x, model):
        # convert to pytorch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(0)
        # prepare model
        model.to(self.device)
        model.eval()
        # eval
        prediction = model(x_tensor)

        return prediction

class FCNet(nn.Module):
    def __init__(self, width, mean, std, device, bn):
        super().__init__()
        self.width = width
        self.mean = mean
        self.std = std
        self.device = device
        self.bn = bn

        # Normal BN/FC layers.
        if bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(width[i+1]) for i in range(len(width)-2)])

        self.fc_layers = nn.ModuleList([nn.Linear(width[i], width[i+1]).double() for i in range(len(width)-1)])
    
    def forward(self, x, sgn=None):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        
        for i in range(len(self.fc_layers)-1):
            x = F.relu(self.fc_layers[i](x))
            if self.bn:
                x = self.bn_layers[i](x)
        
        x = self.fc_layers[-1](x)  # No activation function here for regression
        
        return x
    
    def set_dataloaders(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_dataloaders(self):
        return self.train_dataloader, self.test_dataloader
    
    def train_model(self, model, loss_fn, optimizer, losses, test_losses=[]):
        
        dataloader, test_dataloader = self.get_dataloaders()
        
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Compute prediction error
            pred = model(X, 1)
            loss = loss_fn(pred, y)
            losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # Test after every batch
            # test_losses = test(test_dataloader, model, loss_fn, test_losses)

            if batch % 25 == 0:  #25
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return losses

    def test(self, model, loss_fn, losses):
        
        _, dataloader = self.get_dataloaders()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X, sgn=None)
                loss = loss_fn(pred, y)
                test_loss += loss.item()
        test_loss /= num_batches
        losses.append(test_loss)
        print(f"Test avg loss: {test_loss:>8f} \n")
        return losses

# ## Define KAN external class
# class kanNetwork():
#     def __init__(self, load_path=None, device='cpu'):
#         self.device = device
#         self.load_path = load_path

#     def create_model(self):
#         model = KAN()

    
#     def plot_results(self, results):
#         train_losses = []
#         test_losses = []

#         train_losses += results['train_loss']
#         test_losses += results['test_loss']
        
#         plt.figure()
#         plt.plot(train_losses)
#         plt.plot(test_losses)
#         plt.legend(['train', 'test'])
#         plt.ylabel('RMSE')
#         plt.xlabel('step')
#         plt.yscale('log')
#         plt.show()

class Dataset(torch.utils.data.Dataset):
  #'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
    #'Initialization'
    self.labels = labels
    self.list_IDs = list_IDs

  def __len__(self):
    #'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    #'Generates one sample of data'
    # Select sample
    ID = self.list_IDs[index]

    # Load data and get label
    X = torch.load('data/' + ID + '.pt')
    y = self.labels[ID]

    return X, y

class Dataset_list(torch.utils.data.Dataset):
  #'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, mode='train'):
    #'Initialization'
    self.labels = labels
    self.features = features
    self.mode = mode

  def __len__(self):
    #'Denotes the total number of samples'
    return len(self.features)

  def __getitem__(self, index):
    #'Generates one sample of data'
    
    if self.mode == 'test':
        index = 0

    # Load data and get label
    X = self.features[index]
    y = self.labels[index]

    return X, y
  

def run_simulation(env):
    num_steps = 300

    exit_bool = False
    exit_step = 0
    exit_state = np.zeros((2,1))
    state = env.reset()
    for step in range(num_steps):
        action = env.ref_controller(state)
        # choose random action and convert to float
        #action = np.random.uniform(-10.0, 10.0)
        #action = 0.0
        next_state, _, _, done, _ = env.step(action)
        actual_CBF_value = env.h(state, action)
        state = next_state 
        #print('Step: {}, State: {}, Action: {}, CBF: {}'.format(step, state, action, actual_CBF_value))
        if actual_CBF_value < -1e-3:
            exit_bool, exit_step, exit_state = True, step, state
            ## exit loop
            break
        #env.render()
        if step == num_steps - 1:
            #print("Inverted Pendulum did not fall in {} steps".format(num_steps))
            exit_bool, exit_step, exit_state = False, step, state
    
    return exit_bool, exit_step, exit_state

    

