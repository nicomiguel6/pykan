import numpy as np
import torch
import torch.nn.functional as F
import cvxopt as cvx
import matplotlib.pyplot as plt
import gymnasium as gym
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from kan import *
from typing import Optional
from sklearn.utils import shuffle

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
    
    def reset(self):
        #self.state = np.array([np.random.uniform(0, 2*np.pi), np.random.uniform(-8.0, 8.0)])
        self.state = np.array([-0.1,0.5])
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
    
    def forward(self, x, sgn):
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

## Define KAN wrapper class
class kanNetwork(KAN):
    def __init__(self, environment, width, grid, k, device, dataset: Optional[np.ndarray] = None):
        super().__init__(width=width, grid=grid, k=k, device=device, base_fun='identity')
        self.device = device
        self.width = width
        self.environment = environment
        
        if dataset is not None:
            self.dataset = dataset

    def create_dataset(self, CBF, n_var):
        self.dataset = create_dataset(f=CBF, n_var=n_var, device = self.device)
        return self.dataset
    
    def results(self, dataset, opt, steps, lr):
        loss_fn = self.loss_function()
        results = self.fit(dataset=dataset, opt=opt, steps=steps, lr=lr, loss_fn=loss_fn)
        return results

    def loss_function(self):
        def compute_loss(prediction, label):

            ## Split into three individual loss functions: mean of prediction for initial states, mean of prediction for unsafe states, and meand of equality constraint for all states
            loss_safe = 0.
            loss_unsafe = 0.
            loss_desc = 0.

            ## iterate through unsafe states [want the predictions to be negative]
            num_unsafe = len(self.id_dict['train_id']['unsafe'])
            for iter in self.id_dict['train_id']['unsafe']:
                loss_unsafe += prediction[iter].item()
            
            ## iterate through safe states [want the predictions to be positive]
            num_safe = len(self.id_dict['train_id']['initial'])
            for iter in self.id_dict['train_id']['initial']:
                loss_safe += prediction[iter].item()
            
            ## iterate through all states
            for iter in range(len(prediction)):
                x = self.dataset['train_input'][self.train_id[iter]]
                dBdx = self.dx[iter].detach().numpy()
                LfB = dBdx @ self.environment.f(x)
                LgB = dBdx @ self.environment.g_dyn()
                ind_desc_loss = LfB + LgB*self.environment.ref_controller(x) + prediction[iter].item()
                loss_desc += ind_desc_loss
            
            ## Take mean and reLu
            loss_safe_mean = torch.relu(torch.tensor(-loss_safe/num_safe))
            loss_unsafe_mean = torch.relu(torch.tensor(loss_unsafe/num_unsafe))
            loss_desc_mean = torch.relu(torch.tensor(-loss_desc/len(prediction)))
            
            # # loop through all predictions, identify label
            # for iter, pred in enumerate(prediction):
            #     # extract input
            #     x = self.dataset['train_input'][self.train_id[iter]]
            #     #x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device).unsqueeze(0)
            #     x_tensor = x.to(self.device).clone().detach().requires_grad_(True)
            #     ## calculate descent loss
            #     # take gradient of model with respect to input
            #     pred_new = self.forward(x_tensor, singularity_avoiding=False, y_th=1000.)
            #     pred_new.requires_grad_()
            #     dBdx = torch.autograd.grad(pred_new, x_tensor, create_graph=True)[0]
            #     dBdx = dBdx.detach().numpy()

            #     LfB = np.array(dBdx @ self.environment.f(x)).item()
            #     LgB = np.array(dBdx @ self.environment.g_dyn()).item()
            #     ind_desc_loss = LfB + LgB*self.environment.ref_controller(x) + pred_new.detach().numpy()
            #     ## calculate the descent loss
            #     loss_desc.append(ind_desc_loss.item())

            #     if label[iter] == 0: # safe
            #         loss_safe.append(pred_new.detach().numpy())
            #     elif label[iter] == 1: # unsafe
            #         loss_unsafe.append(pred_new.detach().numpy()) 

        # # Convert lists to numpy arrays before converting to tensors
        #     loss_safe_np = np.array(loss_safe)
        #     loss_safe_tensor = torch.tensor(loss_safe_np, dtype=torch.float64)
        #     loss_safe_mean = torch.relu(-torch.mean(loss_safe_tensor))

        #     loss_unsafe_np = np.array(loss_unsafe)
        #     loss_unsafe_tensor = torch.tensor(loss_unsafe_np, dtype=torch.float64)
        #     loss_unsafe_mean = torch.relu(-torch.mean(loss_unsafe_tensor))

        #     loss_desc_np = np.array(loss_desc)
        #     loss_desc_tensor = torch.tensor(loss_desc_np, dtype=torch.float64)
        #     loss_desc_mean = torch.relu(-torch.mean(loss_desc_tensor))

            return loss_safe_mean + loss_unsafe_mean + loss_desc_mean
        return compute_loss
                
    def plot_results(self, results):
        train_losses = []
        test_losses = []

        train_losses += results['train_loss']
        test_losses += results['test_loss']
        
        plt.figure()
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.legend(['train', 'test'])
        plt.ylabel('RMSE')
        plt.xlabel('step')
        plt.yscale('log')
        plt.show()

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

    
## dataset generator
"""
Returns
-------
X : ndarray of shape (n_samples, 2)
    The input samples.

y : ndarray of shape (n_samples,)
    The integer labels (0 or 1) for class membership of each sample.
    0 - safe set
    1 - unsafe set


"""
def generate_dictionary(n_samples, random_state=None):

    generator = np.random.RandomState(random_state)

    x_theta = generator.uniform(-np.pi, np.pi, n_samples)
    x_thetadot = generator.uniform(-8.0, 8.0, n_samples)

    X = np.vstack((x_theta, x_thetadot)).T

    y = np.zeros(n_samples)

    data_dict = {'initial': [], 'unsafe': [], 'all': []}

    dataset = {}

    for iter, state in enumerate(X):
        theta = state[0]
        theta_dot = state[1]

        data_dict['all'].append(state)

        if np.linalg.norm(state) >= 2.5 and np.linalg.norm(state) <= 3:
            data_dict['unsafe'].append(state)
            y[iter] = 1
        elif np.linalg.norm(state) <= 2.0:
            data_dict['initial'].append(state)
    



    return data_dict, X, y



