import numpy as np
import torch
import torch.nn.functional as F
import cvxopt as cvx
from CBF_src import *
import matplotlib.pyplot as plt
import gymnasium as gym
from gym.envs.classic_control.pendulum import angle_normalize
from gymnasium import spaces
from kan import *
from typing import Optional

class CustomCar(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device='cpu', mode='CBF'):
        super(CustomCar, self).__init__()
        self.dt = 0.05
        self.max_speed = 2.0
        self.max_omega = 0.2
        self.max_accel = 0.5
        self.device = device
        self.mode = mode

        # Define action and obseevation space
        # Observation space: [x, y, theta, v]
        high = np.array([np.inf, np.inf, np.inf, self.max_speed], dtype=np.float32)
        low = np.array([-np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

        # Action space: [turning speed u1, forward acceleration u2]
        high = np.array([self.max_omega, self.max_accel], dtype=np.float32)
        low = np.array([-self.max_omega, -self.max_accel], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

    def reset(self):
        self.state = np.array([5.0, 25.0, 0.0, 2.0], dtype=np.float32)
        return self.state
    
    def step(self, action):
        x, y, theta, v = self.state
        u1, u2 = action
        dt = self.dt

        u1 = np.clip(u1, -self.max_omega, self.max_omega)
        u2 = np.clip(u2, -self.max_accel, self.max_accel)

        self.last_u = np.array([u1, u2], dtype=np.float32)
        costs = angle_normalize(theta)

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += u1 * dt
        v += u2 * dt

        self.state = np.array([x, y, theta, v], dtype=np.float32)
        
        return self._get_obs(), -costs, False, False, {}
    
    def _get_obs(self):
        x, y, theta, v = self.state
        return np.array([x, y, theta, v], dtype=np.float32)
    
    def render(self, mode='human'):
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
    
        ## Dynamics
    def f(self, x):

        x_pos, y_pos, theta, v = x[0], x[1], x[2], x[3]

        f = np.array([v*np.cos(theta), v*np.sin(theta), 0, 0])

        return f

    def g_dyn(self):

        g_dyn = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

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
    