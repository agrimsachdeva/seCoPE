import math
import numpy as np
# import scipy as sc
from scipy import special

import torch
from torch import nn
from torch.nn import functional as F


class InvNet(nn.Module):
    
    def __init__(self, order):
        super().__init__()
        self.order = order
    
    def forward(self, A, x, alpha=1.):
        zs = [x]
        z = x
        for _ in range(self.order):
            # print("IN FOR LOOP FORWARD INVNET")

            # print("alpha", alpha)
            # print("z", z)
            # print("A", A)
            # print("A@z", A @ z)
            # print("alpha * (A @ z)", alpha * (A @ z))
            
            # print("alpha", alpha.shape)
            # print("z", z.shape)
            # print("A", A.shape)
            # print("A@z", (A @ z).shape)
            # print("alpha * (A @ z)", (alpha * (A @ z)).shape)

            # @ is matrix multiplication
            z = alpha * (A @ z)
            zs.append(z)
        return torch.stack(zs, 0).sum(0)

    
class ExpNet(nn.Module):
    
    def __init__(self, order):
        super().__init__()
        self.order = order
        self.coefs = self.compute_bessel_coefs(order)
    
    def compute_bessel_coefs(self, order):
        coefs = special.jv(np.arange(order+1), 0-1j) * (0+1j) ** np.arange(order+1)
        coefs = 2 * coefs.real
        coefs[0] /= 2
        return torch.from_numpy(coefs).float()
    
    def forward(self, A, x, alpha=1.):

        # print("IN forward expnet")
        # print("alpha_forward_expnet", alpha)
        # print("alpha_forward_expnet", alpha.shape)

        pp_state = x
        # alpha not changed
        # next line 
        # can alpha be 2d here
        p_state = alpha * (A @ x)

        zs = [pp_state, p_state]
        # pay attention
        for _ in range(self.order-1):
            new_state = 2 * alpha * (A @ p_state) - pp_state
            zs.append(new_state)
            pp_state, p_state = p_state, new_state
        return (torch.stack(zs, 0) * self.coefs.to(x.device).reshape(-1, 1, 1)).sum(0)

    
class ACGNN(nn.Module):
    
    def __init__(self, inv_order, exp_order, n_nodes, learnable_alpha=False):
        super().__init__()
        self.inv_net = InvNet(inv_order)
        self.exp_net = ExpNet(exp_order)
        self.n_nodes = n_nodes
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            # initilization of alpha 
            self.alpha = nn.Parameter(torch.ones(n_nodes) * 3)
            # change to 2d matrix
            # self.alpha = nn.Parameter(torch.ones(n_nodes, 32) * 3)
        else:
            # self.register_buffer('alpha', torch.ones(n_nodes))
            self.register_buffer('alpha', torch.ones(n_nodes, n_nodes))
    
    def forward(self, A, init_state, last_state, t):

            # novelty # 1
            # suggested by domain knowledge
            # if alpha is based on init state - good initialization based on severity
            # better then uniform initilization 
            # or sum over the neighboring information of last state (not sure) 
            # get from init state (last 3) features 
            # how does the alpha control the continuous update? the effect it is
            # look into expnet and invnet
            # look at pairwise propagation (instead of alpha, alpha1, alpha2, alpha3..)
            # influence of one node on each of it's neighbors
            # can we have a 2d matrix for alpha?
            # static 2d can be updated once we know what to calculate from the pairwise attention

        # print("IN ACGNN forward") 
        d = last_state.size(1)
        if self.learnable_alpha:

            # print(self.alpha)
            # print(self.alpha.shape)
            # print("self.alpha")

            alpha = torch.sigmoid(self.alpha)
        else:
            alpha = self.alpha

        # print("alpha_1", alpha)
        alpha = alpha.unsqueeze(1)
        # print("alpha_2", alpha)

        # e^{(A_I)t} x
        scale = math.ceil(t)

        z = torch.cat([init_state, last_state], 1) * math.exp(-t)

        for _ in range(scale):
            z = self.exp_net(A / scale, z, alpha)

        init_exp, last_exp = torch.split(z, d, 1)
        # (A-I)^{-1} (x - e^{(A_I)t} x)
        init_inv = self.inv_net(A, init_state - init_exp, alpha)
        return init_inv + last_exp
