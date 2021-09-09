"""
@author: Heiner Stuke
"""

from pyro.nn import PyroSample, PyroParam, PyroModule
from torch.distributions import constraints
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


class BayesianLasso(PyroModule):
    def __init__(self, nullmodel):
        super().__init__()
        self.nullmodel = nullmodel

    def forward(self, X, y):
        if self.nullmodel:
            weights = torch.zeros(X.shape[1])
        else:
            alpha = pyro.sample('alpha',dist.Uniform(0., 10.))
            weights = pyro.sample('weights',dist.Laplace(0., alpha).expand([X.shape[1]]).to_event(1))
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        predy = torch.matmul(X,weights)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(predy, sigma), obs=y)
        return predy
