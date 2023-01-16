import torch
import torch.nn as nn
import numpy as np

class GaussianVariational(nn.Module):
    def __init__(self, mu : torch.Tensor, rho : torch.Tensor):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        
        self.w = None
        self.sigma = None
        
        self.normal = torch.distributions.Normal(0,1)
        
    def sample(self):
        eps = self.normal.sample(self.mu.size()).to(self.mu.device)
        sig = torch.log(1 + torch.exp(self.rho)).to(self.rho.device)
        w = self.mu + sig * eps
        
        self.w = w
        self.sigma = sig
        
        return self.w
    
    def log_posterior(self):
        
        if self.w is None or self.sigma is None:
            raise ValueError("self.w must have a value...!")
        
        log_const = np.log(np.sqrt(2 * np.pi))

        log_exp = ((self.w - self.mu) ** 2) / (2*self.sigma ** 2)
        log_posterior = -log_const - torch.log(self.sigma) - log_exp
        
        return log_posterior.sum()