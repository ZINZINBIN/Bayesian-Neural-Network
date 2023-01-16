import functools as ft
import torch
import torch.nn as nn

class ScaleMixture(nn.Module):
    def __init__(self, pi : float, sigma1 : float, sigma2 : float):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        self.norm1 = torch.distributions.Normal(0, sigma1)
        self.norm2 = torch.distributions.Normal(0, sigma2)
    
    def log_prior(self, w: torch.Tensor):
        li_n1 = torch.exp(self.norm1.log_prob(w))
        li_n2 = torch.exp(self.norm2.log_prob(w))
        
        p_mixture = self.pi * li_n1 + (1-self.pi) * li_n2
        log_prob = torch.log(p_mixture).sum()
        
        return log_prob