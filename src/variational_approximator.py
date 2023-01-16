from typing import Any, Optional
import torch
import torch.nn as nn
from src.abstract import BayesianModule

# use this function as function decorator
# this is the key function for Bayesian Neural Network
def variational_approximator(model : nn.Module):
    
    def kl_divergence(self:nn.Module):
        kl = 0
        
        for module in self.modules():
            if isinstance(module, BayesianModule):
                kl += module.kl_divergence
        
        return kl
    
    setattr(model, 'kl_divergence', kl_divergence)
    
    def elbo(self : nn.Module, inputs : torch.Tensor, targets:torch.Tensor, criterion : Any, n_samples : int, w_complexity : Optional[float] = 1.0):
        loss = 0
        
        # MCMC sampling
        for sample in range(n_samples):
            outputs = self(inputs)
            loss += criterion(outputs, targets)
            loss += self.kl_divergence() * w_complexity
            
        return loss / n_samples
    

    setattr(model, 'elbo', elbo)
    
    return model