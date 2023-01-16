import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.GaussianVariational import GaussianVariational
from src.ScaleMixture import ScaleMixture

class BayesLinear(nn.Module):
    def __init__(
        self, 
        in_features : int, 
        out_features : int, 
        prior_pi:Optional[float] = 0.5, 
        prior_sigma1 : Optional[float] = 1.0, 
        prior_sigma2 : Optional[float] = 0.0025
        ):
        super().__init__()
        
        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)
        
        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)
        
        self.w_posterior = GaussianVariational(w_mu, w_rho)
        self.bias_posterior = GaussianVariational(bias_mu, bias_rho)
        
        self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.bias_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        
        self.kl_divergence = 0.0
        
    def forward(self, x : torch.Tensor):
        
        w = self.w_posterior.sample()
        b = self.bias_posterior.sample()
        
        # compute posterior and prior of the p(w) and p(w|D)
        w_log_prior = self.w_prior.log_prior(w)
        b_log_prior = self.bias_prior.log_prior(b)
        
        w_log_posterior = self.w_posterior.log_posterior()
        b_log_posterior = self.bias_posterior.log_posterior()
        
        total_log_prior = w_log_prior + b_log_prior
        total_log_posterior = w_log_posterior + b_log_posterior
        
        self.kl_divergence = self.compute_kld(total_log_prior, total_log_posterior)
        
        return F.linear(x, w, b)
    
    # simple due to the Monte-Carlo sampling which can replace the average over the posterior q(w|theta)
    def compute_kld(self, log_prior : torch.Tensor, log_posterior : torch.Tensor):
        return log_posterior - log_prior