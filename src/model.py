import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
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
    

# using stochastic gradient of variational inference, the variance of the log-likelihood can not be vanished
# so we alternate the covariance term using local reparameterization trick
class BayesLinearLRT(nn.Module):
    def __init__(
        self, 
        in_features : int, 
        out_features : int, 
        std_prior : Optional[float] = 1.0
        ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_prior = std_prior
        
        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        self.w_mu = nn.Parameter(w_mu)
        
        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)
        self.w_rho = nn.Parameter(w_rho)
        
        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        self.bias_mu = nn.Parameter(bias_mu)
        
        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)
        self.bias_rho = nn.Parameter(bias_rho)
        
        self.eps_norm = torch.distributions.Normal(0,1)
        
        self.kl_divergence = 0.0
        
    def forward(self, x : torch.Tensor):
        
        # In this case, we use learnable parameter w_rho and bias_rho
        # these learnable parameters or also called as deterministic parts can make the backpropagation work
        w_std = torch.log(1 + torch.exp(self.w_rho))
        b_std = torch.log(1 + torch.exp(self.bias_rho))

        act_mu = F.linear(x, self.w_mu)
        act_std = torch.sqrt(F.linear(x.pow(2), w_std.pow(2)))

        # This is the stochastic part that the latent variable has randomness
        w_eps = self.eps_norm.sample(act_mu.size())
        bias_eps = self.eps_norm.sample(b_std.size())

        # w_out and b_out contains both deterministic part and stochastic part
        # If back-propagation works, then the deterministic part transfer the gradient from the child node to parent node
        w_out = act_mu + act_std * w_eps
        b_out = self.bias_mu + b_std * bias_eps

        w_kl = self.kld(
            mu_prior=0.0,
            std_prior=self.std_prior,
            mu_posterior=self.w_mu,
            std_posterior=w_std
        )

        bias_kl = self.kld(
            mu_prior=0.0,
            std_prior=0.1,
            mu_posterior=self.bias_mu,
            std_posterior=b_std
        )

        self.kl_divergence = w_kl + bias_kl

        return w_out + b_out
    
    def compute_kld(self, mu_prior : float, std_prior : float, mu_posterior : torch.Tensor, std_posterior : torch.Tensor):
        kl_divergence = 0.5 * (
            2 * torch.log(std_prior / std_posterior) - 1 + (std_posterior / std_prior).pow(2) + \
                ((mu_prior - mu_posterior) / std_prior).pow(2)
        ).sum()
        
        return kl_divergence
    
    
# Bayesian dropout
# reference : https://signing.tistory.com/108
class BayesDropout(nn.Module):
    def __init__(self, p : float = 0.5):
        super().__init__()
        assert p <= 1, "p must be smaller than 1"
        self.p = p
        
        self._multiplier = 1.0 / (1.0 - p)
        
    def forward(self, x : torch.Tensor):
        if not self.training:
            return x
        
        selected_ = torch.Tensor(x.size()).uniform_(0,1) > self.p
        selected_ = torch.autograd.Variable(selected_.type(torch.FloatTensor), requires_grad = False).to(x.device)
        res = torch.mul(selected_, x) * self._multiplier
        return res
    
    
# Bayesian Convolution Neural network
class BayesConv2d(nn.Module):
    def __init__(
        self,
        in_channels : int, 
        out_channels : int,
        kernel_size : Union[int, Tuple[int, int]],
        stride : int = 1,
        padding : int = 0,
        dilation : int = 1, 
        bias : bool = True,
        priors = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        
        self.groups = 1
        
        
        if priors is None:
            priors = {
                "prior_mu":0,
                "prior_sigma":0.1,
                "posterior_mu_initial":(0,0.1),
                "posterior_rho_initial":(-3,0.1)
            }
        
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']
        
        self.W_mu = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        self.W_rho = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((out_channels)))
            self.bias_rho = nn.Parameter(torch.empty((out_channels)))
        
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)
        
        self.reset_parameters()
        
        self.kl_divergence = 0
    
    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)
        
        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)
        
    def forward(self, x : torch.Tensor):
        
        if self.training:
            W_eps = torch.empty(self.W_mu.size()).normal_(0,1).to(x.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma
            
            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0,1).to(x.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
                
            else:
                bias = None
        
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None
        
        outputs = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        
        self.kl_divergence = self.compute_kld(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        
        if self.use_bias:
            self.kl_divergence += self.compute_kld(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        
        return outputs
    

    def compute_kld(self, mu_prior : float, std_prior : float, mu_posterior : torch.Tensor, std_posterior : torch.Tensor):
        kl_divergence = 0.5 * (
            2 * torch.log(std_prior / std_posterior) - 1 + (std_posterior / std_prior).pow(2) + \
                ((mu_prior - mu_posterior) / std_prior).pow(2)
        ).sum()
        return kl_divergence
    
    def compute_conv_output_dim(self, input_dim : int, kernel_size : int, stride : int, padding : int, dilation : int):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)