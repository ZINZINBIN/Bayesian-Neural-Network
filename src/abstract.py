import torch.nn as nn

class BayesianModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def compute_kld(self, *args):
        return NotImplementedError("BayesianModule::kld()")