import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.abstract import BayesianModule
from src.minibatch_weight import minibatch_weight
from sklearn.metrics import f1_score

def evaluate(
    test_loader : DataLoader, 
    model : BayesianModule,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    test_loss = 0
    test_acc = 0
    total_size = 0
    total_label = np.array([])
    total_pred = np.array([])

    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)
            
            pi_weight = minibatch_weight(batch_idx, num_batches = len(test_loader))
            loss = model.elbo(
                inputs = data,
                targets = target,
                criterion = loss_fn,
                n_samples = 3,
                w_complexity = pi_weight
            )
        
            test_loss += loss.item()
            prob = torch.nn.functional.softmax(outputs, dim = 1)
            _, pred = torch.max(prob, 1)
            
            test_acc += pred.eq(target.view_as(pred)).sum().item()
        
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
            
            total_size += data.size()[0]


    test_loss /= total_size
    test_acc /= total_size
    test_f1 = f1_score(total_label, total_pred, average = "macro")

    return test_loss, test_acc, test_f1