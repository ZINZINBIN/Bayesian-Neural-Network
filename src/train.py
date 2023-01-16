from typing import Optional, List, Literal, Union
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.abstract import BayesianModule
from src.minibatch_weight import minibatch_weight
from sklearn.metrics import f1_score
import os

def train_per_epoch(
    train_loader : DataLoader, 
    model : BayesianModule,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):

    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0
    total_pred = np.array([])
    total_label = np.array([])
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        outputs = model(data)
        
        pi_weight = minibatch_weight(batch_idx, num_batches = len(train_loader))
        loss = model.elbo(
            inputs = data,
            targets = target,
            criterion = loss_fn,
            n_samples = 3,
            w_complexity = pi_weight
        )
    
        loss.backward()

        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()
        train_loss += loss.item()
        
        prob = torch.nn.functional.softmax(outputs, dim = 1)
        _, pred = torch.max(prob, 1)
        
        train_acc += pred.eq(target.view_as(pred)).sum().item()
        
        total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
        total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
        
        total_size += data.size()[0]
        

    if scheduler:
        scheduler.step()

    train_loss /= total_size
    train_acc /= total_size
    train_f1 = f1_score(total_label, total_pred, average = "macro")
    
    return train_loss, train_acc, train_f1


def valid_per_epoch(
    valid_loader : DataLoader, 
    model : BayesianModule,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0
    total_size = 0
    total_label = np.array([])
    total_pred = np.array([])

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)
            
            pi_weight = minibatch_weight(batch_idx, num_batches = len(valid_loader))
            loss = model.elbo(
                inputs = data,
                targets = target,
                criterion = loss_fn,
                n_samples = 3,
                w_complexity = pi_weight
            )
        
            valid_loss += loss.item()
            prob = torch.nn.functional.softmax(outputs, dim = 1)
            _, pred = torch.max(prob, 1)
            
            valid_acc += pred.eq(target.view_as(pred)).sum().item()
        
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
            
            total_size += data.size()[0]


    valid_loss /= total_size
    valid_acc /= total_size
    valid_f1 = f1_score(total_label, total_pred, average = "macro")

    return valid_loss, valid_acc, valid_f1

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : BayesianModule,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_dir : str = "./weights",
    tag : str = "model",
    max_norm_grad : Optional[float] = None,
    ):

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_acc = 0
    best_f1 = 0

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    save_best = os.path.join(save_dir, "{}_best.pt".format(tag))
    save_last = os.path.join(save_dir, "{}_last.pt".format(tag))
    
    print("save best : ", save_best)
    print("save_last : ", save_last)

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {} | train loss : {:.3f} | valid loss : {:.3f} | train f1 : {:.3f} | valid f1 : {:.3f} | train acc : {:.3f} | valid acc : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_f1, valid_f1, train_acc, valid_acc
                ))

        # save the best parameters
        if valid_acc > valid_acc:
            best_loss = valid_loss
            best_acc = valid_acc
            best_f1 = valid_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best)

        # save the last parameters
        torch.save(model.state_dict(), save_last)

    print("training process finished, best loss : {:.3f}, best f1 : {:.3f}, best acc : {:.3f}, best epoch : {}".format(
        best_loss, best_f1, best_acc, best_epoch
    ))
    
    return train_loss_list, valid_loss_list