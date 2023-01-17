import torch
import argparse
from torchvision import datasets
from src.model import BayesLinear, BayesDropout, BayesConv2d
from src.utils import transform, augment_transform
from src.variational_approximator import variational_approximator
from src.train import train

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="training bayesian neural network for mnist dataset")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "BNN", choices=["BNN", "BNN-CNN"])
    parser.add_argument("--save_dir", type = str, default = "./results")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 256)
    parser.add_argument("--num_epoch", type = int, default = 64)
    parser.add_argument("--verbose", type = int, default = 4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)

    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW")
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)

    args = vars(parser.parse_args())
    return args

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ =="__main__":
    
    args = parsing()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    
    # dataset
    train_set = datasets.MNIST(root = "./dataset", train = True, download=True, transform = transform)
    test_set = datasets.MNIST(root = "./dataset", train = False, download=True, transform = transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = args['batch_size'], shuffle = True, num_workers = args['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = args['batch_size'], shuffle = True, num_workers = args['num_workers'])
    
    @variational_approximator
    class BayesianNetwork(torch.nn.Module):
        def __init__(self, input_dim : int, output_dim : int):
            super().__init__()
            self.layer1 = BayesLinear(input_dim, 1024)
            self.dropout1 = BayesDropout(p = 0.5)
            self.layer2 = BayesLinear(1024, 1024 // 2)
            self.dropout2 = BayesDropout(p = 0.5)
            self.layer3 = BayesLinear(1024 // 2, output_dim)
            
        def forward(self, x : torch.Tensor):
            x= x.view(-1, 28 * 28)
            x = torch.nn.functional.relu(self.layer1(x))
            x = self.dropout1(x)
            x = torch.nn.functional.relu(self.layer2(x))
            x = self.dropout2(x)
            x = self.layer3(x)
            return x
        
    @variational_approximator
    class BayesianConvNetwork(torch.nn.Module):
        def __init__(self, output_dim : int):
            super().__init__()
            self.conv1 = BayesConv2d(1, 16, 3, 1, 0, 1, True, None)
            self.max_pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.conv2 = BayesConv2d(16, 32, 3, 1, 0, 1, True, None)
            self.max_pool2 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
            
            self.linear_dim = 800
            self.layer1 = BayesLinear(800, 1024)
            self.dropout1 = BayesDropout(p = 0.5)
            self.layer2 = BayesLinear(1024, 1024 // 2)
            self.dropout2 = BayesDropout(p = 0.5)
            self.layer3 = BayesLinear(1024 // 2, output_dim)
            
        def forward(self, x : torch.Tensor):
            
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.max_pool1(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.max_pool2(x)
            x = x.view(x.size()[0], -1)
            
            x = torch.nn.functional.relu(self.layer1(x))
            x = self.dropout1(x)
            x = torch.nn.functional.relu(self.layer2(x))
            x = self.dropout2(x)
            x = self.layer3(x)
            return x
    
    if args['tag'] == "BNN":
        model = BayesianNetwork(28 * 28, 10).to(device)      
    elif args['tag'] == "BNN-CNN":
        model = BayesianConvNetwork(10).to(device)  
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')
    
    train_loss, valid_loss = train(
        train_loader,
        test_loader,
        model,
        optimizer,
        None,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        "./weights",
        args['tag'],
        args['max_norm_grad']
    )