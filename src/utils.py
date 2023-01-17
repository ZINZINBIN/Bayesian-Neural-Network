import torch
from torchvision import datasets
from torchvision import transforms

# normal transform
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

# transform for augmentation
augment_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)