from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader
import torch


def get_mnist(data_path: str = './data'):

    
    transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = CIFAR10(data_path, train=False, download=True, transform=transform)

    return testset

def prepare_dataset(num_partitions: int,
                     batch_size: int,
                       val_ratio: float = 0.1):
    
    testset = get_mnist()
    
    testloaders = DataLoader(testset, batch_size=128)

    return testloaders









