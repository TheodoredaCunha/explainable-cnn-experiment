from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


def data(batch_size):
    trainset = MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())

    testset = MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

    loaders = {
        'train' : DataLoader(trainset, 
                            batch_size= batch_size, 
                            shuffle=True),
        
        'test'  : DataLoader(testset, 
                            batch_size= batch_size, 
                            shuffle=True)
    }

    return loaders