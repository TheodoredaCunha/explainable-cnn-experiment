from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


def data(batch_size):
    train_loader = DataLoader(
        MNIST(
            "mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        MNIST(
            "mnist_data", 
            train=False, 
            transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=batch_size,
        shuffle=True,
        )

    loaders = {'train' : train_loader, 'test'  : test_loader}

    return loaders