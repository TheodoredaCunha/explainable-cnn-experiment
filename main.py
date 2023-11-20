import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CNNModel
import torch.optim as optim
from torchsummary import summary
from train import train 
from test import test
from dataset import data
from explainer import explain

loaders = data(batch_size = 4)
device = torch.device("cpu")
for i in range(1, 2):
    print(f"Current number of layers: {i}")
    model = CNNModel(num_classes = 10, num_layers = i, first_out_channels = 16)
    summary(model, (1, 28, 28))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 2

    train(num_epochs, model, loaders, optimizer, loss_func, device, print_progress=False)
    test(model, loaders, device)
    explain(model, loaders, device)

    