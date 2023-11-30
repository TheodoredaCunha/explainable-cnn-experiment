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

loaders = data(batch_size = 128)
device = torch.device("cpu")

model = CNNModel()
summary(model, (1, 28, 28))
loss_func = nn.functional.nll_loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = [2, 4, 6, 10, 15]
for i in num_epochs:
    print(f'training with {i} epochs')
    train(i, model, loaders, optimizer, loss_func, device, print_progress=True)
    test(model, loaders, loss_func, device)
    explain(model, loaders, i, device)

    