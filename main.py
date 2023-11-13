import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()


