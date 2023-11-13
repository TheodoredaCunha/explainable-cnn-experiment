import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

class CNNModel(nn.Module):
    def __init__(self, num_classes, num_layers, first_out_channels):
        super(CNNModel, self).__init__()
        self.conv_layers = []
        self.prev_channels = 1
        self.first_out_channels = first_out_channels
        for i in range(num_layers):
            seq = nn.Sequential(
                nn.Conv2d(
                    in_channels = self.prev_channels,              
                    out_channels = self.first_out_channels,            
                    kernel_size=5,              
                    stride=1,                   
                    padding=2,                  
                ),                              
                nn.ReLU(),                      
                nn.MaxPool2d(kernel_size=2), 
            )
            self.conv_layers.append(seq)
            self.prev_channels = self.first_out_channels
            self.first_out_channels *= 2

        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualizationv

