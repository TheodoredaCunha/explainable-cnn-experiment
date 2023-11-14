import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, num_classes, num_layers, first_out_channels):
        super(CNNModel, self).__init__()
        list_of_seq = []
        self.prev_channels = 1
        self.first_out_channels = first_out_channels
        for i in range(num_layers):
            seq = nn.Sequential(
                nn.Conv2d(
                    in_channels = self.prev_channels,              
                    out_channels = self.first_out_channels,            
                    kernel_size=5,              
                    stride=1,                   
                    padding='same',                  
                ),                              
                nn.ReLU(),                      
                nn.MaxPool2d(kernel_size=2), 
            )
            list_of_seq.append(seq)
            self.prev_channels = self.first_out_channels
            self.first_out_channels *= 2

        self.conv_layers = nn.ModuleList(list_of_seq)

        # fully connected layer, output 10 classes
        self.out = nn.Linear(self.prev_channels * 28//(2 ** num_layers) * 28//(2 ** num_layers), num_classes)

    def forward(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
        
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

