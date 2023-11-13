import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import CNNModel
import torch.optim as optim
from torch.autograd import Variable

model = CNNModel(num_classes = 10, num_layers = 2, first_out_channel = 16)

batch_size = 4
trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

loaders = {
    'train' : DataLoader(trainset, 
                         batch_size=100, 
                        shuffle=True),
    
    'test'  : DataLoader(testset, 
                        batch_size=100, 
                        shuffle=True)
}



loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 2
def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
               
        
        pass
    
    
    pass
train(num_epochs, model, loaders)
        