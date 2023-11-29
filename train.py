from torch.autograd import Variable

def train(num_epochs, cnn, loaders, optimizer, loss_func, device, print_progress = False):
    cnn.train()
    for i, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = cnn(data)
        loss = loss_func(output.log(), target)
        loss.backward()
        optimizer.step()           
            
        if print_progress:
            if (i+1) % 2000 == 0:
                print ('Epoch [{}], Step [{}], Loss: {:.4f}' 
                    .format(num_epochs, i + 1, loss.item()))
        