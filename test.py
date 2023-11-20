import torch
from dataset import data
import shap

loaders = data(batch_size = 4)

def test(cnn, loaders, device):
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images.to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            
    print('Test Accuracy: %.2f' % accuracy)