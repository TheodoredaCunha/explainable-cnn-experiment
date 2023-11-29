import torch
from dataset import data
import shap

loaders = data(batch_size = 4)

def test(cnn, loaders, loss_func, device):
    cnn.eval()
    test_loss = 0
    correct = 0
    length = len(loaders['test'].dataset)
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = cnn(data)
            test_loss += loss_func(output.log(), target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= length
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            length,
            100.0 * correct / length,
        )
    )