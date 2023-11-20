import shap
import torch
import numpy as np

def explain(cnn, loader, device):

    background, _ = next(iter(loader['test']))
    test_images = next(iter(loader['test']))
    e = shap.DeepExplainer(cnn.to(device), background.to(device))
    shap_values = e.shap_values(test_images.to(device))

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

