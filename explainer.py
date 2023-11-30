import shap
import matplotlib.pyplot as plt
import numpy as np

def explain(cnn, loader, i, device):

    images, _ = next(iter(loader['test']))
    background = images[:100]
    test_images = images[100:103]
    e = shap.DeepExplainer(cnn, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    # plot the feature attributions
    shap.image_plot(shap_numpy, -test_numpy)
