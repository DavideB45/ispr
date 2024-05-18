import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from thirdAssignement.denoisingMLP import Autoencoder
from thirdAssignement.utilities import load_mnist

path = './thirdAssignement/model'
files = os.listdir(path)
print(files)
exit()

model = Autoencoder()
model.load_state_dict(torch.load(path))

_, _, x_test = load_mnist()
x_test = x_test[10:30]
x_test = x_test.view(-1,1,28,28)
x_test_noisy = x_test + 0.5 * torch.randn(x_test.size())
x_test_noisy = torch.clamp(x_test_noisy, 0., 1.)
with torch.no_grad():
    x_reconstructed = model(x_test_noisy)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(n*3, 4))

for i in range(n):
    # display noisy
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].cpu().numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_reconstructed[i].cpu().numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display original
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(x_test[i].cpu().numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()