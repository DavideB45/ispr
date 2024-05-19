import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from thirdAssignement.denoisingMLP import Autoencoder
from thirdAssignement.utilities import load_mnist
from sklearn.metrics import f1_score

path = './thirdAssignement/models'
files = os.listdir(path)
print(files)
files.remove('train.txt')
device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

path = './thirdAssignement/models/DAE_0.5.pth'
#for file in files:
model = Autoencoder()
#model.load_state_dict(torch.load(path + '/' + file))
model.load_state_dict(torch.load(path))

model.to(device)

_, _, x_test = load_mnist()
x_test = x_test[10:30]
x_test = x_test.to(device)
x_test = x_test.view(-1,1,28,28)
x_test_noisy = 0.0*x_test + 1.0 * torch.randn(x_test.size(), device=device)
x_test_noisy = torch.clamp(x_test_noisy, 0., 1.)
with torch.no_grad():
    x_reconstructed = model(model(((model(model(x_test_noisy))))))

#    original = x_test.cpu().numpy().reshape(-1)
#    reconstructed = x_reconstructed.cpu().numpy().reshape(-1)
    # make boolean array
#    original = original > 0.5
#    reconstructed = reconstructed > 0.5
#    f_1 = f1_score(original, reconstructed, average='macro')
#    print(f'model: {file}\tf1 score: {f_1}')

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