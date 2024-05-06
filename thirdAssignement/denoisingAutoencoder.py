import numpy as np
import torch
from torch import nn
from utilities import load_mnist, plot_loss
import matplotlib.pyplot as plt
import time

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.BatchNorm2d(8),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 2, 2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def lossMSE(self, x, x_noisy) -> torch.Tensor:
        return nn.functional.mse_loss(x, x_noisy)
    
    def lossStrange(self, x, x_noisy, lambdaParameter) -> torch.Tensor:
        mse = nn.functional.mse_loss(x, x_noisy)
        frobNorm = torch.norm(self.encoder[0].weight, p='fro')
        return mse + lambdaParameter*frobNorm
    

    
    def train(self, device, epochs=10, batch_size=128, lr=0.001, validation_split=0.1, noise_factor=0.5, weight_decay=1e-5, loss_func=lossMSE, noise=True) -> dict:
        # stup training
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        x_train, x_val, _ = load_mnist(validation_split)
        x_train = x_train.view(-1, 1, 28, 28)
        x_val = x_val.view(-1, 1, 28, 28)
        x_val = x_val.to(device)
        x_train = x_train.to(device)
        train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
        history = {
            'tr_loss': [],
            'val_loss': []
        }
        self.to(device)
        begin = time.time()
        # traininig loop
        for epoch in range(epochs):
            for data in train_loader:
                img = data.to(device)
                if noise:
                    img_noisy = img + noise_factor * torch.randn(img.size(), device=device)
                    img_noisy = torch.clamp(img_noisy, 0., 1.)
                else:
                    img_noisy = img
                # ===================forward=====================
                output = self(img_noisy)
                loss = loss_func(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================

            if noise:
                img_val_noisy = x_val + noise_factor * torch.randn(x_val.size(), device=device)
                img_val_noisy = torch.clamp(img_val_noisy, 0., 1.)
                img_val_noisy = img_val_noisy.to(device)
            else:
                img_val_noisy = x_val
            with torch.no_grad():
                output_val = self(img_val_noisy)
                loss_val = loss_func(output_val, x_val)
                history['val_loss'].append(loss_val.item())
                history['tr_loss'].append(loss.item())
            now = time.time()
            eta = (epochs - epoch -1)/((epoch + 1)/(now - begin))
            print(f'epoch [{epoch + 1}/{epochs}], loss:{loss.item():.4f}, val_loss:{loss_val.item():.4f}\teta = {int(eta//60):02d}m {int(eta%60):02d}s    ', end='\r')
            if epoch % (epochs // 5) == 0 or epoch == epochs - 1:
                print()
        return history
    
    def reconstruct(self, x:torch.Tensor) -> torch.Tensor:
        return self(x)
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    device = torch.device('mps')
    dae = DenoisingAutoencoder()
    start = time.time()
    history = dae.train(device=device,
                        epochs=60, 
                        batch_size=10000, 
                        lr=0.001, 
                        validation_split=0.1, 
                        noise_factor=0.5,# to change encoder kynd
                        #noise_factor=0,# to change encoder kynd
                        weight_decay=0,
                        loss_func=dae.lossMSE,# to change encoder kynd
                        #loss_func=lambda x, y,: dae.lossStrange(x,y,0.3),# to change encoder kynd
                        noise=True# to change encoder kynd
                        #noise=False# to change encoder kynd
                        )
    end = time.time()
    print(f'Training time: {end - start:.2f}s')
    plot_loss(history)
    #exit()
    # Plotting the original and reconstructed images
    _, _, x_test = load_mnist()
    x_test = x_test.view(-1, 1, 28, 28)
    x_test = x_test.to(device)
    x_test_noisy = x_test + 0.5 * torch.randn(x_test.size(), device=device)
    x_test_noisy = torch.clamp(x_test_noisy, 0., 1.)
    x_test_noisy = x_test_noisy.to(device)
    with torch.no_grad():
        x_reconstructed = dae(x_test_noisy)
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