import numpy as np
import torch
from torch import nn
from utilities import load_mnist, plot_loss
import matplotlib.pyplot as plt

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def lossMSE(self, x, x_noisy) -> torch.Tensor:
        criterion = nn.BCELoss()
        return criterion(x, x_noisy)
    
    def lossStrange(self, x, x_noisy) -> torch.Tensor:
        mse = nn.functional.mse_loss(x, x_noisy)
        jac = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        normJac = torch.norm(jac, p='fro')
        return mse + normJac

    
    def train(self, device, epochs=10, batch_size=128, lr=0.001, validation_split=0.1, noise_factor=0.5, weight_decay=1e-5, loss_func=lossMSE, noise=True) -> dict:
        # stup training
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        x_train, x_val, _ = load_mnist(validation_split)
        x_val = x_val.to(device)
        x_train = x_train.to(device)
        train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
        history = {
            'tr_loss': [],
            'val_loss': []
        }
        self.to(device)
        # traininig loop
        for epoch in range(epochs):
            for data in train_loader:
                img = data.to(device)
                img = img.view(img.size(0), -1)
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
                print(f'epoch [{epoch + 1}/{epochs}], loss:{loss.item():.4f}     ', end='\r')
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

            '''img_tr_noisy = x_train + noise_factor * torch.randn(x_train.size(), device=device)
            img_tr_noisy = torch.clamp(img_tr_noisy, 0., 1.)
            img_tr_noisy = img_tr_noisy.to(device)
            output_tr = self(img_tr_noisy)
            loss = self.lossMSE(output_tr, x_train)
            history['tr_loss'].append(loss.item())'''

            print(f'epoch [{epoch + 1}/{epochs}], loss:{loss.item():.4f}, val_loss:{loss_val.item():.4f}     ', end='\r')
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
    history = dae.train(device=device,
                        epochs=30, 
                        batch_size=300, 
                        lr=0.002, 
                        validation_split=0.1, 
                        noise_factor=0,
                        weight_decay=0,
                        loss_func=dae.lossStrange,
                        noise=False
                        )
    
    plot_loss(history)

    # Plotting the original and reconstructed images
    x_train, x_val, x_test = load_mnist()
    x_test = x_test.to(device)
    x_test_noisy = x_test + 0.5 * torch.randn(x_test.size(), device=device)
    x_test_noisy = torch.clamp(x_test_noisy, 0., 1.)
    x_test_reconstructed = dae.reconstruct(x_test_noisy)
    n = 10
    plt.figure(figsize=(30, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].cpu().detach().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].cpu().detach().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(x_test_reconstructed[i].cpu().detach().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()