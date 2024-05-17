import numpy as np
import torch
from torch import nn
import torch.types
import torch.utils
import torch.utils.data
from utilities import load_mnist, plot_loss
import matplotlib.pyplot as plt
import time

class Autoencoder(nn.Module):
    def __init__(self, activation = nn.LeakyReLU, ae_type = 'DAE'):
        '''
        activatoin: the activation function to use in the hidden layers
        ae_type: one between denoising autoencoder (DAE) and contractive autoencoder (CAE)
        '''
        if ae_type != 'DAE' and ae_type != 'CAE':
            raise Exception('the type of autoecoder should be one between [\'DAE\',\'CAE\']')
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=700),
            activation(),
            nn.BatchNorm1d(700),
            nn.Linear(in_features=700, out_features=10),
            activation(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=10, out_features=700),
            activation(),
            nn.BatchNorm1d(700),
            nn.Linear(in_features=700, out_features=28*28),
            nn.Sigmoid()
        )
        self.loss_for_gradient_descent = self._lossMSE if ae_type == 'DAE' else self._lossStrange
        self.ae_type = ae_type
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _lossMSE(self, x, x_noisy, _) -> torch.Tensor:
        return nn.functional.mse_loss(x, x_noisy)
    
    def _lossStrange(self, x, x_noisy, lambdaParameter) -> torch.Tensor:
        mse = nn.functional.mse_loss(x, x_noisy)
        print(self.encoder)
        print("change here the loss computation")
        raise NotImplementedError()
        frobNorm = torch.norm(self.encoder[0].weight, p='fro')
        return mse + lambdaParameter*frobNorm
    

    
    def train(self, device, epochs=10, batch_size=128, lr=0.001, validation_split=0.1, regularize=0.5, weight_decay=1e-5) -> dict:
        '''
        device: the device in which to compute the stuff
        epoch: number of epoch
        batch_size: the batch size
        lr: learnign rate for Adam optimizer
        validaiton split: percentage of development that will be used as validation
        regularize: DAE -> amount of noise
                    CAE -> the lambda parameter of the loss function
        weight_decay: the lambda parameter fro the Adam optimizer
        '''
        # stup training
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        x_train, x_val, _ = load_mnist(validation_split)
        x_train = x_train.view(-1,28*28)
        x_val = x_val.view(-1, 28*28)
        x_val = x_val.to(device)
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
                if self.ae_type == 'DAE':
                    input_img = img + regularize * torch.randn(img.size(), device=device)
                    input_img = torch.clamp(input_img, 0., 1.)
                else:
                    input_img = img
                # ===================forward=====================
                output = self(input_img)
                loss = self.loss_for_gradient_descent(output, input_img, regularize)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================

            #if self.ae_type == 'DAE':
            img_val_noisy = x_val + regularize * torch.randn(x_val.size(), device=device)
            img_val_noisy = torch.clamp(img_val_noisy, 0., 1.)
            img_val_noisy = img_val_noisy.to(device)

            with torch.no_grad():
                output_val = self(img_val_noisy)
                loss_val = self.loss_for_gradient_descent(output_val, img_val_noisy, 0)
                history['val_loss'].append(loss_val.item())
                history['tr_loss'].append(loss.item())

            now = time.time()
            eta = (epochs - epoch -1)/((epoch + 1)/(now - begin))
            print(f'epoch [{epoch + 1}/{epochs}], loss:{loss.item():.4f}, val_loss:{loss_val.item():.4f}\teta = {int(eta//60):02d}m {int(eta%60):02d}s    ', end='\r')
            if epoch % (epochs // 5) == 0 or epoch == epochs - 1:
                print()
        return history
    
    def reconstruct(self, x:torch.Tensor) -> torch.Tensor:
        return self(self(self(x)))
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    device = torch.device('mps')
    dae = Autoencoder(
        activation=nn.Tanh,
        ae_type='DAE'
    )
    start = time.time()
    history = dae.train(device=device,
                        epochs=50, 
                        batch_size=10000, 
                        lr=0.005, 
                        validation_split=0.1, 
                        regularize=0.5,
                        weight_decay=0,
                        )
    end = time.time()
    print(f'Training time: {end - start:.2f}s')
    plot_loss(history)
    #exit()
    # Plotting the original and reconstructed images
    _, _, x_test = load_mnist()
    x_test = x_test.view(-1,28*28)
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