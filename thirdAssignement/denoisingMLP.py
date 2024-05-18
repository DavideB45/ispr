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
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(8),

            nn.Flatten(),
            nn.Linear(8*14*14, 20),
            activation(),
            nn.BatchNorm1d(20)
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 8*14*14),
            activation(),
            nn.BatchNorm1d(8*14*14),
            nn.Unflatten(1, (8, 14, 14)),

            nn.ConvTranspose2d(8, 16, 5, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(16),
            
            nn.ConvTranspose2d(16, 2, 2, stride=2, padding=1),
            activation(),
            nn.BatchNorm2d(2),

            nn.ConvTranspose2d(2, 2, 1, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(2),

            nn.Conv2d(2, 1, 3, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.loss_for_gradient_descent = self._lossMSE if ae_type == 'DAE' else self._lossStrange
        self.ae_type = ae_type
        

    def forward(self, x, return_enc=False):
        x_enc = self.encoder(x)
        x = self.decoder(x_enc)
        if return_enc:
            return x, x_enc
        return x
    
    def _lossMSE(self, x, x_noisy, _, __) -> torch.Tensor:
        #print(x.shape)
        #print(x_noisy.shape)
        return nn.functional.mse_loss(x, x_noisy)
    
    def _lossStrange(self, x:torch.Tensor, x_noisy:torch.Tensor, x_enc:torch.Tensor, lambdaParameter:float) -> torch.Tensor:
        '''
        x: the output of the autoencoder
        x_noisy: the input of the autoencoder
        x_enc: the output of the encoder
        lambdaParameter: the lambda parameter of the loss function
        '''
        mse = nn.functional.mse_loss(x, x_noisy)
        x_enc.backward(torch.ones(x_enc.size()).to(device), retain_graph=True)
        reg_loss = torch.sqrt(torch.sum(torch.pow(x_noisy.grad, 2)))
        x_noisy.grad.data.zero_()
        return mse + lambdaParameter*reg_loss
    
    def accuracy(self, x:torch.Tensor, x_noisy:torch.Tensor) -> float:
        with torch.no_grad():
            x_reconstructed = self(x_noisy)
            return 1 - torch.mean(torch.abs(x - x_reconstructed)).item()
    
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
        x_train = x_train.view(-1, 1, 28, 28)
        x_val = x_val.view(-1, 1, 28, 28)
        x_val = x_val.to(device)
        train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
        history = {
            'tr_loss': [],
            'val_loss': [],
            'tr_acc': [],
            'val_acc': []
        }
        self.to(device)
        begin = time.time()
        # traininig loop
        for epoch in range(epochs):
            tr_acc = 0
            for data in train_loader:
                img:torch.Tensor = data.to(device)
                if self.ae_type == 'DAE':
                    input_img = img + regularize * torch.randn(img.size(), device=device)
                    input_img = torch.clamp(input_img, 0., 1.)
                else:
                    input_img = img
                    input_img.requires_grad_(True)
                    input_img.retain_grad()
                # ===================forward=====================
                output, enc = self(input_img, return_enc=True)
                loss = self.loss_for_gradient_descent(output, input_img, enc, regularize)
                tr_acc += self.accuracy(img, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================

            if self.ae_type == 'DAE':
                img_val_noisy = x_val + regularize * torch.randn(x_val.size(), device=device)
                img_val_noisy = torch.clamp(img_val_noisy, 0., 1.)
                img_val_noisy = img_val_noisy.to(device)
                with torch.no_grad():
                    output_val = self(img_val_noisy)
                    loss_val = self._lossMSE(output_val, img_val_noisy, _, _)
                    history['val_loss'].append(loss_val.item())
                    history['tr_loss'].append(loss.item())
                    history['tr_acc'].append(tr_acc/len(train_loader))
                    history['val_acc'].append(self.accuracy(x_val, x_val))
            if self.ae_type == 'CAE':
                img_val_noisy = x_val
                img_val_noisy.requires_grad_(True)
                img_val_noisy.retain_grad()
                output_val, enc_v = self(img_val_noisy, return_enc=True)
                loss_val = self._lossStrange(output_val, img_val_noisy, enc_v, regularize)
                optimizer.zero_grad()
                history['val_loss'].append(loss_val.item())
                history['tr_loss'].append(loss.item())
                history['tr_acc'].append(tr_acc/len(train_loader))
                history['val_acc'].append(self.accuracy(x_val, x_val))

            now = time.time()
            eta = (epochs - epoch -1)/((epoch + 1)/(now - begin))
            print(f'epoch [{epoch + 1}/{epochs}], loss:{loss.item():.4f}, \
val_loss:{loss_val.item():.4f}, \
tr_acc:{history["tr_acc"][-1]:.2f}, \
val_acc:{history["val_acc"][-1]:.2f}\
\teta = {int(eta//60):02d}m {int(eta%60):02d}s    ', end='\r')
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
    start = time.time()
    learning_rate = 0.005
    for ae_type in ['DAE', 'CAE']:
        for regularize in [0.1, 0.2, 0.4, 0.5, 0.8]:
            print(f'Training {ae_type} with regularize {regularize}')
            model = Autoencoder(
                activation=nn.LeakyReLU,
                ae_type=ae_type
            )
            history = model.train(device=device,
                                epochs=40, 
                                batch_size=10000, 
                                lr=learning_rate, 
                                validation_split=0.01, 
                                regularize=regularize,
                                weight_decay=0,
                                )
            #plot_loss(history)
            print('accuracy:', history['val_acc'][-1])
            print('@'+'='*80+'@')
            torch.save(model.state_dict(), f'./thirdAssignement/models/{ae_type}_{regularize}.pth')
    end = time.time()
    print(f'Training time: {end - start:.2f}s')