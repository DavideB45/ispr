from keras.datasets import mnist
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_mnist(validation_split:float=0.2) -> tuple:
    (x_train, _), (x_test, _) = mnist.load_data()
    # preprocess development set
    x_train = x_train.astype('float32') / 255.
    #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_train = x_train[:int(len(x_train) * (1 - validation_split))]
    x_val = x_train[int(len(x_train) * (1 - validation_split)):]
    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)
    # preprocess test set
    x_test = x_test.astype('float32') / 255.
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x_test = torch.from_numpy(x_test)
    return x_train, x_val, x_test

def plot_loss(history:dict):
    # Plotting the training and validation loss
    plt.plot(history['tr_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()