from src.Functions import *
from src.Layers import *
import numpy as np

class NeuralNetwork:
    """
        The NeuralNetwork wrapper class
    """
    def __init__(self, layers):
        """
            @param: layers : list[DenseLinearLayer]
                ordered, last must be OutputLayer
        """
        self.layers = layers
    def forward(self, X):
        """
            Forward pass of neural network
            @param: X : a 2d np.array of shape (784, batch_size)
            @return: res : model prediction: of shape (16, batch_size)
        """
        res = X
        for layer in self.layers:
            res = layer.forward(res)
        self.res = res
        return res
    def backward(self, Y, lr=0.01):
        """
            Backward pass of neural network
            @param: Y : a 2d np.array of shape (16, batch_size)
            @param: Optional lr=0.01 : Learning rate
            @action: perform backpropagation, use ____Layer.backward
                    successively
            @return: None
        """
        grad = Y
        #self.layers[-1].A_of_z
        grad = self.layers[-1].backward(Y, lr)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad, lr=lr)
        return None


#self.layers[-1].A_of_z
#percent = percent_good(self.res, Y)


if __name__=="__main__":
    print(f"Empty main in : '{__file__[-16:]}'")