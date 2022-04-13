from src.Functions import *
from src.Layers import *
import numpy as np

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers
    def forward(self, X):
        res = X
        for layer in self.layers:
            res = layer.forward(res)
        self.res = res
        return res
    def backward(self, Y, lr=0.01):
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