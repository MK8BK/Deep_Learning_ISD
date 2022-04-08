import pickle
from Functions import *
from Layers import *

#have'nt decided how constructor should work (list of int, DanseLayer, or else)
class NeuralNetwork:

    def __init__(self, layers: [DenseLayer],
                        classes: list, C: Callable, L: LossFunction,
                        batch_size: int=32):
        self.layers = layers
        self.W = {}
        self.b = {}
        self.batch_size = batch_size
        self.classes = classes
        self.C = C
        self.L = L

    def forward(self, X: np.array):
        assert(X.shape[1] == self.batch_size)
        res = X
        for layer in self.layers:
            res = layer.forward(res)
        return res

    def backward(self, Y: np.array, lr: float=0.01):
        grad = self.L.derive(self.layers[-1].A_of_z, Y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr=lr)
        return None

    def compute_cost(self, predictions: np.array, labels: np.array)->float:
        assert(labels.shape == predictions.shape),\
            f"""Incompatible shapes: predictions: {predictions.shape} 
                                     labels:      {labels.shape}"""
        return self.C(predictions, labels)

if __name__=="__main__":
    print(f"Empty main in : '{__file__[-16:]}'")