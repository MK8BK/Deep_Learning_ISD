from LoadData import *
from NeuralNetwork import *

#class Trainer:
#    def __init__(self, NN: NeuralN)

#learning rate, dropout, momentum


if __name__=="__main__":
    batch_size = 32
    learning_rate = 5
    nn = NeuralNetwork([DenseLayer(ReLu, 784, 89, batch_size=batch_size),
            DenseLayer(ReLu, 89, 89, batch_size=batch_size),
            OutputLayer(Softmax,89,16,
                batch_size=batch_size)],
            CLASSES,cross_entropy_cost, batch_size=32)
    X, Y = load_data_set("../EMNIST_DATA_SET/", batch_size=batch_size,
                            classes=CLASSES, equilibrium=False)
    P = nn.forward(X)
    #print(Y.shape, "\n", P.shape)
    c = nn.compute_cost(P, Y)
    print(c)
    nn.backward(Y, lr=learning_rate)
    P = nn.forward(X)
    c = nn.compute_cost(P, Y)
    print(c)
    #print(X.shape,"\n\n\n\n\n",Y.shape)
