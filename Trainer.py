from LoadData import *
from NeuralNetwork import *


class Trainer:

    def __init__(self, NN: NeuralNetwork, lr: float, lr_stop: float,
                 lr_decay: str = 'exponential', momentum: float = 0.9, dropout: bool = False):
        #    assert(lr_decay=="exponential" or lr_decay=="linear"),\
        #        f"""learning rate decay not implemented: '{lr_decay}'"""
        self.NN = NN
        self.lr = lr
        self.lr_stop = lr_stop
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.dropout = dropout

    def train(self, epochs: int = 1000):
        for i in range(epochs):
            X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=self.NN.batch_size,
                                 classes=CLASSES, equilibrium=False)
            P = self.NN.forward(X)
            c = self.NN.compute_cost(P, Y)
            nn.backward(Y, lr=self.lr)
            print(c)
        print("\n")


# learning rate, dropout, momentum


if __name__ == "__main__":
    #batch_size = 32
    #learning_rate = 0.01
    #learning_rate_stop = 0.01
    #nn = NeuralNetwork([DenseLayer(ReLu, 784, 89, batch_size=batch_size),
    #                    DenseLayer(ReLu, 89, 89, batch_size=batch_size),
    #                    DenseLayer(ReLu, 89, 16, batch_size=batch_size)],
    #                    #OutputLayer(Softmax, 89, 16,
    #                    #            batch_size=batch_size)],
    #                   CLASSES, cross_entropy_cost, batch_size=32)
    #SGD = Trainer(nn, learning_rate, learning_rate_stop)
    #SGD.train()
    
    print(f"Empty main in : '{__file__[-10:]}'")

    batch_size = 32
    learning_rate = 5
    nn = NeuralNetwork([DenseLayer(ReLu, 784, 89, batch_size=batch_size),
            DenseLayer(ReLu, 89, 89, batch_size=batch_size),
            OutputLayer(Softmax,89,16,
                batch_size=batch_size)],
            CLASSES,cross_entropy_cost, batch_size=32)
    X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=batch_size,
                            classes=CLASSES, equilibrium=False)
    P = nn.forward(X)
    #print(Y.shape, "\n", P.shape)
    c = nn.compute_cost(P, Y)
    print(c)
    nn.backward(Y, lr=learning_rate)
    P = nn.forward(X)
    c = nn.compute_cost(P, Y)
    print(c)
    print(X.shape,"\n\n\n\n\n",Y.shape)