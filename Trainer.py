from LoadData import *
from NeuralNetwork import *
from os import system

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

    def train(self, epochs: int = 50):
        for i in range(epochs):
            X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=self.NN.batch_size,
                                 classes=CLASSES, equilibrium=True)
            P = self.NN.forward(X)
            c = self.NN.compute_cost(P, Y)
            print(c, " | ",mae(P,Y))
            #print(P,Y)
            self.NN.backward(Y, lr=self.lr)
        print("\n")




# learning rate, dropout, momentum


if __name__ == "__main__":
    batch_size = 96
    learning_rate = 0.02
    learning_rate_stop = 0.1
    nn = NeuralNetwork([DenseLayer(89, 784, Tanh, batch_size=batch_size),
                        #DenseLayer(89, 89, ReLu, batch_size=batch_size),
                        DenseLayer(16,89, Id, batch_size=batch_size)],
                       CLASSES, cross_entropy_cost, Softmax, batch_size=batch_size)
    SGD = Trainer(nn, learning_rate, learning_rate_stop)
    SGD.train(epochs=500)
    SGD.train(epochs=1)
    #P = np.array([[0.1,0.3,0.2],
    #              [0.8,0.6,0.3],
    #              [0.1,0.1,0.5]])
    #Y = np.array([[1,0,0],
    #              [0,1,0],
    #              [0,0,1]])
    #print(percent_good(P,Y))


    #print(f"Empty main in : '{__file__[-10:]}'")

    #batch_size = 32
    #learning_rate = 5
    #nn = NeuralNetwork([DenseLayer(ReLu, 784, 89, batch_size=batch_size),
    #                    DenseLayer(ReLu, 89, 89, batch_size=batch_size),
    #                    DenseLayer(ReLu,89,16, batch_size=batch_size)],
    #                    CLASSES,cross_entropy_cost, batch_size=32)
    #X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=batch_size,
    #                        classes=CLASSES, equilibrium=False)
    #P = nn.forward(X)
    ##print(Y.shape, "\n", P.shape)
    #c = nn.compute_cost(P, Y)
    #print(c)
    #nn.backward(Y, lr=learning_rate)
    #P = nn.forward(X)
    #c = nn.compute_cost(P, Y)
    #print(c)
    #print(X.shape,"\n\n\n\n\n",Y.shape)