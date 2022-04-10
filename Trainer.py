from LoadData import *
from NeuralNetwork import *
from os import system



# learning rate, dropout, momentum


if __name__ == "__main__":
    batch_size = 160
    lr = 0.01
    SCE = SoftmaxCrossEntropyLoss()
    layers = [DenseActivatedLayer(112, 784, ReLu), 
        DenseActivatedLayer(112, 112, ReLu),
        OuputLayer(16, 112, SCE)]

    nn = NeuralNetwork(layers, classes=CLASSES)
    #comprends pas pk ca marche pas, tu peux checker ?
    for i in range(100):
        X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=batch_size,
                             classes=CLASSES, equilibrium=True)
        nn.forward(X)
        print(nn.backward(Y))

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