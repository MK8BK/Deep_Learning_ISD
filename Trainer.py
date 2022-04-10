from LoadData import *
from NeuralNetwork import *
from os import system



# learning rate, dropout, momentum


if __name__ == "__main__":
    batch_size = 48
    lr = 1
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
        nn.backward(Y)
        #print(lr)
        lr*=0.9999
    X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=batch_size,
                         classes=CLASSES, equilibrium=True)
    nn.forward(X)
    print(nn.backward(Y))

    #print(f"Empty main in : '{__file__[-10:]}'")