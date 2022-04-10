from LoadData import *
from NeuralNetwork import *

# learning rate, dropout, momentum
#class trainer()

#on consider que la couche d'entree e
#si on a 784 neurones dentrees, et 16 classes de sorties
    #il est preferable de mettre sqrt(784*16) neurones

if __name__ == "__main__":
    batch_size = 48
    lr = 0.1
    SCE = SoftmaxCrossEntropyLoss()
    layers = [DenseActivatedLayer(112, 784, Sigmoid), 
        DenseActivatedLayer(112, 112, Sigmoid),
        OuputLayer(16, 112, SCE)]

    nn = NeuralNetwork(layers, classes=CLASSES)
    #print(CLASSES)
    #comprends pas pk ca marche pas, tu peux checker ?
    for i in range(100):
        #X.shape = (784,32)
        X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=batch_size,
                             classes=CLASSES, equilibrium=True)
        nn.forward(X)
        print(nn.backward(Y))
        #print(lr)
        #lr*=lr
    #X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=batch_size,
    #                     classes=CLASSES, equilibrium=True)
    #nn.forward(X)
    #print(nn.backward(Y))

    #print(f"Empty main in : '{__file__[-10:]}'")