from LoadData import *
from NeuralNetwork import *
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pickle

# learning rate, dropout, momentum
#class trainer()
    
#on consider que la couche d'entree e
#si on a 784 neurones dentrees, et 16 classes de sorties
    #il est preferable de mettre sqrt(784*16) neurones

if __name__ == "__main__":
    accuracies = []
    batch_size = 32
    lr = 1
    epochs = 1200
    SCE = SoftmaxCrossEntropyLoss()
    layers = [DenseActivatedLayer(112, 784, ReLu), 
        #DenseActivatedLayer(112, 112, ReLu),
        OuputLayer(16, 112, SCE)]

    nn = NeuralNetwork(layers, classes=CLASSES)
    for i in range(epochs):
        #X.shape = (784,32)
        X, Y = load_data_set("./EMNIST_DATA_SET/", batch_size=batch_size,
                             classes=CLASSES, equilibrium=True)
        nn.forward(X)
        cost, percent = nn.backward(Y)
        #if i%30==0:
#            print(f"cost: {cost}  | accuracy: {percent}")
        #    print(f"Iteration: {i:3} | Accuracy: {percent}") 
        accuracies.append(percent)
    #print(accuracies)
    plt.scatter(list(range(epochs)), accuracies, s=1)
    ytickss = [str(i)+" %" for i in range(0,101,10)]
    plt.yticks(list(range(0,101,10)), ytickss)
    plt.plot([0,1201],[75,75], color="red")
    plt.show()
    with open('nn.pkl', 'wb') as outp:
        pickle.dump(nn, outp, pickle.HIGHEST_PROTOCOL)
    #print(f"Empty main in : '{__file__[-10:]}'")