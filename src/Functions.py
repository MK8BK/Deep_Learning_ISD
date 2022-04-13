import numpy as np
from typing import Callable
from src.LoadData import *

class Activation:
    def __init__(self, fn, dfn):
        self.fn = fn
        self.dfn = dfn

    def forward(self, x):
        return self.fn(x)

    def backward(self, x):
        return self.dfn(x)

# ReLu, tanh, sigmoid, weight evaluation, matrix operations, bias addition

def r(x):
    return np.maximum(0,x)
def rb(x):
    return 1.0*(x>0)
ReLu = Activation(r,rb)


def s(x):
    return 1.0/(1.0+np.exp(-x))
def sb(x):
    return s(x)*(1-s(x))
Sigmoid = Activation(s,sb)


def th(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def thb(x):
    return 1-th(x)**2
Tanh = Activation(th, thb)


#class Softmax():
#    def __init__(self):
#        pass
#    def forward(self, x, axis=0, eps=1.e-9):
#        self.p = np.exp(x)/np.sum(np.exp(x), axis=axis)
#        self.p = np.clip(self.p, eps, 1 - eps)
#        return self.p
#    def backward(self, y):
#        return self.p - y


class SoftmaxCrossEntropyLoss():
    def __init__(self, eps: float=1e-9, axis=0):
        self.eps = eps
        self.axis = axis
    def forward(self, x):
        self.p = np.exp(x)/np.sum(np.exp(x), axis=self.axis)
        self.p = np.clip(self.p, self.eps, 1 - self.eps)    
        #self.p = softmax(X, axis=0)
        return self.p
    def backward(self, y):
        self.y = y
        #self.P = np.clip(self.P, self.eps, 1 - self.eps)
        #a modifier, \ na pas sa place ici
        #loss = (-1.0 * y * np.log(self.p) - (1.0 - y) * np.log(1 - self.p))
        #loss = np.squeeze(np.nansum(loss))

        grad = (self.p - self.y)#/self.p.shape[1]
        #grad = ()
        #grad = (-(self.Y / self.P) + ((1 - self.Y) / (1 - self.P))) / self.P.shape[1]
        return grad#, loss


# def apply_momentum(past_gradients, current_gradient, momentum=0.9):
#    for i in range(len(past_gradients)):
#        past_gradients[i] *= momentum
#        current_gradient +=past_gradients[i]
#    return past_gradients, current_gradient#/(len(past_gradients)+1)


def compute_cost(p, y, eps=1.e-9):
    np.clip(p, eps, 1 - eps)
    loss = (-1.0 * y * np.log(p) - (1.0 - y) * np.log(1 - p))
    return np.squeeze(np.nansum(loss))



def predicted_labels(p):
    maxvals = np.argmax(p, axis=0)
    return [label_to_char(maxval) for maxval in maxvals]


def percent_good(p, y):
    y = predicted_labels(y)
    p = predicted_labels(p)
    missed = 0
    for observation, prediction in zip(y, p):
        if observation!=prediction:
            missed+=1
    accuracy = 100*(len(p)-missed)/len(p)
    return accuracy


# draft, useless for now



if __name__=="__main__":
    final_z = np.array([[0.91, 0.2, 0.2],
                        [0.9, 0.8, 0.8]])
    labels = np.array([[1., 0., 1.],
                       [0., 0., 0.],
                       [0., 1., 0.]])
    predictions = np.array([[0., 0., 0.],
                            [1., 1., 1.],
                            [0., 0., 0.]])
    grads = np.array([[-0.32282815, -0.32282815,  0.17717185],
                     [ 0.32282815 , 0.32282815 ,-0.17717185]])
    softmax = Softmax()
    if type(softmax)==Softmax:
        print("yes")
    #softmax = Softmax()
    #print(type(final_z))
    #print(softmax.forward(final_z))
    #print(percent_good(predictions, labels))
    #print(SCE.backward())
    #print(cross_entropy_cost(final_z, labels))
