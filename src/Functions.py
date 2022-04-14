import numpy as np
from typing import Callable
from src.LoadData import *

class Activation:
    """
        Abstract Activation function class
    """
    def __init__(self, fn, dfn):
        """
            @param: Callable self.fn: forward pass function
            @param: Callable self.dfn: backward pass function
                    (derivative of forward pass function)
        """
        #forward pass function
        self.fn = fn
        #backward pass function (derivative of forward pass)
        self.dfn = dfn

    def forward(self, x):
        """
            apply forward pass function to input matrix (2d np.array)
            @param: x : a 2d np.array, the results of the linear part of the layer
            @return: a 2d np.array (same shape as x), elementwise evaluation using fn
        """
        return self.fn(x)

    def backward(self, x):
        """
            apply backward pass function to input gradient matrix(2d np.array)
            @param: x : 2d np.array , gradient of loss wrt layer output
            @return: a 2d np.array (same shape as x), elementwise evaluation using dfn
                    ie: gradient of error wrt linear part of layer
        """
        return self.dfn(x)

# ReLu, tanh, sigmoid, weight evaluation, matrix operations, bias addition

def r(x):
    "ReLu forward pass"
    return np.maximum(0,x)
def rb(x):
    "ReLu backward pass"
    return 1.0*(x>0)
ReLu = Activation(r,rb)


def s(x):
    "Sigmoid forward pass"
    return 1.0/(1.0+np.exp(-x))
def sb(x):
    "Sigmoid backward pass"
    return s(x)*(1-s(x))
Sigmoid = Activation(s,sb)


def th(x):
    "Tanh forward pass"
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def thb(x):
    "Tanh backward pass"
    return 1-th(x)**2
Tanh = Activation(th, thb)

class SoftmaxCrossEntropyLoss():
    "Softmax Cross Entropy loss function wrapper class"
    def __init__(self, eps: float=1e-9, axis=0):
        """
            @param: optional eps = 1e-9: safety pre-log clipping precision
            @param: optional axis = 0 (columns) 
                                apply softmax by columns or rows of matrix

        """
        self.eps = eps
        self.axis = axis
    def forward(self, x):
        """
            apply softmax to matrix by axis (0 columns; 1 rows)
            @param: x : a 2d np.array, the results of the linear part of the layer
            @return: self.p (cached for backward use), axis-wise softmax
                        probability distribution per axis
        """
        self.p = np.exp(x)/np.sum(np.exp(x), axis=self.axis)
        self.p = np.clip(self.p, self.eps, 1 - self.eps)    
        return self.p
    def backward(self, y):
        """
            return gradient of loss with respect to prediction 
                for Softmax Cross Entropy Loss function
            @param: y : 2d np.array of labels
            @return: softmax_preds - labels 
                    (gradient of loss wrt linear output layer)
        """
        self.y = y
        grad = (self.p - self.y)#/self.p.shape[1]
        return grad


# def apply_momentum(past_gradients, current_gradient, momentum=0.9):
#    for i in range(len(past_gradients)):
#        past_gradients[i] *= momentum
#        current_gradient +=past_gradients[i]
#    return past_gradients, current_gradient#/(len(past_gradients)+1)


def compute_cost(p, y, eps=1.e-9):
    """
        Compute cost using Cross Entropy loss function
        @param: p : np.array of shape (16, batch_size),
                 column wise probabilities per sample
        @param: y : one hot encoded np.array of labels
        @return: loss (float)
    """
    np.clip(p, eps, 1 - eps)
    loss = (-1.0 * y * np.log(p) - (1.0 - y) * np.log(1 - p))
    return np.squeeze(np.nansum(loss))



def predicted_labels(p):
    """
        Converts a columnwise matrix of probabilities into prediction labels
        @param: p : np.array of shape (16, batch_size), sum(column)=1
        @return: a list characters, predictions based on max probability per image
    """
    maxvals = np.argmax(p, axis=0)
    return [label_to_char(maxval) for maxval in maxvals]


def percent_good(p, y):
    """
        Returns a percentage (range 0 to 100 float) of good predictions
        @param: p : np.array of shape (16, batch_size), sum(column)=1
        @param: y : np.array of shape (16, batch_size), one-hot-encoded
        @return: accuracy : a float between 0 and 100,
                the percentage of good predictions

    """
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
