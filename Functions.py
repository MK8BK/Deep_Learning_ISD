import numpy as np
from typing import Callable


def derivative(f: Callable, epsilon: float = 1.0e-6, domain: np.ndarray=np.linspace(-10,10,200)):
    #print(domain, len(list(domain)))
    lst = [(f(x+epsilon)-(f(x-epsilon)))/(2*epsilon) for x in list(domain)]
    return np.array(lst)


def chain_derivative(functions: list[Callable], epsilon: float = 1.0e-6, domaine: np.ndarray=np.linspace(-10,10,200)):
    product = np.ones(len(domaine))
    inner = domaine.copy()
    for f in reversed(functions):
        product *= derivative(f,domain=inner)
        inner = f(inner)
    return list(product)


class Activation:
    def __init__(self, fn, dfn):
        self.fn = fn
        self.dfn = dfn

    def forward(self, X):
        return self.fn(X)

    def backward(self, X):
        return self.dfn(X)


#ReLu, tanh, sigmoid, weight evaluation, matrix operations, bias addition
ReLu = Activation(lambda x: np.maximum(0,x),
    lambda x: 1.0*(x>0))


Sigmoid = Activation(lambda x: 1.0/(1.0+np.exp(-x)),
    lambda x:(1.0/(1.0+np.exp(-x)))*(1- 1.0/(1.0+np.exp(-x))))


Tanh = Activation(lambda x:(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)),
                lambda x:1-((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2)

#Id = Activation(lambda x: x, lambda x: 1)


def softmax(X, axis=0):
    #X = X - np.max(X, axis=axis)
    return np.exp(X)/np.sum(np.exp(X), axis=axis)

class Loss:
    def __init__(self):
        pass
    def forward(self, X):
        pass
    def compute_cost(self, Y):
        pass
    def backward(self):
        pass

class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, eps: float=1e-9):
        super().__init__()
        self.eps = eps
    def forward(self, X):
        self.P = softmax(X, axis=0)
        return self.P
    def compute_cost(self, Y):
        self.Y = Y
        self.P = np.clip(self.P, self.eps, 1 - self.eps)
        loss = (-1.0 * Y * np.log(self.P) - \
            (1.0 - Y) * np.log(1 - self.P))
        return np.sum(loss)
    def backward(self):
        grad = self.P - self.Y
        return grad

# n = len(self.real)
#        return (-(self.real / self.predicted) + ((1 - self.real) / (1 - self.predicted))) / n


def MeanAbsoluteError(P, Y):
    return np.mean(np.abs(P - Y))

def MeanSquaredError(P, Y):
    return np.mean(np.power(P - Y, 2))

def RootMeanSquaredError(P, Y):
    return np.sqrt(MeanSquaredError(P,Y))

def percent_good(P, Y):
    Y = np.argmax(Y, axis=0)
    P = np.argmax(P, axis=0)
    missed = 0
    for o,p in zip(Y, P):
        if o!=p:
            missed+=1
    accuracy = 100*(P.shape[0]-missed)/P.shape[0]
    return str(accuracy)+' %'

#draft, useless for now
def standardize(x: np.ndarray):
    return np.exp(x-x.max())



if __name__=="__main__":
    final_z = np.array([[0.91, 0.2, 0.2],
                        [0.9, 0.8, 0.8]])
    labels = np.array([[1., 1., 0.],
                       [0., 0., 1.]])
    grads = np.array([[-0.32282815, -0.32282815,  0.17717185],
                     [ 0.32282815 , 0.32282815 ,-0.17717185]])
    SCE = SoftmaxCrossEntropyLoss()
    print(SCE.forward(grads))
    #print(SCE.compute_cost(labels))
    #print(SCE.backward())
    #print(cross_entropy_cost(final_z, labels))