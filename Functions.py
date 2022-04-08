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


class Function:
    def __init__(self, fn: Callable, dfn: Callable=None):
        self.fn = fn
        self.dfn = dfn
    def output(self, x: np.ndarray):
        return self.fn(x)
    def derive(self, x: np.ndarray, 
                labels: np.array=np.array([])):
        if self.dfn==None:
            return derivative(self.fn, domain=x)
        return self.dfn(x)


#ReLu, tanh, sigmoid, weight evaluation, matrix operations, bias addition
ReLu = Function(lambda x: np.maximum(0,x), lambda x: 1.0*(x>0))


Sigmoid = Function(lambda x: 1.0/(1.0+np.exp(-x)), lambda x:(1.0/(1.0+np.exp(-x)))*(1- 1.0/(1.0+np.exp(-x))))


Tanh = Function(lambda x:(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)),
                lambda x:1-((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2)

Id = Function(lambda x: x, lambda x: 1)

class LossFunction(Function):
    def __init__(self, fn: Callable, dfn: Callable):
        self.fn = fn
        self.dfn = dfn
    def derive(self, x: np.ndarray, labels: np.array)->np.array:
        assert(x.shape == labels.shape),\
            f"""different shapes: predictions:{x.shape}
                                  labels     :{labels.shape}"""
        return self.dfn(x, labels)


#softmax per column
def s(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

Softmax = LossFunction(s,
    lambda predictions, labels: (s(predictions)-labels)/len(predictions))

#cross entropy cost for mini-batch
def cross_entropy_cost(predictions: np.array, labels: np.array) -> float:
    #assert same shape
    assert(predictions.shape==labels.shape),\
            f"""different shapes: predictions:{predictions.shape}
                                  labels     :{labels.shape}"""
    
    #avoid log(0)
    predictions = np.clip(predictions, 1.e-9, 1 - 1.e-9)
    
    #compute cost
    logprobs = labels * np.log(predictions) + (1-labels) * np.log(1 - predictions)
    cost = (-1/float(labels.shape[1])) * np.sum(logprobs)
    
    return cost

def mae(preds: np.array, actuals: np.array):
    '''
    Compute mean absolute error.
    '''
    return np.mean(np.abs(preds - actuals))

def rmse(preds: ndarray, actuals: ndarray):
    '''
    Compute root mean squared error.
    '''
    return np.sqrt(np.mean(np.power(preds - actuals, 2)))

def percent_good(predictions: np.array, observations: np.array):
    assert(predictions.shape==observations.shape),\
        f"""you dun goofed up {predictions.shape}!={observations.shape}"""
    observations = np.argmax(observations, axis=1)
    predictions = np.argmax(predictions, axis=1)
    missed = 0
    for o,p in zip(observations, predictions):
        if o!=p:
            missed+=1
    accuracy = 100*(predictions.shape[0]-missed)/predictions.shape[0]
    return accuracy

#draft, useless for now
def standardize(x: np.ndarray):
    return np.exp(x-x.max())



if __name__=="__main__":
    final_z = np.array([[0.2, 0.2, 0.2],
                        [0.8, 0.8, 0.8]])
    labels = np.array([[1., 1., 0.],
                       [0., 0., 1.]])
    grads = np.array([[-0.32282815, -0.32282815,  0.17717185],
                     [ 0.32282815 , 0.32282815 ,-0.17717185]])
    print(Id.derive(grads))
    #print(cross_entropy_cost(final_z, labels))